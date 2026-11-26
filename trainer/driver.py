"""One job namespace, one controller and one optimizer commit per round."""
from contextlib import ExitStack, closing, nullcontext
from dataclasses import asdict
import fcntl
import json
import logging
import math
import os
from pathlib import Path
import tempfile
import time

from ..scheduler.controller import Controller, Trace
from .config import build_job
from .tracking import open_run_log
from .data import get_rows, load_data, replay_inputs, save_samples, simulated_inputs, vocabulary

ROOT = Path(__file__).resolve().parents[1]


def fit(config: dict, *, output, data_path=None, state_dir=None, simulate=False, replay=None,
        resume_version=0, on_round=None) -> list[dict]:
    """Run from a JSON-compatible configuration. on_round receives each committed round summary."""
    cfg, executor, policy, algorithm = build_job(config, simulate=simulate, replay=replay, resume_version=resume_version)
    interval = cfg.get("checkpoint_interval", 1)
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    state = Path(state_dir or out/"weights").resolve()
    cfg.update(state_dir=str(state), log_dir=str(out), final_version=resume_version+cfg["rounds"])
    tokenizer = common = None
    restored_max_ema = None
    with ExitStack() as stack:
        trace = stack.enter_context(closing(Trace(out/"events.jsonl")))
        backend = None
        results = []
        try:
            log = stack.enter_context(open_run_log(out, cfg, simulate))
            if simulate:
                data = simulated_inputs(cfg)
            else:
                state.mkdir(parents=True, exist_ok=True)
                lock = stack.enter_context((state/".controller.lock").open("a"))
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                published = [int(d.name[1:]) for d in state.glob("v*") if d.is_dir() and d.name[1:].isdigit()]
                if max(published, default=0) != resume_version:
                    raise ValueError("resume_version must match the latest published checkpoint")
                if resume_version:
                    checkpoint = state / f"v{resume_version}"
                    executor.validate_checkpoint(checkpoint)
                    scheduler_state = checkpoint / "scheduler.json"
                    if cfg["max_length_ema_alpha"] is not None and scheduler_state.is_file():
                        restored_max_ema = json.loads(scheduler_state.read_text()).get("max_length_ema")
                        if restored_max_ema is not None and (type(restored_max_ema) not in (int, float)
                                or not math.isfinite(restored_max_ema) or restored_max_ema <= 0):
                            raise ValueError(f"invalid max_length_ema in {scheduler_state}")
                if interval != 1 and cfg.get("weight_sync", "disk") == "disk":
                    cfg["export_dir"] = stack.enter_context(tempfile.TemporaryDirectory(prefix=".exports-", dir=state))
                tokenizer, common = vocabulary(cfg)
                data = load_data(Path(data_path or ROOT/"data/native-smoke.jsonl"))

            trace.emit("config", backend="simrl" if simulate else cfg["engine"], policy=cfg["policy"],
                       visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"), config=cfg)
            start = time.monotonic()
            backend = executor(cfg)
            placements = backend.placements
            if getattr(backend, "teacher_preflight", None):
                trace.emit("teacher_preflight", groups=backend.teacher_preflight)
            if getattr(backend,"tp_setup",None):
                trace.emit("wide_mesh_ready",meshes=backend.tp_setup)
            if [p.group for p in placements] != list(range(cfg["groups"])):
                raise ValueError("executor placement must describe every logical group in order")
            devices = [(p.node,d) for p in placements for d in p.devices]
            if any(not p.devices for p in placements) or len(devices) != len(set(devices)):
                raise ValueError("physical GPU groups must be nonempty and disjoint on each node")
            trace.emit("executor_ready", t=backend.now(), startup_s=0 if simulate else time.monotonic()-start,
                       placements=[asdict(p) for p in placements])
            sample_interval = cfg.get("gpu_sample_interval", 1.)
            monitor = nullcontext()
            if not simulate and sample_interval and hasattr(backend, "gpu_monitor"):
                monitor = backend.gpu_monitor(out / "gpu-telemetry", sample_interval)
            log.note(f"executor ready: {len(placements)} GPU groups")
            with monitor:
                planner_cfg = cfg
                if simulate and cfg.get("first_round_calibration"):
                    # The simulated hardware keeps its own timing model. Its
                    # hidden parameters must not leak into the online planner.
                    planner_cfg = dict(cfg, costs={}, sim_worker_profile={}, sim_decode_roofline={})
                controller = Controller(planner_cfg, backend, policy, trace)
                calibration_stream = None
                if cfg.get("first_round_calibration"):
                    from ..scheduler.calibration import update_profile
                    calibration_stream = stack.enter_context((out/"events.jsonl").open())
                controller.max_length_ema = restored_max_ema
                for version in range(resume_version, cfg["final_version"]):
                    if replay:
                        data = replay_inputs(Path(replay)/f"samples-v{version}.jsonl", cfg)
                    rows = get_rows(cfg, tokenizer, common, data, version, simulate, algorithm)
                    if simulate:
                        backend.targets = {r.rid: data[i]["target"] for i,r in enumerate(rows)}
                    begin = backend.now()
                    summary = dict(version=version, **controller.run_round(rows, version))
                    elapsed = backend.now() - begin
                    if calibration_stream is not None:
                        events = [json.loads(line) for line in calibration_stream]
                        update_profile(controller, [e for e in events if e.get("version") == version])
                    metrics = dict(round_s=elapsed, tokens=summary["tokens"],
                                   effective_tokens_per_s=summary["tokens"] / elapsed if elapsed else 0.,
                                   generated_tokens=sum(len(r.tokens) for r in rows),
                                   eos=sum(r.finish == "eos" for r in rows), trained=sum(r.trained for r in rows),
                                   truncated=sum(r.finish == "length" for r in rows))
                    if not simulate:
                        metrics["loss_per_token"] = controller.round_loss_sum / summary["tokens"]
                        if "grad_norm" in summary:
                            metrics["grad_norm"] = summary["grad_norm"]
                    log.record(summary["published"], metrics)
                    if not simulate:
                        save_samples(out/f"samples-v{version}.jsonl", rows)
                    results.append(summary)
                    if on_round:
                        on_round(summary)
            closing_backend, backend = backend, None
            closing_backend.close()
            trace.emit("run_finished")
        except BaseException as exc:
            trace.emit("run_failed", error=str(exc))
            if backend is not None:
                backend.close(failed=True)
            raise
        finally:
            if cfg.get("timeline", True):
                from ..tools.timeline import write_timeline
                try:
                    write_timeline(out)
                except Exception:
                    logging.getLogger(__name__).exception("Could not render GPU timeline for %s", out)
        return results
