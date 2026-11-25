"""Driver-only experiment logging. Remote SDKs are imported only when selected."""
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from importlib import import_module
from inspect import signature
import json
import logging
import math
import traceback

logger = logging.getLogger(__name__)


def options(cfg):
    names = cfg.get("track", [])
    if not isinstance(names, (list, tuple)) or any(n not in ("wandb", "swanlab") for n in names):
        raise ValueError("track must list wandb and/or swanlab")
    if cfg.get("tracking_mode", "offline") not in ("offline", "online"):
        raise ValueError("tracking_mode must be online or offline")
    interval = cfg.get("gpu_sample_interval", 1.)
    if type(interval) not in (float, int) or not math.isfinite(interval) or interval < 0:
        raise ValueError("gpu_sample_interval must be nonnegative and finite")
    return list(dict.fromkeys(names))


class RunLog:
    def __init__(self, log, metrics, clients):
        self.log, self.metrics, self.clients = log, metrics, clients

    def note(self, message):
        print(datetime.now(timezone.utc).isoformat(timespec="seconds"), message, file=self.log, flush=True)

    def record(self, step, values):
        self.metrics.write(json.dumps(dict(step=step, **values), allow_nan=False) + "\n")
        self.note(f"round {step} committed: {values['tokens']} effective tokens, "
                  f"{values['round_s']:.3f} s, {values['effective_tokens_per_s']:.1f} token/s")
        for name, run in list(self.clients.items()):
            try:
                run.log(values, step=step)
            except Exception as exc:
                self.note(f"{name} logging failed: {exc}")
                logger.warning("%s logging failed: %s", name, exc)
                del self.clients[name]


@contextmanager
def open_run_log(out, cfg, simulate):
    names = options(cfg)
    clients, finishers = {}, []
    with ExitStack() as stack:
        text = stack.enter_context((out / "driver.log").open("a", buffering=1))
        metrics = stack.enter_context((out / "metrics.jsonl").open("x", buffering=1))
        log = RunLog(text, metrics, clients)
        log.note("starting simulation" if simulate else "starting training")
        failure = None
        try:
            keys = ("algorithm", "policy", "engine", "groups", "group_size", "inference_tp_size", "batch_size",
                    "rounds", "microbatch", "max_new_tokens", "quantum", "lr", "seed")
            public = {k: cfg[k] for k in keys if k in cfg}
            public["simulate"] = simulate
            for name in names:
                try:
                    sdk = import_module(name)
                except ModuleNotFoundError as exc:
                    raise RuntimeError(f"install the {name} extra to enable {name} tracking") from exc
                common = dict(project=cfg.get("tracking_project", "DynaMOPD"), config=public,
                              mode=cfg.get("tracking_mode", "offline"))
                run_name = cfg.get("run_name") or out.name
                if name == "wandb":
                    run = sdk.init(**common, name=run_name, dir=str(out), save_code=False)
                    finishers.append((name, run.finish))
                else:
                    run = sdk.init(**common, experiment_name=run_name, logdir=str(out / "swanlog"))
                    finishers.append((name, sdk.finish))
                clients[name] = run
            yield log
        except BaseException as exc:
            failure = exc
            log.note("run failed")
            traceback.print_exc(file=text)
            raise
        finally:
            for name, finish in reversed(finishers):
                try:
                    if name == "wandb":
                        finish(exit_code=int(failure is not None))
                    elif failure is not None and "state" in signature(finish).parameters:
                        finish(state="crashed", error=type(failure).__name__)
                    elif failure is not None and "error" in signature(finish).parameters:
                        finish(error=type(failure).__name__)
                    else:
                        finish()
                except Exception as exc:
                    log.note(f"{name} finalization failed: {exc}")
            if failure is None:
                log.note("run finished")
