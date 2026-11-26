"""Lease the group's GPUs to frozen SGLang scorers, independent of the student's architecture."""
from contextlib import ExitStack
import json
import math
import os
from multiprocessing import Pipe
from pathlib import Path
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from uuid import uuid4

from .sglang_engine import sglang_args


class TeacherRanks:
    def __init__(self, cfg, devices, handles, properties=None):
        self.cfg = cfg
        self.handles = handles
        self.devices = devices
        self.width = cfg.get("inference_tp_size", len(devices))
        self.leaders = range(0, len(devices), self.width)
        self.ranks = []
        self.active_pids = []
        self.host_pids = set()
        self.host_baseline = []
        self.standby_contexts = None
        self.role = None
        self.directory = None
        self.activation_s = {}
        self.required_free_bytes = {}
        self.reference_capacity = {}
        self.checked = set()
        self.images = {}
        self.device_type = tuple((p.get("name"), p.get("major"), p.get("minor")) for p in (properties or []))
        self.preload = cfg.get("preload_teacher_imports", True) and properties is not None
        self.waiting = False
        self.released_at = None
        self.preparation_s = None
        self.timeout = cfg.get("startup_timeout", 1200)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1",
                   MKL_NUM_THREADS="1", TOKENIZERS_PARALLELISM="false")
        try:
            for i, gpu in enumerate(devices):
                rank_env = env.copy()
                if self.preload:
                    first = i - i % self.width
                    spec = dict(devices=devices[first:first+self.width], properties=properties[first:first+self.width],
                                rank=i % self.width)
                    rank_env["DYNAMOPD_TEACHER_PREPARE"] = json.dumps(spec)
                control, child = Pipe()
                try:
                    with (Path(cfg["log_dir"])/f"sglang-teacher-rank-{gpu}.log").open("a") as log:
                        proc = subprocess.Popen([sys.executable, "-m", "DynaMOPD.backends.teacher_rank",
                                                 str(child.fileno())], pass_fds=(child.fileno(),), env=rank_env,
                                                start_new_session=True, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    control.close()
                    raise
                finally:
                    child.close()
                self.ranks.append((control, proc))
            self.await_standby()
        except BaseException:
            self.close()
            raise

    def await_standby(self):
        replies = [self.receive(i) for i in range(len(self.ranks))]
        if not all(r.get("standby") for r in replies):
            raise RuntimeError("teacher ranks did not enter standby")
        if self.preload:
            self.standby_contexts = [r["primary_contexts"] for r in replies]
            if any(self.standby_contexts):
                raise RuntimeError("a CPU-prepared teacher acquired CUDA before its lease")
            if self.released_at is not None:
                self.preparation_s = max(0., max(r["ready_at"] for r in replies) - self.released_at)
        self.waiting = False

    def receive(self, rank):
        link, proc = self.ranks[rank]
        if not link.poll(self.timeout):
            raise TimeoutError(f"timeout waiting for teacher rank {rank} (supervisor {proc.pid})")
        reply = link.recv()
        if isinstance(reply, dict) and "error" in reply:
            raise RuntimeError(f"teacher rank {rank}: {reply['error']}")
        return reply

    def activate(self, role, *, calibrate_memory=False):
        import pynvml
        if self.role is not None:
            raise RuntimeError("teacher lease already active")
        began = time.monotonic()
        if self.waiting:
            self.await_standby()
        prepared_at = time.monotonic()
        before = [pynvml.nvmlDeviceGetMemoryInfo(h).free for h in self.handles]
        host_before = self.compute_pids()
        model = self.cfg["staged"][self.cfg["teachers"][role]]
        image = self.teacher_image(model)
        reference = image.reference() if image else None
        args = sglang_args(self.cfg, self.width, role, model, role)
        if reference is not None:
            args.extend(image.loader_args())
        args.extend(["--skip-tokenizer-init", "--random-seed", str(self.cfg["seed"])])
        run_ids = [uuid4().hex for _ in self.leaders]
        self.directory = TemporaryDirectory(prefix="dmtp-")
        paths = ("tokenizer", "scheduler_input", "detokenizer", "rpc", "metrics")
        # Reserve all lane ports before any TP group starts binding its rendezvous.
        with ExitStack() as sockets:
            lane_ports = []
            for _ in self.leaders:
                sock = sockets.enter_context(socket.socket())
                sock.bind(("127.0.0.1", 0))
                lane_ports.append(sock.getsockname()[1])
        for i, (link, _) in enumerate(self.ranks):
            lane, rank = divmod(i, self.width)
            ports = {p+"_ipc_name": f"ipc://{self.directory.name}/{lane}-{p}" for p in paths}
            ports.update(nccl_port=lane_ports[lane], tokenizer_worker_ipc_name=None)
            if not self.preload:
                link.send("activate")
            # The mesh is local to this worker. Each lease gets a fresh c10d store;
            # a probe-and-close TCP port can be claimed during CPU preparation.
            rendezvous = (Path(self.directory.name) / f"{lane}-rendezvous").as_uri()
            link.send(dict(args=args, rank=rank, ports=ports, run_id=run_ids[lane], rendezvous=rendezvous,
                           devices=self.devices[i-rank:i-rank+self.width]))
        ready = [self.receive(i) for i in range(len(self.ranks))]
        if any(not r.get("ready") or (r["role"], r["model_path"]) != (role, model) for r in ready):
            raise RuntimeError("SGLang teacher mesh did not acknowledge its role")
        self.active_pids = [r["pid"] for r in ready]
        host_after = self.compute_pids()
        new_pids = [sorted(active-prior) for prior, active in zip(host_before, host_after, strict=True)]
        # The executor owns the whole group. Only the newly leased ranks may acquire it here.
        if host_after and not all(new_pids):
            raise RuntimeError("cannot identify the new teacher CUDA context on every leased GPU")
        self.host_baseline = host_before
        self.host_pids = {pid for lane in new_pids for pid in lane}
        self.role = role
        load_s = time.monotonic()-began
        checked_at = time.monotonic()
        load_format = "sharded_state" if reference is not None else "hf"
        checked_key = (model, load_format)
        if reference is not None and checked_key not in self.checked:
            for leader in self.leaders:
                actual = self.probe(leader)
                if len(actual) != len(reference) or any(
                    not math.isfinite(v) or abs(v-r) > .02 for v, r in zip(actual, reference, strict=True)
                ):
                    raise RuntimeError("reloaded teacher shard disagrees with its frozen HF scoring probe")
        if self.cfg.get("check_alignment", True) and checked_key not in self.checked:
            from .checks import check_server
            for leader in self.leaders:
                lane = ScoreLane(self, leader)
                check = check_server(lane)
                path = Path(self.cfg["log_dir"])/f"sglang-{self.devices[leader]}-{role}.check.json"
                path.write_text(json.dumps(check, indent=2))
        self.checked.add(checked_key)
        alignment_s = time.monotonic() - checked_at
        build_start = time.monotonic()
        built, build_error = False, None
        if image and reference is None and calibrate_memory:
            try:
                built = image.build(self.save_lane, lambda: self.probe(0))
            except OSError as exc:
                build_error = str(exc)
        build_s = time.monotonic() - build_start if built or build_error else 0.
        capacity = [r["max_total_num_tokens"] for r in ready]
        if calibrate_memory or role not in self.reference_capacity:
            # SGLang budgets weights and KV as a fraction of available memory at launch.
            after = [pynvml.nvmlDeviceGetMemoryInfo(h).free for h in self.handles]
            fraction = self.cfg.get("mem_fraction_static", .36)
            self.reference_capacity[role] = capacity
            self.required_free_bytes[role] = [
                min(free, math.ceil(max(0, free-left) / fraction)) + self.cfg["memory_margin"]
                for free, left in zip(before, after, strict=True)
            ]
        self.activation_s[role] = load_s - (prepared_at-began)
        return dict(teacher_nvml_pids=new_pids, standby_cuda_contexts=self.standby_contexts,
                    teacher_kv_capacity=capacity, teacher_required_free_bytes=self.required_free_bytes[role],
                    teacher_rank_load_s=load_s, standby_wait_s=prepared_at-began, teacher_alignment_s=alignment_s,
                    teacher_prepare_s=self.preparation_s,
                    teacher_load_format=load_format, teacher_image_build_s=build_s, teacher_image_built=built,
                    teacher_image_write_error=build_error,
                    teacher_rank_pids=list(self.active_pids),
                    teacher_rank_timings=[{key: r[key] for key in ("pid", "native_import_s", "scheduler_init_s",
                                                                    "prepared_import_s", "lease_gate") if key in r}
                                          for r in ready])

    def teacher_image(self, model):
        if not self.cfg.get("teacher_rank_images", False):
            return None
        if model not in self.images:
            from .teacher_image import TeacherImage
            config = json.loads((Path(model) / "config.json").read_text())
            # The validated layout here is the dense unquantized Qwen2 scorer.
            supported = config.get("architectures") == ["Qwen2ForCausalLM"] and not config.get("quantization_config")
            self.images[model] = (TeacherImage(self.cfg["state_dir"], model, self.width, self.device_type,
                                               source_id=Path(model).name) if supported else None)
        return self.images[model]

    def save_lane(self, directory):
        for link, _ in self.ranks[:self.width]:
            link.send(("save_image", {"directory": directory}))
        # Consume every rank reply, including failures, before reusing the pipes.
        reports = [self.receive(rank) for rank in range(self.width)]
        for rank, report in enumerate(reports):
            if report.get("saved_rank") != rank:
                raise RuntimeError("unexpected teacher image writer")
            if "write_error" in report:
                raise OSError(report["write_error"])

    def probe(self, leader):
        ids = [758, 11, 220, 17, 10, 18, 25, 12]
        (output,) = ScoreLane(self, leader).post("generate", dict(input_ids=[ids], logprob_start_len=[0]))
        entries = output["meta_info"]["input_token_logprobs"]
        if [e[1] for e in entries] != ids or entries[0][0] is not None:
            raise RuntimeError("misaligned frozen teacher image probe")
        values = [e[0] for e in entries[1:]]
        if not all(map(math.isfinite, values)):
            raise RuntimeError("nonfinite frozen teacher image probe")
        return values

    def release(self):
        if self.role is None:
            raise RuntimeError("no active teacher lease")
        start = time.monotonic()
        # A helper can acquire CUDA during scoring, after the initial ready receipt.
        current = self.compute_pids()
        late_pids = {pid for before, active in zip(self.host_baseline, current, strict=True)
                     for pid in active-before} - self.host_pids
        for link, _ in self.ranks:
            link.send(("release", {}))
        receipts = [self.receive(i) for i in range(len(self.ranks))]
        pids = [r.get("pid") for r in receipts]
        if (pids != self.active_pids or
                any(not r.get("released") or r.get("exit_code") != 0 for r in receipts)):
            raise RuntimeError("teacher GPU rank did not exit cleanly")
        helpers = [pid for r in receipts for pid in r["helper_pids"]]
        host_pids = sorted(self.host_pids | late_pids)
        nvml_verified = bool(host_pids) and self.wait_for_gpu_release(set(host_pids))
        self.host_pids.clear()
        self.host_baseline = []
        self.active_pids = []
        self.waiting = bool(self.preload)
        self.released_at = time.monotonic()
        self.role = None
        self.directory.cleanup()
        self.directory = None
        return dict(released_teacher_pids=pids, released_teacher_nvml_pids=host_pids, late_teacher_nvml_pids=sorted(late_pids),
                    reaped_teacher_helpers=helpers,
                    teacher_gpu_ranks_after_release=len(self.active_pids),
                    release_nvml_checked=nvml_verified, teacher_prepare_enabled=self.preload,
                    teacher_rank_release_s=time.monotonic()-start)

    def compute_pids(self):
        import pynvml
        try:
            return [{p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(h)} for h in self.handles]
        except (pynvml.NVMLError_NotSupported, pynvml.NVMLError_FunctionNotFound, pynvml.NVMLError_NoPermission):
            return []

    def wait_for_gpu_release(self, exited_pids):
        import pynvml

        # A completed waitpid is mandatory on all drivers. Where NVML exposes
        # compute PIDs, also wait for the driver to withdraw their contexts.
        deadline = time.monotonic() + self.timeout
        while True:
            try:
                still_on_device = {p.pid for handle in self.handles
                                   for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                                   if p.pid in exited_pids}
            except (pynvml.NVMLError_NotSupported, pynvml.NVMLError_FunctionNotFound,
                    pynvml.NVMLError_NoPermission):
                return False
            if not still_on_device:
                return True
            if time.monotonic() >= deadline:
                raise TimeoutError(f"teacher processes still reported on GPU: {still_on_device}")
            time.sleep(.02)

    def lane(self, seed):
        return seed // self.cfg["groups"] % len(self.leaders)

    def flush_score_lane(self, lane):
        leader = self.leaders[lane]
        self.ranks[leader][0].send(("flush", {}))
        self.receive(leader)

    def batch(self, items, params, **options):
        if self.role is None or params.get("max_new_tokens") != 0:
            raise RuntimeError("need an active prefill-only teacher")
        positions = [[] for _ in self.leaders]
        for i, row in enumerate(items):
            positions[self.lane(row["seed"])].append(i)
        self.last_batch_sizes = [len(lane) for lane in positions]
        output = [None]*len(items)
        request_ids = [f"{x['rid']}/{uuid4().hex}" for x in items]
        for leader, indices in zip(self.leaders, positions, strict=True):
            if indices:
                body = dict(input_ids=[items[i]["prompt"]+items[i]["tokens"] for i in indices],
                            rid=[request_ids[i] for i in indices],
                            logprob_start_len=[options["logprob_start_len"][i] for i in indices],
                            top_logprobs_num=options["top_logprobs_num"])
                self.ranks[leader][0].send(("score", body))
        for leader, indices in zip(self.leaders, positions, strict=True):
            if indices:
                for i, result in zip(indices, self.receive(leader), strict=True):
                    meta = result["meta_info"]
                    if meta["id"] != request_ids[i] or meta["weight_version"] != self.role:
                        raise RuntimeError("mismatched teacher receipt")
                    output[i] = (items[i], result, meta)
        return output

    def post(self, endpoint, payload):
        if not endpoint.startswith("flush_cache"):
            raise ValueError(endpoint)
        for leader in self.leaders:
            self.ranks[leader][0].send(("flush", {}))
        return [self.receive(leader) for leader in self.leaders]

    def close(self):
        # Also kills partial TP launches and unresponsive children on an aborted job.
        for _, proc in self.ranks:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        for link, proc in self.ranks:
            proc.wait()
            link.close()
        self.ranks.clear()
        if self.directory is not None:
            self.directory.cleanup()
            self.directory = None


class ScoreLane:
    """Run the usual cold/warm chunk-boundary probe on one physical TP mesh."""
    name = "teacher"

    def __init__(self, pool, leader):
        self.pool, self.leader = pool, leader

    def post(self, endpoint, body):
        link = self.pool.ranks[self.leader][0]
        if endpoint.startswith("flush_cache"):
            link.send(("flush", {}))
        else:
            ids = body["input_ids"]
            link.send(("score", dict(input_ids=ids, rid=[uuid4().hex for _ in ids],
                                     logprob_start_len=body.get("logprob_start_len", [0]*len(ids)))))
        return self.pool.receive(self.leader)
