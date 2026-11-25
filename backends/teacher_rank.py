"""Private SGLang score rank. The supervisor stays on CPU for its entire lifetime."""
import os
from importlib import import_module
import sys
import time
import traceback
from multiprocessing.connection import Connection


def score_windows(inputs, *, kv_tokens, prefill_tokens, max_requests, page_size):
    """Partition cold prefixes into batches that a single SGLang lane can admit."""
    start = kv_used = prefill_used = 0
    for index, ids in enumerate(inputs):
        length = len(ids)
        if length >= kv_tokens:
            raise ValueError(f"teacher prefix of {length} tokens exceeds usable KV capacity {kv_tokens - 1}")
        pages = (length + page_size - 1) // page_size
        rounded = pages * page_size
        # SGLang uses a strict prefill limit after the first request and reserves
        # one extra allocator page per admitted sequence. Count full prefixes so
        # correctness is independent of whether the radix cache still holds them.
        if index > start and (index - start >= max_requests or kv_used + rounded >= kv_tokens
                              or prefill_used + rounded >= prefill_tokens):
            yield slice(start, index)
            start, kv_used, prefill_used = index, 0, 0
        kv_used += rounded + page_size
        prefill_used += rounded
    if start < len(inputs):
        yield slice(start, len(inputs))


def score_batch(sender, receiver, body, role, scheduler):
    from sglang.srt.managers.io_struct import (
        AbortReq, BatchTokenIDOutput, BatchTokenizedGenerateReqInput, TokenizedGenerateReqInput,
    )
    from sglang.srt.sampling.sampling_params import SamplingParams

    requests = []
    for rid, ids, start in zip(body["rid"], body["input_ids"], body["logprob_start_len"], strict=True):
        params = SamplingParams(max_new_tokens=0, temperature=1.)
        params.normalize(None)
        params.verify(scheduler.model_config.vocab_size)
        requests.append(TokenizedGenerateReqInput(
            rid=rid, input_text="", input_ids=ids, mm_inputs=None, sampling_params=params,
            return_logprob=True, logprob_start_len=start, top_logprobs_num=body.get("top_logprobs_num", 0),
            token_ids_logprob=None, stream=False,
        ))
    output = {}
    windows = score_windows(body["input_ids"], kv_tokens=scheduler.max_total_num_tokens,
                            prefill_tokens=scheduler.max_prefill_tokens, max_requests=scheduler.max_running_requests,
                            page_size=scheduler.page_size)
    for window in windows:
        batch = requests[window]
        if len(batch) != len(requests):
            print(f"teacher {role} admission: {len(batch)}/{len(requests)} requests, "
                  f"{sum(len(req.input_ids) for req in batch)} input tokens", flush=True)
        sender.send_pyobj(BatchTokenizedGenerateReqInput(batch=batch))
        pending = {request.rid for request in batch}
        while pending:
            message = receiver.recv_pyobj()
            if isinstance(message, AbortReq):
                raise RuntimeError(f"SGLang aborted teacher request {message.rid}")
            if not isinstance(message, BatchTokenIDOutput):
                raise RuntimeError(f"unexpected SGLang score reply {type(message).__name__}")
            for i, rid in enumerate(message.rids):
                reason = message.finished_reasons[i]
                if rid not in pending or not reason or reason["type"] not in ("length", "stop"):
                    raise RuntimeError(f"invalid teacher completion for {rid}: {reason}")
                meta = dict(id=rid, weight_version=role, finish_reason=reason, cached_tokens=message.cached_tokens[i],
                            input_token_logprobs=list(zip(message.input_token_logprobs_val[i],
                                                          message.input_token_logprobs_idx[i], strict=True)))
                if body.get("top_logprobs_num"):
                    meta["input_top_logprobs"] = [
                        [(lp, token, None) for lp, token in zip(vals, indices, strict=True)] if vals is not None else None
                        for vals, indices in zip(message.input_top_logprobs_val[i],
                                                 message.input_top_logprobs_idx[i], strict=True)
                    ]
                output[rid] = {"meta_info": meta}
                pending.remove(rid)
    return [output[rid] for rid in body["rid"]]


def serve_commands(link, scheduler, ports, leader, timings):
    """Only TP rank zero submits batches. SGLang broadcasts them over its own TP group."""
    import zmq
    from sglang.srt.managers.io_struct import FlushCacheReqInput, FlushCacheReqOutput

    try:
        if leader:
            context = zmq.Context()
            sender = context.socket(zmq.PUSH)
            receiver = context.socket(zmq.PULL)
            sender.bind(ports.scheduler_input_ipc_name)
            receiver.bind(ports.tokenizer_ipc_name)
        role = scheduler.server_args.weight_version
        link.send(dict(ready=True, role=role, model_path=scheduler.model_config.model_path,
                       pid=os.getpid(), **timings, **scheduler.get_init_info()))
        while True:
            action, body = link.recv()
            if action == "release":
                # The supervisor reaps this process before acknowledging release.
                os._exit(0)
            if action == "save_image":
                # Only the representative TP lane receives this command, on
                # each of its ranks, before it is admitted to score requests.
                from safetensors import SafetensorError
                result = {"saved_rank": scheduler.tp_rank}
                try:
                    scheduler.tp_worker.model_runner.save_sharded_model(body["directory"], max_size=512 * 2**20)
                except (OSError, SafetensorError) as exc:
                    # A full/unwritable host volume does not invalidate the HF-loaded scorer.
                    result["write_error"] = str(exc)
            elif action == "score" and leader:
                result = score_batch(sender, receiver, body, role, scheduler)
            elif action == "flush" and leader:
                sender.send_pyobj(FlushCacheReqInput(timeout_s=10))
                result = receiver.recv_pyobj()
                if not isinstance(result, FlushCacheReqOutput) or not result.success:
                    raise RuntimeError(f"SGLang could not flush teacher KV: {result}")
                result = {"success": True}
            else:
                raise ValueError(f"invalid teacher rank operation {action}")
            link.send(result)
    except BaseException:
        try:
            link.send({"error": traceback.format_exc()})
        finally:
            os._exit(1)


def active_rank(link, cfg=None, prepared=None):
    import threading

    if cfg is None:
        cfg = link.recv()
    start = prepared["lease_start"] if prepared else time.monotonic()
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(cfg["devices"])
    from sglang.srt.entrypoints.engine import _set_envs_and_config
    from sglang.srt.managers.scheduler import Scheduler, configure_scheduler
    from sglang.srt.server_args import PortArgs, prepare_server_args

    args = prepare_server_args(cfg["args"])
    _set_envs_and_config(args)
    os.environ["SGLANG_RUN_ID"] = cfg["run_id"]
    os.environ["SGLANG_DISTRIBUTED_INIT_METHOD_OVERRIDE"] = cfg["rendezvous"]
    args.check_server_args()
    rank = cfg["rank"]
    dp = configure_scheduler(args, rank, 0, 0, 0, 0, None)
    ports = PortArgs(**cfg["ports"])
    imported = time.monotonic()
    scheduler = Scheduler(args, ports, gpu_id=rank, tp_rank=rank, moe_ep_rank=0, pp_rank=0,
                          attn_cp_rank=0, moe_dp_rank=0, dp_rank=dp)
    timings = dict(native_import_s=imported-start, scheduler_init_s=time.monotonic()-imported)
    if prepared:
        timings.update(prepared_import_s=prepared["seconds"], lease_gate=prepared["gate"])
    threading.Thread(target=serve_commands, args=(link, scheduler, ports, rank == 0, timings),
                     daemon=True).start()
    scheduler.run_event_loop()
    raise RuntimeError("SGLang teacher scheduler exited unexpectedly")


def standby_context_count():
    """Query this process's primary CUDA contexts without creating or retaining one."""
    import ctypes

    driver = ctypes.CDLL("libcuda.so.1")
    def checked(result):
        if result:
            raise RuntimeError(f"CUDA driver error {result} checking teacher standby")
    checked(driver.cuInit(0))
    count = ctypes.c_int()
    checked(driver.cuDeviceGetCount(ctypes.byref(count)))
    active = 0
    for index in range(count.value):
        device = ctypes.c_int()
        flags = ctypes.c_uint()
        state = ctypes.c_int()
        checked(driver.cuDeviceGet(ctypes.byref(device), index))
        checked(driver.cuDevicePrimaryCtxGetState(device, ctypes.byref(flags), ctypes.byref(state)))
        active += bool(state.value)
    return active


def prepared_rank(link):
    """Stop at the first device operation, or at Scheduler construction if imports stay on CPU."""
    import json
    from threading import Lock
    from types import SimpleNamespace
    import torch

    spec = json.loads(os.environ["DYNAMOPD_TEACHER_PREPARE"])
    class DeviceProperties(SimpleNamespace):
        def __getattr__(self, field):
            ready_time[0] = time.monotonic()
            start_lease("device_property")
            return getattr(real_props(self.index), field)

    props = [DeviceProperties(**dict(p, index=i)) for i, p in enumerate(spec["properties"])]
    begin = time.monotonic()
    real_init = torch.cuda._lazy_init
    real_current = torch.cuda.current_device
    real_props = torch.cuda.get_device_properties
    lease = {}

    gate_lock = Lock()

    def start_lease(gate):
        with gate_lock:
            acquire_lease(gate)

    def acquire_lease(gate):
        if lease:
            return
        contexts = standby_context_count()
        if contexts or torch.cuda.is_initialized():
            raise RuntimeError("teacher imported a CUDA context before receiving its lease")
        link.send(dict(standby=True, pid=os.getpid(), primary_contexts=contexts))
        cfg = link.recv()
        if tuple(cfg["devices"]) != tuple(spec["devices"]) or cfg["rank"] != spec["rank"]:
            raise RuntimeError("teacher preparation and lease disagree on device placement")
        stamp = time.monotonic()
        lease.update(cfg=cfg, lease_start=stamp, seconds=ready_time[0]-begin, gate=gate)
        torch.cuda._lazy_init, torch.cuda.current_device, torch.cuda.get_device_properties = real_init, real_current, real_props

    def enter_cuda(*args, **kwargs):
        ready_time[0] = time.monotonic()
        start_lease("first_cuda_use/" + sys._getframe(1).f_globals.get("__name__", "unknown"))
        return real_init(*args, **kwargs)

    def device_props(device=None):
        if lease:
            return real_props(device)
        index = torch.device(device).index if isinstance(device, (str, torch.device)) else device
        return props[spec["rank"] if index is None else index]

    ready_time = [begin]
    # Import-time hardware selection uses properties from the already-running local FSDP ranks.
    # Real allocations and capability queries during execution use the native PyTorch APIs.
    torch.cuda._lazy_init = enter_cuda
    torch.cuda.current_device = lambda: real_current() if lease else spec["rank"]
    torch.cuda.get_device_properties = device_props
    # Import scheduler dependencies while the lease gate protects CUDA initialization.
    for module in ("sglang.srt.entrypoints.engine", "sglang.srt.managers.scheduler",
                   "sglang.srt.server_args", "sglang.srt.models.registry"):
        import_module(module)
    ready_time[0] = time.monotonic()
    start_lease("scheduler_ready")
    active_rank(link, lease["cfg"], lease)


def supervise_prepared(link):
    import ctypes
    import json
    import subprocess

    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0):
        raise OSError(ctypes.get_errno(), "cannot become teacher rank subreaper")
    spec = json.loads(os.environ["DYNAMOPD_TEACHER_PREPARE"])
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(spec["devices"]), PYTORCH_NVML_BASED_CUDA_CHECK="1")
    while True:
        child = subprocess.Popen([sys.executable, "-m", __spec__.name, str(link.fileno()), "prepare"],
                                 pass_fds=(link.fileno(),), env=env)
        status = child.wait()
        helpers = reap_descendants()
        link.send(dict(released=True, pid=child.pid, exit_code=status, helper_pids=helpers))
        if status:
            return

def run_active(link):
    try:
        active_rank(link)
    except BaseException:
        try:
            link.send({"error": traceback.format_exc()})
        finally:
            os._exit(1)


def reap_descendants():
    # The rank may have started compiler/loader helpers. Linux reparents any
    # survivors to this subreaper on rank exit. Repeat for nested helpers.
    import psutil
    supervisor = psutil.Process()
    reaped = []
    while True:
        try:
            pid, _ = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return reaped
        if pid:
            reaped.append(pid)
        else:
            for child in supervisor.children():
                try:
                    child.kill()
                except psutil.NoSuchProcess:
                    pass
            time.sleep(.01)


def supervise(link):
    # Import only CPU-neutral dependencies here. SGLang's CUDA/platform probes
    # and model-specific imports run in the leased child with its actual devices.
    import ctypes
    import_module("numpy")
    import torch
    import_module("safetensors.torch")

    if os.environ["CUDA_VISIBLE_DEVICES"] or torch.cuda.is_initialized():
        raise RuntimeError("teacher supervisor requires CPU-only visibility")
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        raise OSError(ctypes.get_errno(), "cannot become teacher rank subreaper")
    link.send({"standby": True})
    while link.recv() == "activate":
        # A threaded import or an earlier CUDA runtime probe can poison fork.
        # Use a clean interpreter in either case. The supervisor never runs CUDA.
        clean_exec = len(os.listdir("/proc/self/task")) != 1
        child = os.fork() if not clean_exec else None
        if child is None:
            import subprocess
            proc = subprocess.Popen([sys.executable, "-m", __spec__.name, str(link.fileno()), "active"],
                                    pass_fds=(link.fileno(),))
            child, status = proc.pid, proc.wait()
        elif child == 0:
            if torch.cuda._is_in_bad_fork():
                os.set_inheritable(link.fileno(), True)
                os.execv(sys.executable, [sys.executable, "-m", __spec__.name, str(link.fileno()), "active"])
            run_active(link)
        else:
            _, status = os.waitpid(child, 0)
            status = os.waitstatus_to_exitcode(status)
        helpers = reap_descendants()
        link.send(dict(released=True, pid=child, exit_code=status, helper_pids=helpers))


if __name__ == "__main__":
    with Connection(int(sys.argv[1])) as connection:
        if sys.argv[2:] == ["prepare"]:
            try:
                prepared_rank(connection)
            except BaseException:
                connection.send({"error": traceback.format_exc()})
                raise
        elif "DYNAMOPD_TEACHER_PREPARE" in os.environ:
            supervise_prepared(connection)
        elif len(sys.argv) > 2:
            run_active(connection)
        else:
            supervise(connection)
