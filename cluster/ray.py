"""Ray transport for node-local SGLang TP / FSDP2 groups and shared round checkpoints."""
from contextlib import ExitStack, contextmanager
from concurrent.futures import ThreadPoolExecutor
import logging
import os
from pathlib import Path
import tempfile
from threading import Condition
import time
from uuid import uuid4

import ray
from ray.util import get_node_ip_address, placement_group, remove_placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from ..backends.elastic_profile import TailProfile
from ..backends.executor import GroupExecutor
from ..workers.group import GroupWorker
from ..workers.gpu_monitor import GpuMonitor
from ..workers.model import stage_models
from ..workers.node import NodeRuntime, uses_wide_mesh
from .topology import allocations, node_shape

logger = logging.getLogger(__name__)


class RayNode:
    """One actor owns a placement bundle. Disjoint group leases can run concurrently."""

    def __init__(self, allocation, cfg, markers):
        for path, nonce in markers:
            if Path(path).read_text() != nonce:
                raise RuntimeError(f"allocation {allocation.groups} cannot see shared directory at {path}")
        node = get_node_ip_address()
        self.placements = allocation.place(node, os.environ["CUDA_VISIBLE_DEVICES"].split(","), cfg["group_size"])
        logs = Path(cfg["log_dir"]) / f"groups-{allocation.groups[0]}"
        logs.mkdir(exist_ok=True)
        self.cfg = dict(cfg, log_dir=str(logs))
        self.runtime = self.store = self.monitor = None
        self.model_cache = None
        self.condition = Condition()
        self.inflight = 0
        self.closing = False

    def describe(self):
        return self.placements

    def clock(self):
        return time.monotonic()

    def start_monitor(self, directory, interval, offset, uncertainty):
        self.monitor = GpuMonitor(directory, self.placements, lambda: time.monotonic() + offset, interval, uncertainty)
        self.monitor.__enter__()

    def stop_monitor(self):
        if self.monitor:
            self.monitor.__exit__(None, None, None)
            self.monitor = None

    def start_store(self):
        from torch.distributed import TCPStore
        node = self.placements[0].node
        self.store = TCPStore(node, 0, None, True, wait_for_workers=False)
        return node, self.store.port

    @contextmanager
    def activity(self):
        with self.condition:
            if self.closing:
                raise RuntimeError("node allocation is closing")
            self.inflight += 1
        try:
            yield
        finally:
            with self.condition:
                self.inflight -= 1
                self.condition.notify_all()

    def prepare(self, profile, staged, run_id, endpoint):
        with self.activity():
            if staged is None:
                cache = self.cfg.get("node_model_cache")
                if cache is None:
                    self.model_cache = tempfile.TemporaryDirectory(prefix="dynamopd-models-")
                    cache = self.model_cache.name
                with ThreadPoolExecutor() as io:
                    staged = stage_models(self.cfg, io, folder=cache)
            self.cfg.update(staged=staged, weight_sync_run_id=run_id, weight_sync_endpoint=endpoint)
            self.runtime = NodeRuntime(self.cfg, self.placements, profile)
            return self.runtime.prepare()

    def execute(self, group, op, version, payload):
        with self.activity():
            begin = time.monotonic()
            value, free = self.runtime.call(group, op, version, payload)
            value.update(worker_s=time.monotonic() - begin, node=self.placements[0].node)
            return value, free

    def wide(self, action):
        with self.activity():
            begin = time.monotonic()
            value, free = self.runtime.wide(action)
            value.update(worker_s=time.monotonic() - begin, node=self.placements[0].node)
            return value, free

    def close(self, failed):
        self.stop_monitor()
        with self.condition:
            self.closing = True
            while self.inflight:
                if failed and self.runtime:
                    self.runtime.abort()
                self.condition.wait(timeout=1)
        if self.runtime:
            if failed:
                self.runtime.abort()
            self.runtime.close(failed)
        if self.model_cache is not None:
            self.model_cache.cleanup()


class RayExecutor(GroupExecutor):
    """Reserve an entire experiment before any inference engine or FSDP rank starts."""
    supervision = GroupWorker.supervision
    supports_trainer_replicas = True

    def __init__(self, cfg):
        width = cfg["group_size"]
        if width < 1 or cfg["temperature"] != 1:
            raise ValueError("SGLang/FSDP2 requires a positive group size and temperature 1")
        options = cfg["cluster"]
        cpus = options.get("cpus_per_group", 2)
        plan = allocations(cfg)
        shaped = bool(node_shape(options))
        super().__init__(cfg["groups"])
        self.actors, self.owners = [], {}
        self.pg = None
        self.owns_connection = not ray.is_initialized()
        timeout = cfg.get("startup_timeout", 1200)
        self.timeout = cfg.get("action_timeout_s", 1800)
        try:
            if self.owns_connection:
                ray.init(address=options.get("address", "auto"))
            bundles = [a.resources(width, cpus) for a in plan]
            self.pg = placement_group(bundles, strategy="STRICT_SPREAD" if shaped else "SPREAD")
            try:
                ray.get(self.pg.ready(), timeout=options.get("placement_timeout_s", 180))
            except ray.exceptions.GetTimeoutError as exc:
                capacities = [(n["NodeManagerAddress"], {k: n["Resources"].get(k, 0) for k in ("CPU","GPU")})
                              for n in ray.nodes() if n["Alive"]]
                raise RuntimeError(f"Ray reservation timed out: requested {bundles}, live node capacities {capacities}") from exc
            actor = ray.remote(max_restarts=0, max_task_retries=0)(RayNode)
            variables = ("PYTHONPATH", "MOPD_TRACE_DIR", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "TOKENIZERS_PARALLELISM",
                         "HF_HUB_OFFLINE", "HF_HUB_DISABLE_PROGRESS_BARS", "PYTHONUNBUFFERED",
                         "SGLANG_ENABLE_LOGITS_PROCESSER_CHUNK", "SGLANG_LOGITS_PROCESSER_CHUNK_SIZE")
            runtime_env = {"env_vars": {k: os.environ[k] for k in variables if k in os.environ}}
            with ExitStack() as stack:
                markers = []
                for directory in {cfg["state_dir"], cfg["log_dir"], cfg.get("export_dir", cfg["state_dir"])}:
                    file = stack.enter_context(tempfile.NamedTemporaryFile(mode="w", dir=directory))
                    nonce = uuid4().hex
                    file.write(nonce)
                    file.flush()
                    markers.append((file.name, nonce))
                for i, allocation in enumerate(plan):
                    strategy = PlacementGroupSchedulingStrategy(placement_group=self.pg, placement_group_bundle_index=i)
                    count = len(allocation.groups)
                    handle = actor.options(scheduling_strategy=strategy, runtime_env=runtime_env,
                                           num_cpus=cpus * count, num_gpus=width * count,
                                           max_concurrency=count + 1).remote(allocation, cfg, markers)
                    self.actors.append(handle)
                    self.owners.update(dict.fromkeys(allocation.groups, handle))
                descriptions = ray.get([a.describe.remote() for a in self.actors], timeout=timeout)
            self.placements = tuple(p for bundle in descriptions for p in bundle)
            physical = {(p.node, d) for p in self.placements for d in p.devices}
            if (len(physical) != width * cfg["groups"] or
                    (shaped and len({p.node for p in self.placements}) != len(plan))):
                raise RuntimeError(f"overlapping Ray allocations: {self.placements}")
            self.tail_tp = TailProfile(cfg, self.placements) if cfg.get("rollout_tp_expansion") else None
            # Check shared paths and placements before copying checkpoints.
            # Checkpoints and receipts remain shared. Immutable model images used
            # by each teacher switch are read from that actor's local filesystem.
            cfg["weight_sync_run_id"] = uuid4().hex
            endpoint = None
            if cfg.get("weight_sync", "tensor") == "tensor":
                endpoint = ray.get(self.actors[0].start_store.remote(), timeout=timeout)
            preparations = [a.prepare.remote(self.tail_tp, None, cfg["weight_sync_run_id"], endpoint)
                            for a in self.actors]
            ready = ray.get(preparations, timeout=timeout)
            self.tp_setup = [mesh for meshes, _ in ready for mesh in meshes]
            self.teacher_preflight = [entry for _, entries in ready for entry in entries]
            if self.tail_tp:
                self.tail_tp.configure_meshes(self.tp_setup)
        except BaseException:
            self.close(failed=True)
            raise

    @contextmanager
    def gpu_monitor(self, directory, interval):
        # Anchor remote monotonic clocks to the same driver timeline as RPC receipts.
        def start(actor):
            probes = []
            for _ in range(3):
                before = self.now()
                remote = ray.get(actor.clock.remote(), timeout=10)
                after = self.now()
                probes.append((after - before, (before + after) / 2 - remote))
            rtt, offset = min(probes)
            ray.get(actor.start_monitor.remote(str(directory), interval, offset, rtt / 2), timeout=10)

        try:
            futures = [self.pool.submit(start, actor) for actor in self.actors]
            for f in futures:
                try:
                    f.result()
                except Exception as exc:
                    logger.warning("Remote GPU telemetry unavailable: %s", exc)
            yield
        finally:
            try:
                ray.get([actor.stop_monitor.remote() for actor in self.actors], timeout=10)
            except Exception as exc:
                logger.warning("Closing remote GPU telemetry: %s", exc)

    def execute(self, action):
        if uses_wide_mesh(action):
            if self.owners[action.gpu] is not self.owners[action.peer]:
                raise ValueError("a wide TP lease must stay inside one Ray allocation")
            start = self.now()
            ref = self.owners[action.gpu].wide.remote(action)
            value, free = ray.get(ref, timeout=self.timeout)
            return dict(start=start, end=self.now(), value=value, free_bytes=free)
        return super().execute(action)

    def call(self, gpu, kind, version, payload):
        start = self.now()
        ref = self.owners[gpu].execute.remote(gpu, kind, version, payload)
        value, free = ray.get(ref, timeout=self.timeout)
        return dict(start=start, end=self.now(), value=value, free_bytes=free)

    def close(self, failed=False):
        # An in-flight ray.get() may be waiting on another failed role or rank.
        # Give actors a bounded graceful close, then kill before joining RPC threads.
        self.pool.shutdown(wait=not failed, cancel_futures=True)
        errors = []
        try:
            if self.actors:
                closing = [actor.close.remote(failed) for actor in self.actors]
                finished, pending = ray.wait(closing, num_returns=len(closing), timeout=120)
                if pending:
                    errors.append("worker shutdown timed out")
                for ref in finished:
                    try:
                        ray.get(ref)
                    except Exception as exc:
                        errors.append(str(exc))
        except Exception as exc:
            errors.append(str(exc))
        finally:
            for actor in self.actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as exc:
                    errors.append(str(exc))
            if failed:
                self.pool.shutdown(wait=True, cancel_futures=True)
            self.actors.clear()
            try:
                if self.pg is not None:
                    remove_placement_group(self.pg)
                    self.pg = None
            finally:
                if self.owns_connection and ray.is_initialized():
                    ray.shutdown()
        if errors:
            if not failed:
                raise RuntimeError("; ".join(errors))
            logger.warning("Ray teardown after failure: %s", errors)
