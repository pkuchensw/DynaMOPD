"""One GPU group owns the inference engines, training ranks and safe role transitions."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, nullcontext
from importlib import import_module
import multiprocessing as mp
from pathlib import Path
import time
from tempfile import TemporaryDirectory

from ..backends.inference_replicas import InferenceReplicas
from ..backends.sglang_engine import Inference, Server
from ..backends.teacher_pool import TeacherRanks
from ..cluster.topology import warm_teacher_routes
from .model import inference_signature


def shard_boot(cfg, gpu, rank, world, init, child, gpu_gate=None):
    # These are CPU imports. CUDA placement, model construction and NCCL
    # initialization in serve_shard all happen after the wide mesh parks.
    preload_s = gate_wait_s = 0.
    if gpu_gate is not None:
        begin = time.monotonic()
        # Only import the core training runtime here. External model code and
        # attention libraries may probe or allocate CUDA during their imports.
        import_module("torch.distributed.checkpoint")
        import_module("torch.distributed.fsdp")
        imported = time.monotonic()
        gpu_gate.wait()
        preload_s, gate_wait_s = imported - begin, time.monotonic() - imported
    from ..backends.fsdp2 import serve_shard
    serve_shard(cfg, gpu, rank, world, init, child, preload_s=preload_s, gate_wait_s=gate_wait_s)


def publish_local(engine, handles, version, last):
    engine.publish_bucket(handles, version, last)
    return time.monotonic()


class GroupWorker:
    supervision = frozenset({"sampled_token_logprob", "teacher_topk"})

    def __init__(self, index, devices, cfg, local_ranks=None):
        self.index, self.devices, self.cfg = index, devices, cfg
        self.local_ranks = local_ranks if local_ranks is not None else range(index*len(devices), (index+1)*len(devices))
        self.processes, self.links, self.servers = [], [], {}
        self.rendezvous_dir = None
        self.role, self.version = "cold", -1
        self.infer = Inference(cfg)
        self.wake_bytes = {}
        self.measured_wake = {}
        self.resident_version = None
        self.publish_scratch_bytes = 0
        self.teacher_ranks = None
        self.isolated_roles = set()
        self.wide_mesh = None
        self.wide_publish_s = 0.
        self.wide_publish_exposed_s = 0.
        self.teacher_preflight = []
        self.single_engine = cfg.get("inference_layout") == "single"

    def engine_aliases(self):
        paths = dict(rollout=self.cfg["student"], **self.cfg["teachers"])
        if self.single_engine:
            limit = self.cfg["max_prompt"] + self.cfg["max_new_tokens"] + 1
            signatures = {name: inference_signature(self.cfg["staged"][path], limit,
                                                    self.cfg.get("qwen2_dynamic_rope", False))
                          for name, path in paths.items()}
            aliases = {name for name in paths if signatures[name] == signatures["rollout"]}
        else:
            aliases = {"rollout"}
            if self.cfg.get("share_student_teacher_engine", True):
                aliases.update(name for name, path in paths.items() if path == self.cfg["student"])
        return aliases, set(paths) - aliases

    def start_shards(self, gpu_gate=None):
        import pynvml
        pynvml.nvmlInit()
        self.handles = [pynvml.nvmlDeviceGetHandleByUUID(d.encode()) if d.startswith("GPU-")
                        else pynvml.nvmlDeviceGetHandleByIndex(int(d)) for d in self.devices]
        world = len(self.devices)
        ctx = mp.get_context("spawn")
        self.rendezvous_dir = TemporaryDirectory(prefix=f"dynamopd-fsdp-{self.index}-")
        init = (Path(self.rendezvous_dir.name).resolve() / "store").as_uri()
        self.aliases, self.isolated_roles = self.engine_aliases()
        shard_cfg = dict(self.cfg, worker_group=self.index)
        shard_cfg.setdefault("mirror_optimizer_on_rollout", bool(self.isolated_roles) and
                             self.cfg.get("trainer_residency") == "auto")
        shard_cfg.setdefault("mirror_parameters_on_rollout", shard_cfg["mirror_optimizer_on_rollout"])
        for rank in range(world):
            host, child = ctx.Pipe()
            gpu = self.local_ranks[rank]
            proc = ctx.Process(target=shard_boot, args=(shard_cfg, gpu, rank, world, init, child, gpu_gate))
            try:
                proc.start()
            except BaseException:
                host.close()
                raise
            finally:
                child.close()
            self.processes.append(proc)
            self.links.append(host)

    def prepare(self):
        if not self.processes:
            self.start_shards()
        ready = self.gather()
        self.publish_scratch_bytes = max(r["publish_scratch_bytes"] for r in ready)
        self.shard_startup = [dict(cpu_preload_s=r["cpu_preload_s"], gpu_gate_wait_s=r["gpu_gate_wait_s"]) for r in ready]
        if self.isolated_roles:
            self.teacher_ranks = TeacherRanks(self.cfg, self.devices, self.handles, [r["device_properties"] for r in ready])
            # Pay first-time kernel compilation and alignment checks before the
            # rollout barrier. Nothing from these GPU processes survives release.
            if self.cfg.get("preflight_heterogeneous_teachers", True):
                routes = self.isolated_roles if self.single_engine else self.isolated_roles.intersection(
                    warm_teacher_routes(self.cfg, self.index))
                for role in sorted(routes):
                    report = self.teacher_ranks.activate(role, calibrate_memory=True)
                    released = self.teacher_ranks.release()
                    self.teacher_preflight.append(dict(group=self.index, teacher=role,
                        load_format=report["teacher_load_format"], load_s=report["teacher_rank_load_s"],
                        image_build_s=report["teacher_image_build_s"], image_write_error=report["teacher_image_write_error"],
                        alignment_s=report["teacher_alignment_s"], release_s=released["teacher_rank_release_s"]))
        server = self.launch_rollout()
        self.servers = dict.fromkeys(self.aliases, server)
        if self.cfg.get("calibrate_engine_wake", False):
            server.sleep()
            free = self.free_on_devices()
            server.wake()
            self.measured_wake[server] = max(a-b for a,b in zip(free,self.free_on_devices(),strict=True))
            server.publish(self.cfg["staged"][self.cfg["student"]], 0)
            self.set_role_rope("rollout", force=True)
        self.role, self.version = "rollout", 0

    def launch_rollout(self):
        width = self.cfg.get("inference_tp_size", len(self.devices))
        path = self.cfg["staged"][self.cfg["student"]]
        if width == len(self.devices):
            return Server(self.cfg, self.devices, "rollout", path, "v0")
        return InferenceReplicas(self.cfg, self.devices, width, "rollout", path, "v0")

    def set_role_rope(self, role, force=False):
        if self.cfg.get("qwen2_dynamic_rope", False):
            path = self.cfg["student"] if role == "rollout" else self.cfg["teachers"][role]
            return self.servers[role].restore_rope(self.cfg["staged"][path], force=force)
        return False

    def gather(self):
        result = []
        for link in self.links:
            if not link.poll(self.cfg.get("startup_timeout",1200)):
                raise TimeoutError(f"FSDP2 group {self.index} timed out")
            value = link.recv()
            if "error" in value:
                raise RuntimeError(value["error"])
            result.append(value)
        return result

    def collective(self, op, p, version):
        for link in self.links:
            link.send((op,p,version))
        return self.gather()

    def free_on_devices(self):
        import pynvml
        return [pynvml.nvmlDeviceGetMemoryInfo(h).free for h in self.handles]

    def available(self):
        return min(self.free_on_devices())

    def suspend_engine(self, role):
        before = self.free_on_devices()
        server = self.servers[role]
        server.sleep()
        released = max(a-b for a,b in zip(self.free_on_devices(), before, strict=True))
        for name in self.servers:
            self.wake_bytes[name] = max(self.wake_bytes.get(name, 0), released)

    def can_retain_trainer(self, role, scratch_bytes):
        wake = self.measured_wake.get(self.servers.get(role), self.wake_bytes.get(role))
        if self.cfg.get("trainer_residency", "cpu") != "auto" or not wake:
            return False
        required = wake + self.cfg["memory_margin"] + scratch_bytes
        return self.available() >= required

    def student_image(self,version):
        if not version:
            return self.cfg["staged"][self.cfg["student"]]
        durable = Path(self.cfg["state_dir"])/f"v{version}"
        if durable.is_dir():
            return durable
        if self.cfg.get("weight_sync", "tensor") == "disk":
            return Path(self.cfg["export_dir"])/f"v{version}"
        raise RuntimeError(f"no persisted rollout image for v{version}")

    def publish_tensors(self, source, version):
        begin = time.monotonic()
        manifests = self.collective("begin_weights", {"source": source}, version)
        manifest = manifests[0]
        if any(m != manifest for m in manifests):
            raise RuntimeError("TP ranks disagree on the weight manifest")
        self.wide_publish_s = self.wide_publish_exposed_s = 0.
        memory = self.load_buckets(self.servers["rollout"], manifest, version)
        return dict(**memory, weight_sync="tensor", publish_source=source, wide_publish_s=self.wide_publish_s,
                    wide_publish_exposed_s=self.wide_publish_exposed_s, tensor_publish_s=time.monotonic()-begin,
                    weight_buckets=manifest["buckets"], weight_bytes=manifest["weight_bytes"],
                    weight_schema=manifest["schema"])

    def load_buckets(self, engine, manifest, version):
        wide = self.wide_mesh is not None and isinstance(version,int) and version < self.cfg["final_version"]
        parallel = wide and self.cfg.get("parallel_wide_publication", True)
        with ThreadPoolExecutor(1) if parallel else nullcontext() as pool:
            for index in range(manifest["buckets"]):
                handles = self.collective("weight_bucket", {"index": index, "wide_receiver": wide}, version)
                last = index + 1 == manifest["buckets"]
                local = [r["handle"] for r in handles]
                if parallel:
                    pending = pool.submit(publish_local, engine, local, version, last)
                else:
                    engine.publish_bucket(local, version, last)
                if wide:
                    start = time.monotonic()
                    self.wide_mesh.publish(self.index, [r["wide_handle"] for r in handles], version, index, last)
                    end = time.monotonic()
                    self.wide_publish_s += end - start
                    local_end = pending.result() if parallel else start
                    self.wide_publish_exposed_s += max(0., end - local_end)
        ranks = self.collective("finish_weights", {"wide_receiver": wide}, version)
        return {name:max(r[name] for r in ranks) for name in ranks[0]}

    def switch(self, role, version, publish_from=None, optimizer_owner=True):
        if self.infer.owned:
            raise RuntimeError("role transition with live SGLang requests")
        old = self.role
        start = time.monotonic()
        evicted = list(self.infer.cached)
        self.infer.cached.clear()
        before = self.available()
        stats = {}
        tensor_sync = (role == "rollout" and publish_from is not None
                       and self.cfg.get("weight_sync", "tensor") == "tensor")
        same_engine = old in self.servers and self.servers[old] is self.servers.get(role)
        frozen_sync = (self.cfg.get("frozen_teacher_tensor_sync", False) and
                       role in self.cfg["teachers"] and role in self.servers and self.single_engine and
                       self.servers[role] is self.servers["rollout"])
        isolated = role in self.isolated_roles
        if role == "standby" and old not in self.isolated_roles:
            raise RuntimeError("only a leased teacher mesh can enter standby")
        if old in self.isolated_roles and old != role:
            stats.update(self.teacher_ranks.release())
            stats["free_after_teacher_release_bytes"] = self.free_on_devices()
        if tensor_sync and self.cfg.get("max_train_groups", 1) > 1:
            replicas = self.collective("refresh_parameters", {"owner": publish_from}, version)
            if any(r["master_version"] != version for r in replicas):
                raise RuntimeError("FSDP replicas did not acknowledge the rollout version")
            stats.update(replica_sync_s=time.monotonic() - start, master_version=version,
                         master_bytes=sum(r["master_shard_bytes"] for r in replicas))
            if self.resident_version is not None:
                self.resident_version = version
        if tensor_sync and publish_from == self.index:
            if old != "train" or self.version+1 != version:
                raise RuntimeError("only the trainer that just committed may publish")
            capture_start = time.monotonic()
            captures = self.collective("capture_weights", {}, version)
            self.publish_scratch_bytes = max(x["scratch_bytes"] for x in captures)
            stats["snapshot_shards_s"] = time.monotonic() - capture_start
        # During a student publication both SGLang layouts consume the same CUDA IPC
        # bucket. The wide weights stay awake across buckets, alongside the FSDP
        # gather and the narrow weights. Reserve this before deciding to keep Adam on GPU.
        wide_reserve = (self.wide_mesh.wake_bytes if tensor_sync and self.wide_mesh is not None
                        and version < self.cfg["final_version"] else 0)
        publish_scratch = self.publish_scratch_bytes + wide_reserve
        if wide_reserve:
            stats["wide_publication_reserve_bytes"] = wide_reserve
        if old != role:
            if old == "train":
                # Release unused backward buffers before querying NVML. The
                # caching allocator may still own tens of GB after zero_grad.
                try_retain = tensor_sync and self.cfg.get("trainer_residency") == "auto"
                parked = self.collective("sleep", {"retain": try_retain}, self.version+1)
                keep = try_retain and self.can_retain_trainer(role, publish_scratch)
                if try_retain and not keep:
                    parked = self.collective("evict_trainer", {}, self.version+1)
                device = "cuda" if keep else "cpu"
                if any(x["trainer_state_device"] != device for x in parked):
                    raise RuntimeError("FSDP ranks disagree on trainer residency")
                self.resident_version = version if keep else None
                stats["trainer_state_device"] = device
                stats["rollout_wake_bytes"] = self.wake_bytes.get(role)
                stats["calibrated_wake_bytes"] = self.measured_wake.get(self.servers.get(role))
                stats["publish_scratch_bytes"] = self.publish_scratch_bytes
            elif old in self.servers and not same_engine:
                self.suspend_engine(old)
        needs_capacity = old != role or tensor_sync
        if self.resident_version is not None and old != "train" and role != "train" and needs_capacity:
            scratch = publish_scratch if tensor_sync else 0
            if same_engine and (frozen_sync or tensor_sync):
                fits = self.available() >= self.cfg["memory_margin"] + publish_scratch
            elif isolated:
                required = self.teacher_ranks.required_free_bytes.get(role)
                free = self.free_on_devices()
                fits = bool(required) and all(a >= b for a, b in zip(free, required, strict=True))
                stats.update(teacher_admission_free_bytes=free, teacher_admitted_with_trainer=fits)
            else:
                fits = self.can_retain_trainer(role, scratch)
            if not fits:
                receipts = self.collective("evict_trainer", {}, self.resident_version)
                stats["reused_host_optimizer_bytes"] = sum(r.get("reused_host_optimizer_bytes", 0) for r in receipts)
                stats["reused_host_parameter_bytes"] = sum(r.get("reused_host_parameter_bytes", 0) for r in receipts)
                self.resident_version = None
                stats["trainer_state_device"] = "cpu"
        released = time.monotonic()
        after_release = self.available()
        if role == "train":
            stats.update(self.collective("activate", {"optimizer_owner": optimizer_owner}, version)[0])
            self.resident_version = None
        elif role == "standby":
            stats["engine_weight_version"] = None
        elif isolated:
            if role != old:
                stats["free_before_teacher_activation_bytes"] = self.free_on_devices()
                receipt = self.teacher_ranks.activate(role)
                if stats.get("teacher_admitted_with_trainer") and any(
                    got < target for got, target in zip(receipt["teacher_kv_capacity"],
                            self.teacher_ranks.reference_capacity[role], strict=True)
                ):
                    # Re-profile without the resident trainer before accepting any requests.
                    stats["teacher_capacity_retry"] = self.teacher_ranks.release()
                    receipts = self.collective("evict_trainer", {}, self.resident_version)
                    self.resident_version = None
                    stats.update(trainer_state_device="cpu", teacher_admitted_with_trainer=False,
                                 reused_host_optimizer_bytes=sum(r.get("reused_host_optimizer_bytes", 0) for r in receipts))
                    receipt = self.teacher_ranks.activate(role, calibrate_memory=True)
                stats.update(receipt)
            stats["engine_weight_version"] = self.teacher_ranks.role
        else:
            if role != old:
                if not same_engine:
                    self.servers[role].wake(["weights"] if tensor_sync else None)
                stats["rotary_cache_updated"] = self.set_role_rope(role, force=not same_engine)
                if frozen_sync:
                    began = time.monotonic()
                    manifests = self.collective("begin_teacher", {}, role)
                    if any(m != manifests[0] for m in manifests):
                        raise RuntimeError("frozen teacher tensor manifests differ")
                    self.load_buckets(self.servers[role], manifests[0], role)
                    stats["teacher_tensor_load_s"] = time.monotonic() - began
                elif not tensor_sync:
                    source = (self.student_image(version) if role=="rollout"
                              else self.cfg["staged"][self.cfg["teachers"][role]])
                    self.servers[role].publish(source, version if role=="rollout" else role)
            if tensor_sync:
                stats.update(self.publish_tensors(publish_from, version))
                if role != old and not same_engine:
                    self.servers[role].wake(["kv_cache", "cuda_graph"])
            elif role=="rollout" and self.servers[role].version != f"v{version}":
                self.servers[role].publish(self.student_image(version),version)
            stats["engine_weight_version"] = self.servers[role].version
        current = self.servers["rollout"]
        replicas = current.servers if isinstance(current, InferenceReplicas) else [current]
        stats["inference_pids"] = [s.proc.pid for s in replicas]
        stats["inference_pid"] = stats["inference_pids"][0]
        active_teachers = self.teacher_ranks.active_pids if self.teacher_ranks else []
        stats["active_teacher_gpu_pids"] = list(active_teachers)
        stats["live_inference_engines"] = len(replicas) + (len(replicas) if active_teachers else 0)
        stats["engine_restarted"] = stats["rollout_engine_restarted"] = False
        self.role, self.version = role, version
        return dict(old=old, new=role, evicted=evicted, unload_s=released-start,
                    load_s=time.monotonic()-released, free_before=before, free_after_release=after_release,
                    physical_gpus=self.devices, cached_teacher_weights=[],
                    prepared_teachers=sorted(self.cfg.get("teachers", {})),
                    teacher_activation_s=dict(self.teacher_ranks.activation_s) if self.teacher_ranks else {},
                    teacher_start_s=0., **stats)

    def execute(self, op, p, version):
        if op == "switch":
            return self.switch(p["role"], p.get("load_version",version), p.get("publish_from"),
                               p.get("optimizer_owner", True))
        if op == "reduce_grads":
            needs_trainer = p["counts"][self.index] or self.index == p["owner"]
            if self.version != version or (needs_trainer and self.role != "train"):
                raise RuntimeError("gradient reduction on a stale worker or inactive trainer")
            ranks = self.collective(op, p, version)
            return dict(owner=p["owner"], group_tokens=ranks[0]["group_tokens"],
                        global_tokens=ranks[0]["global_tokens"],
                        gradient_bytes=sum(r["gradient_shard_bytes"] for r in ranks))
        expected = {"train":"train", "commit":"train", "score":p.get("teacher"),
                    "export":"rollout", "install":"rollout", "release":"rollout", "decode":"rollout"}
        if self.role != expected[op] or self.version != version:
            raise RuntimeError("operation on stale engine or role")
        for item in p.get("items",[]):
            if item["student_version"]!=version or (op=="score" and item["teacher"]!=self.role):
                raise RuntimeError(f"stale or misrouted trajectory {item['rid']}")
        if op == "train":
            stats = self.collective(op,p,version)
            return dict(tokens=sum(x["tokens"] for x in stats), loss_sum=sum(x["loss_sum"] for x in stats),
                        max_logp_error=max(x["max_logp_error"] for x in stats), rank_tokens=[x["tokens"] for x in stats])
        if op == "commit":
            return self.collective(op,p,version)[0]
        if op == "export":
            return {"rng":{rid:self.infer.owned[rid] for rid in p["ids"]}}
        if op == "release":
            for rid in p["ids"]:
                del self.infer.owned[rid]
            # Radix entries are evictable. Flush now if the whole group drained.
            if not self.infer.owned:
                self.servers["rollout"].post("flush_cache?timeout=10", {})
            return {"released":p["ids"]}
        if op == "score" and self.role in self.isolated_roles:
            return self.infer.score(self.teacher_ranks, p)
        return getattr(self.infer, op)(self.servers[self.role], p)

    def close(self, failed):
        # Send stop to every rank before joining any process. FSDP peers
        # may need to exit their process group together.
        with ExitStack() as cleanup:
            if self.rendezvous_dir:
                cleanup.callback(self.rendezvous_dir.cleanup)
            for link in self.links:
                cleanup.callback(link.close)
            for proc in self.processes:
                cleanup.callback(self.join_rank, proc)
            for link, proc in zip(self.links, self.processes, strict=True):
                if proc.is_alive():
                    if failed:
                        cleanup.callback(proc.terminate)
                    else:
                        cleanup.callback(link.send, None)
            for server in dict.fromkeys(self.servers.values()):
                cleanup.callback(server.close)
            if self.teacher_ranks is not None:
                cleanup.callback(self.teacher_ranks.close)

    @staticmethod
    def join_rank(proc):
        proc.join(timeout=30)
        if proc.is_alive():
            proc.kill()
            proc.join()
