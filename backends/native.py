"""Node-local executor for parallel GPU group actions and paired rollout meshes."""

from ..cluster.topology import local_placements
from ..workers.group import GroupWorker, shard_boot
from ..workers.model import inference_signature, stage_model, stage_models
from ..workers.node import NodeRuntime, uses_wide_mesh
from .elastic_profile import TailProfile
from .executor import GroupExecutor

# These worker entry points were previously defined in this module.
__all__ = ["NativeExecutor", "GroupWorker", "shard_boot", "inference_signature", "stage_model", "stage_models"]


class NativeExecutor(GroupExecutor):
    supports_trainer_replicas = True
    supervision = GroupWorker.supervision

    def __init__(self, cfg):
        world = cfg["group_size"]
        if world < 1 or cfg["temperature"] != 1:
            raise ValueError("native sampled-KL configuration requires a positive group size and temperature 1")
        if cfg.get("cluster", {}).get("nodes"):
            raise ValueError("explicit cluster.nodes placement requires ray transport")
        self.placements = local_placements(cfg["groups"], world)
        super().__init__(cfg["groups"])
        self.runtime = None
        try:
            cfg["staged"] = stage_models(cfg, self.pool)
            self.tail_tp = TailProfile(cfg, self.placements) if cfg.get("rollout_tp_expansion") else None
            self.runtime = NodeRuntime(cfg, self.placements, self.tail_tp)
            self.tp_setup, self.teacher_preflight = self.runtime.prepare()
            if self.tail_tp:
                self.tail_tp.scratch = max(x["wake_bytes"] for x in self.tp_setup)
        except BaseException:
            self.close(failed=True)
            raise

    def execute(self, action):
        if uses_wide_mesh(action):
            start = self.now()
            value, free = self.runtime.wide(action)
            return dict(start=start, end=self.now(), value=value, free_bytes=free)
        return super().execute(action)

    def call(self, gpu, kind, version, payload):
        start = self.now()
        value, free = self.runtime.call(gpu, kind, version, payload)
        return dict(start=start, end=self.now(), value=value, free_bytes=free)

    def close(self, failed=False):
        if failed and self.runtime:
            self.runtime.abort()
        self.pool.shutdown(wait=True, cancel_futures=True)
        if self.runtime:
            self.runtime.close(failed)
