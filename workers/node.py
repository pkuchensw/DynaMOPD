"""Engines and IPC barriers shared by the GPU groups inside one allocation."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
import multiprocessing as mp

from ..backends.elastic_rollout import WideRollout
from .group import GroupWorker


def uses_wide_mesh(action):
    return action.kind in ("tp_expand", "tp_shrink") or (action.kind == "decode" and action.peer is not None)


class NodeRuntime:
    def __init__(self, cfg, placements, profile):
        self.cfg = cfg
        self.workers = {}
        self.meshes = {}
        self.node = placements[0].node
        width = cfg["group_size"]
        for i, p in enumerate(placements):
            ranks = range(i * width, (i + 1) * width)
            self.workers[p.group] = GroupWorker(p.group, list(p.devices), cfg, local_ranks=ranks)
        local = set(self.workers)
        for pair in (profile.pairs if profile else ()):
            if local.isdisjoint(pair):
                continue
            if not local.issuperset(pair):
                raise ValueError(f"TP mesh {pair} spans independently owned allocations")
            partners = [self.workers[g] for g in pair]
            mesh = WideRollout(cfg, partners, profile)
            self.meshes[frozenset(pair)] = mesh
            for worker in partners:
                worker.wide_mesh = mesh

    def prepare(self):
        with ThreadPoolExecutor(max_workers=len(self.workers)) as pool:
            try:
                futures = [pool.submit(mesh.prepare) for mesh in self.meshes.values()]
                gate = (mp.get_context("spawn").Event() if futures and
                        self.cfg.get("overlap_wide_startup", True) else None)
                if gate is not None:
                    for worker in self.workers.values():
                        worker.start_shards(gate)
                for future in as_completed(futures):
                    future.result()
                if gate is not None:
                    gate.set()
                # All groups must enter cross-replica NCCL initialization concurrently.
                jobs = [pool.submit(w.prepare) for w in self.workers.values()]
                for future in as_completed(jobs):
                    future.result()
            except BaseException:
                self.abort()
                raise
        for mesh in self.meshes.values():
            mesh.startup["node"] = self.node
            mesh.startup["shard_startup"] = {g: self.workers[g].shard_startup for g in mesh.groups}
        return ([mesh.startup for mesh in self.meshes.values()],
                [entry for w in self.workers.values() for entry in w.teacher_preflight])

    def call(self, group, op, version, payload):
        worker = self.workers[group]
        try:
            value = worker.execute(op, payload, version)
            return value, worker.available()
        except BaseException:
            for mesh in self.meshes.values():
                mesh.barrier.abort()
            raise

    def wide(self, action):
        mesh = self.meshes[frozenset(action.devices)]
        leader, peer = self.workers[action.gpu], self.workers[action.peer]
        value = mesh.execute(action, leader, peer)
        value["peer_free_bytes"] = peer.available()
        return value, leader.available()

    def abort(self):
        for mesh in self.meshes.values():
            mesh.barrier.abort()
        for worker in self.workers.values():
            for proc in worker.processes:
                if proc.is_alive():
                    proc.terminate()

    def close(self, failed):
        try:
            with ExitStack() as cleanup:
                for mesh in self.meshes.values():
                    cleanup.callback(mesh.close)
        finally:
            with ThreadPoolExecutor(max_workers=len(self.workers)) as pool:
                jobs = [pool.submit(w.close, failed) for w in self.workers.values()]
                for job in jobs:
                    job.result()
