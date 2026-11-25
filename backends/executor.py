"""Dispatch independent GPU groups and transfer ownership at action boundaries."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import time


class GroupExecutor:
    supervision = frozenset({"sampled_token_logprob"})
    supports_periodic_checkpoint = True
    supports_resume = True

    @staticmethod
    def validate_checkpoint(path):
        if not (path/"training"/".metadata").is_file():
            raise ValueError(f"missing FSDP2 state at {path}")

    def __init__(self, groups):
        self.origin = time.monotonic()
        self.pending = {}
        self.pool = ThreadPoolExecutor(max_workers=groups)

    def gpu_monitor(self, directory, interval):
        from ..workers.gpu_monitor import GpuMonitor
        return GpuMonitor(directory, self.placements, self.now, interval)

    def call(self, gpu, kind, version, payload):
        raise NotImplementedError

    def now(self):
        return time.monotonic()-self.origin

    def execute(self, a):
        if a.kind != "move":
            return self.call(a.gpu, a.kind, a.version, a.payload)
        ids = [x["rid"] for x in a.payload["items"]]
        start = self.now()
        src = self.call(a.gpu, "export", a.version, {"ids":ids})
        rng = src["value"]["rng"]
        if set(rng) != set(ids):
            raise RuntimeError("source exported different requests from the migration lease")
        dest = self.call(a.peer, "install", a.version, dict(a.payload, rng=rng))
        result = dest["value"]
        result["peer_free_bytes"] = dest["free_bytes"]
        result.update(ids=ids, export_start=src["start"], export_end=src["end"],
                      install_start=dest["start"], install_end=dest["end"],
                      rng_bytes=sum(map(len, rng.values())))
        result.update(source_node=self.placements[a.gpu].node, destination_node=self.placements[a.peer].node)
        if result["accepted"]:
            src = self.call(a.gpu, "release", a.version, {"ids":ids})
            released = src["value"]["released"]
            if len(released) != len(ids) or set(released) != set(ids):
                raise RuntimeError("source did not release all migrated requests")
            result.update(release_start=src["start"], release_end=src["end"])
        return dict(start=start, end=self.now(), free_bytes=src["free_bytes"], value=result)

    def submit(self, action):
        self.pending[self.pool.submit(self.execute, action)] = action

    def wait(self):
        wait(self.pending, return_when=FIRST_COMPLETED)
        return [(self.pending.pop(f), f.result()) for f in list(self.pending) if f.done()]
