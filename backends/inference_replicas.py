"""Independent SGLang TP meshes inside a node-local FSDP group."""
from concurrent.futures import ThreadPoolExecutor

from .sglang_engine import Server


class InferenceReplicas:
    def __init__(self, cfg, devices, width, name, model, version):
        self.width = width
        self.groups = cfg["groups"]
        partitions = [devices[i:i + width] for i in range(0, len(devices), width)]
        self.pool = ThreadPoolExecutor(max_workers=len(partitions))
        jobs = [self.pool.submit(Server, cfg, part, name, model, version) for part in partitions]
        self.servers = []
        try:
            self.servers = [job.result() for job in jobs]
        except BaseException:
            self.pool.shutdown(wait=True)
            for job in jobs:
                if job.exception() is None:
                    job.result().close()
            raise

    def lane(self, seed):
        return seed // self.groups % len(self.servers)

    def migration_fits(self, contexts, quantum, limit):
        needed = [0] * len(self.servers)
        for seed, context in contexts:
            needed[self.lane(seed)] += context + quantum
        return all(n <= min(s.capacity, limit) for n,s in zip(needed, self.servers, strict=True))

    @property
    def version(self):
        versions = {s.version for s in self.servers}
        if len(versions) != 1:
            raise RuntimeError("SGLang replicas disagree on weight version")
        return versions.pop()

    @property
    def capacity(self):
        return sum(s.capacity for s in self.servers)

    def each(self, method, *args):
        jobs = [self.pool.submit(getattr(s, method), *args) for s in self.servers]
        return [j.result() for j in jobs]

    def sleep(self, tags=None):
        self.each("sleep", tags)

    def wake(self, tags=None):
        self.each("wake", tags)

    def publish(self, path, version):
        self.each("publish", path, version)

    def flush_score_lane(self, lane):
        self.servers[lane].post("flush_cache?timeout=10", {})

    def confirm_version(self, version):
        self.each("confirm_version", version)

    def restore_rope(self, path, force=False):
        return any(self.each("restore_rope", path, force))

    def post(self, endpoint, payload):
        if not endpoint.startswith("flush_cache"):
            raise ValueError(f"unrouted replica operation {endpoint}")
        return self.each("post", endpoint, payload)

    def publish_bucket(self, handles, version, last):
        if len(handles) != self.width * len(self.servers):
            raise ValueError("one local CUDA IPC handle is required per physical device")
        jobs = [self.pool.submit(s.publish_bucket, handles[i*self.width:(i+1)*self.width], version, last)
                for i,s in enumerate(self.servers)]
        for job in jobs:
            job.result()

    def batch(self, items, params, **options):
        indices = [[] for _ in self.servers]
        for i, item in enumerate(items):
            indices[self.lane(item["seed"])].append(i)
        self.last_batch_sizes = [len(part) for part in indices]
        jobs = []
        for server, positions in zip(self.servers, indices, strict=True):
            if not positions:
                continue
            def select(value):
                return [value[i] for i in positions] if isinstance(value, list) else value
            jobs.append((positions, self.pool.submit(server.batch, select(items), select(params),
                                                     **{k: select(v) for k,v in options.items()})))
        result = [None] * len(items)
        for positions, job in jobs:
            for i, row in zip(positions, job.result(), strict=True):
                result[i] = row
        self.generate_body_bytes = sum(self.servers[i].generate_body_bytes for i, lane in enumerate(indices) if lane)
        return result

    def close(self):
        try:
            self.each("close")
        finally:
            self.pool.shutdown()
