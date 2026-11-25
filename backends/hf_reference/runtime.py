"""Asynchronous single-node process executor. One serial CUDA command stream per GPU."""
import multiprocessing as mp
import time
import traceback

from ..executor import GroupExecutor
from ...cluster.topology import local_placements


def serve(rank, cfg, pipe, origin):
    try:
        from .engine import Worker
        import torch
        worker = Worker(rank, cfg)
        while True:
            msg = pipe.recv()
            if msg is None:
                break
            kind, version, payload = msg
            start = time.monotonic()-origin
            value = worker.execute(kind, payload, version)
            worker.fence()
            pipe.send(dict(start=start, end=time.monotonic()-origin, value=value,
                           free_bytes=torch.cuda.mem_get_info(worker.device)[0]))
    except BaseException:
        pipe.send({"error": traceback.format_exc()})
    finally:
        pipe.close()


class GPUExecutor(GroupExecutor):
    supports_periodic_checkpoint = False
    supports_resume = False

    def __init__(self, cfg):
        self.placements = local_placements(cfg["groups"], 1, allow_extra=True)
        super().__init__(cfg["groups"])
        self.processes, self.pipes = [], []
        ctx = mp.get_context("spawn")
        for gpu in range(cfg["groups"]):
            host, remote = ctx.Pipe()
            process = ctx.Process(target=serve, args=(gpu, cfg, remote, self.origin))
            process.start()
            remote.close()
            self.pipes.append(host)
            self.processes.append(process)

    def call(self, gpu, kind, version, payload):
        self.pipes[gpu].send((kind, version, payload))
        reply = self.pipes[gpu].recv()
        if "error" in reply:
            raise RuntimeError(f"GPU {gpu} {kind}\n{reply['error']}")
        return reply

    def close(self, failed=False):
        if failed:
            for p in self.processes:
                if p.is_alive():
                    p.terminate()
        self.pool.shutdown(wait=True)
        if not failed:
            for p in self.pipes:
                p.send(None)
        for p in self.processes:
            p.join()
        for p in self.pipes:
            p.close()
