"""Sample the allocated physical GPUs on the machine that owns them, without CUDA contexts."""
import json
import logging
from pathlib import Path
from threading import Event, Thread

logger = logging.getLogger(__name__)


class GpuMonitor:
    def __init__(self, directory, placements, now, interval, clock_error=0.):
        self.directory = Path(directory)
        self.placements = placements
        self.devices = [d for p in placements for d in p.devices]
        self.now, self.interval, self.clock_error = now, interval, clock_error
        self.stop_event = Event()
        self.thread = Thread(target=self.sample, name="dynamopd-nvml", daemon=True)

    def __enter__(self):
        self.directory.mkdir(parents=True, exist_ok=True)
        self.thread.start()
        return self

    def __exit__(self, *exc):
        self.stop_event.set()
        self.thread.join()

    def sample(self):
        dest = self.directory / f"group-{self.placements[0].group}.jsonl"
        try:
            with dest.open("x", buffering=1) as file:
                def write(record):
                    file.write(json.dumps(record, allow_nan=False) + "\n")
                write(dict(node=self.placements[0].node, devices=self.devices, interval=self.interval,
                           clock_error_s=self.clock_error))
                try:
                    import pynvml as nv
                    nv.nvmlInit()
                    try:
                        handles = [nv.nvmlDeviceGetHandleByIndex(int(d)) if d.isdecimal()
                                   else nv.nvmlDeviceGetHandleByUUID(d.encode()) for d in self.devices]
                        while not self.stop_event.is_set():
                            start = self.now()
                            values = []
                            for handle in handles:
                                try:
                                    util = nv.nvmlDeviceGetUtilizationRates(handle).gpu
                                except nv.NVMLError:
                                    util = None
                                try:
                                    used = nv.nvmlDeviceGetMemoryInfo(handle).used
                                except nv.NVMLError:
                                    used = None
                                values.append([util, used])
                            write(dict(t=(start + self.now()) / 2, values=values))
                            self.stop_event.wait(max(0., self.interval - (self.now() - start)))
                    finally:
                        nv.nvmlShutdown()
                except Exception as exc:
                    write(dict(error=str(exc)))
                    logger.warning("GPU telemetry unavailable for %s: %s", dest, exc)
        except OSError as exc:
            logger.warning("GPU telemetry file %s unavailable: %s", dest, exc)
