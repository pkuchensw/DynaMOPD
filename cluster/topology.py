"""A group ID is global. A physical device ID is meaningful only on its owning node."""
from dataclasses import dataclass
import math
import os
import socket


@dataclass(frozen=True)
class GroupPlacement:
    group: int
    node: str
    devices: tuple[str, ...]


def visible_devices():
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is None:
        import torch
        return [str(i) for i in range(torch.cuda.device_count())]
    return [device.strip() for device in mask.split(",")] if mask.strip() else []


def local_placements(groups, width, *, allow_extra=False):
    devices = visible_devices()
    if allow_extra:
        devices = devices[:groups*width]
    if not all(devices) or len(devices) != groups * width or len(devices) != len(set(devices)):
        raise ValueError("visible GPUs must exactly partition the configured local groups")
    return tuple(GroupPlacement(i, socket.gethostname(), tuple(devices[i*width:(i+1)*width]))
                 for i in range(groups))


@dataclass(frozen=True)
class Allocation:
    groups: tuple[int, ...]
    address: str | None = None

    def resources(self, width, cpus):
        if type(width) is not int or width < 1:
            raise ValueError("group_size must be a positive integer")
        if type(cpus) not in (int, float) or not math.isfinite(cpus) or cpus <= 0:
            raise ValueError("cpus_per_group must be positive and finite")
        count = len(self.groups)
        bundle = {"CPU": cpus * count, "GPU": width * count}
        if self.address:
            bundle[f"node:{self.address}"] = 0.001
        return bundle

    def place(self, node, devices, width):
        """CUDA ordinals inside the actor follow its actual Ray visibility mask."""
        if self.address and node != self.address:
            raise ValueError(f"Ray placed {self.groups} on {node}, expected {self.address}")
        if len(devices) != width * len(self.groups) or not all(devices) or len(set(devices)) != len(devices):
            raise ValueError(f"invalid visible devices {devices} for groups {self.groups}, width {width}")
        return tuple(GroupPlacement(g, node, tuple(devices[i*width:(i+1)*width]))
                     for i, g in enumerate(self.groups))


def node_shape(cluster):
    """Ordered (node IP or None, logical group count) for whole-node reservations."""
    nodes = cluster.get("nodes", [])
    if "node_count" in cluster or "groups_per_node" in cluster:
        n, k = cluster.get("node_count"), cluster.get("groups_per_node")
        if nodes or any(type(v) is not int or v < 1 for v in (n, k)):
            raise ValueError("specify nodes or positive node_count and groups_per_node")
        return ((None, k),) * n
    shape = []
    seen = set()
    for entry in nodes:
        ip, count = entry["address"], entry["groups"]
        if not isinstance(ip, str) or not ip or ip in seen or type(count) is not int or count < 1:
            raise ValueError("each node needs a unique address and a positive integer group count")
        seen.add(ip)
        shape.append((ip, count))
    return tuple(shape)


def allocations(config):
    shape = node_shape(config.get("cluster", {}))
    groups = config["groups"]
    if not shape:
        return tuple(Allocation((i,)) for i in range(groups))
    if sum(count for _, count in shape) != groups:
        raise ValueError("cluster node group counts must add up to groups")
    plan, start = [], 0
    for ip, count in shape:
        plan.append(Allocation(tuple(range(start, start + count)), ip))
        start += count
    return tuple(plan)


def requested_nodes(config):
    """Node labels for offline simulation, including unpinned multi-node layouts."""
    shape = node_shape(config.get("cluster", {}))
    if not shape:
        return ()
    return tuple(b.address or f"sim-node-{i}" for i, b in enumerate(allocations(config)) for _ in b.groups)


def local_pairs(placements):
    groups = {}
    for p in placements:
        groups.setdefault(p.node, []).append(p.group)
    return [(a, b) for ids in groups.values() for a, b in zip(ids[::2], ids[1::2])]


def describe_layout(cfg):
    """Describe an offline Ray reservation. Device IDs are known only after placement."""
    from ..backends.elastic_profile import TailProfile

    cluster = cfg["cluster"]
    if cluster.get("transport") != "ray" or not node_shape(cluster):
        raise ValueError("layout inspection needs a Ray cluster with nodes or node_count and groups_per_node")
    width, groups = cfg["group_size"], cfg["groups"]
    tp = cfg.get("inference_tp_size", width)
    if cfg.get("engine", "sglang_fsdp2") != "sglang_fsdp2" or width < 2 or tp < 1 or width % tp:
        raise ValueError("expected SGLang/FSDP2 with inference_tp_size dividing a sharded group")
    if cfg["batch_size"] < groups:
        raise ValueError("each rollout group requires at least one request per round")
    plan = allocations(cfg)
    nodes = requested_nodes(cfg)
    placements = [GroupPlacement(g, node, ()) for g, node in enumerate(nodes)]
    profile = TailProfile(dict(cfg, first_round_calibration=True), placements) if cfg.get("rollout_tp_expansion") else None
    return dict(nodes=len(plan), groups=groups, gpus=groups * width,
                inference_tp_size=tp, rollout_replicas=groups * (width // tp),
                fsdp_group_size=width, standby_pairs=profile.pairs if profile else [],
                bundles=[dict(address=a.address, groups=a.groups, resources=a.resources(
                    width, cluster.get("cpus_per_group", 2))) for a in plan])


def warm_teacher_routes(cfg, group):
    """Rotate the configured routes so each group preloads only its share."""
    if not cfg.get("warm_teachers", True):
        return ()
    routes = list(cfg["teachers"])
    count = min(len(routes), cfg.get("teachers_per_group", len(routes)))
    return tuple(routes[(group*count + i) % len(routes)] for i in range(count))

