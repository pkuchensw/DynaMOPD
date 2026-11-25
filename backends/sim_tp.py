"""One-node, two-group tensor parallelism for a single surviving rollout."""
from collections import defaultdict
from itertools import combinations
import math

from srl.cost.rollout.roofline import UnifiedRoofline


class TailTP:
    def __init__(self, cfg, placements, context_capacity):
        profile = cfg["sim_tp"]
        self.base = cfg.get("group_size", 1)
        self.context_capacity = context_capacity
        self.wide = 2 * self.base
        self.models = {}
        for width in (self.base, self.wide):
            cell = profile["roofline_by_tp"].get(str(width))
            if not cell or not all(math.isfinite(v) and v >= 0 for v in cell.values()) or cell["F"] <= 0:
                raise ValueError(f"sim_tp requires finite roofline coefficients and positive F for TP{width}")
            self.models[width] = UnifiedRoofline(**cell)
        for name in ("attention_heads", "kv_heads"):
            heads = profile[name]
            if type(heads) is not int or heads <= 0 or heads % self.wide:
                raise ValueError(f"sim_tp requires {name} divisible by TP{self.wide} for head sharding")
        self.link = profile["kv_bandwidth_bytes_s"]
        self.bytes_per_token = cfg["kv_bytes_per_token"]
        if not all(math.isfinite(x) and x > 0 for x in (self.link, self.bytes_per_token)):
            raise ValueError("TP KV bandwidth and bytes per token must be positive")
        self.timing = {key: profile[key] for key in (
            "communicator_s", "enter_s", "exit_s", "teacher_to_student_s", "student_to_teacher_s")}
        self.scratch = profile["scratch_bytes_per_gpu"]
        if type(self.scratch) is not int or self.scratch < 0 or not all(
                math.isfinite(v) and v >= 0 for v in self.timing.values()):
            raise ValueError("TP transition times and scratch reservation must be nonnegative")
        # Local replicas use matching rank order within each base group. An
        # optional allow-list limits the pairs to measured NVLink islands.
        self.partners = defaultdict(list)
        nodes = defaultdict(list)
        for p in placements:
            nodes[p.node].append(p.group)
        local = {pair for groups in nodes.values() for pair in combinations(groups, 2)}
        pairs = {tuple(sorted(pair)) for pair in profile.get("group_pairs", local)}
        if not pairs <= local:
            raise ValueError("sim_tp group_pairs must name two distinct groups on the same node")
        for left, right in sorted(pairs):
            self.partners[left].append(right)
            self.partners[right].append(left)
        self.communicators = set()

    def transition(self, leader, peer, role, context, entering):
        # Half of the logical KV heads move. Original TP replicas supply local
        # weight shard views. A donor serving a teacher must restore the student.
        kv_bytes = math.ceil(context * self.bytes_per_token / 2)
        pair = tuple(sorted((leader, peer)))
        setup = self.timing["communicator_s"] if entering and pair not in self.communicators else 0.
        layout = self.timing["enter_s" if entering else "exit_s"]
        weights = self.timing["teacher_to_student_s" if entering else "student_to_teacher_s"] if role != "rollout" else 0.
        if role == "standby":
            weights = (self.timing.get("standby_to_student_s", weights) if entering
                       else self.timing.get("student_to_standby_s", 0.))
        transfer = kv_bytes / self.link
        return dict(communicator_s=setup, layout_s=layout, donor_role_s=weights,
                    kv_transfer_bytes=kv_bytes, kv_transfer_s=transfer,
                    transition_s=setup + layout + weights + transfer)

    def predict(self, context, remaining, width):
        # Integrate the floor + linear context term of a single-sequence decode.
        # Sum exactly on either side of the roofline knee, without a token loop.
        steps = max(1, math.ceil(remaining))
        floor, first, slope = self.models[width].decode_coeffs(1, context)
        flat = min(steps, max(0, math.ceil((floor-first)/slope))) if slope else 0
        count = steps - flat
        return flat*floor + count*max(first + flat*slope, floor) + slope*count*(count-1)/2
