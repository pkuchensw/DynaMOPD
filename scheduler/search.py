"""Receding-horizon list scheduling over a bounded set of alternative resource assignments."""
from collections import defaultdict
from bisect import bisect_right
from copy import copy
from dataclasses import dataclass
from hashlib import blake2s
import heapq
from itertools import chain
from math import isqrt
from time import perf_counter

from .protocol import Action
from ..backends.score_cache import trim_score_cache
from ..backends.sim_profile import (SwitchState, decode_duration, decode_features, decode_roofline, decode_windows, rpc_decode_features,
                                    engine_classes, fitted_switch, train_seconds, teacher_lease_window)


@dataclass(frozen=True)
class Segment:
    row: object
    start: int
    end: int
    release: float
    visible: bool


@dataclass
class ScoreJob:
    release: float
    stage: int
    role: str
    batch: list
    cold_s: float
    lane_tokens: list
    credits: list
    retains: dict
    action_items: list | None


def ordered_segments(queue):
    """Preserve a candidate priority order subject to each trajectory's score offsets."""
    producers = {(x.row.rid, x.end): i for i, job in enumerate(queue) for x in job.batch}
    parents = [{producers[key] for x in job.batch if (key := (x.row.rid, x.start)) in producers} for job in queue]
    if all(all(p < i for p in deps) for i, deps in enumerate(parents)):
        return queue
    children = defaultdict(list)
    available = []
    for i, deps in enumerate(parents):
        for parent in deps:
            children[parent].append(i)
        if not deps:
            heapq.heappush(available, i)
    result = []
    while available:
        i = heapq.heappop(available)
        result.append(queue[i])
        for child in children[i]:
            parents[child].remove(i)
            if not parents[child]:
                heapq.heappush(available, child)
    if len(result) != len(queue):
        raise RuntimeError("cyclic score dependencies in forecast")
    return result


def batches(entries, limit, budget, size, packed):
    batch, total, width = [], 0, 0
    for entry in entries:
        n = size(entry)
        projected = total+n if packed else max(width, n)*(len(batch)+1)
        if batch and (len(batch) == limit or projected > budget):
            yield batch
            batch, total, width = [], 0, 0
        batch.append(entry)
        total += n
        width = max(width, n)
    if batch:
        yield batch


class Forecast:
    def __init__(self, s, classes, *, early_training=True):
        self.s, self.cfg = s, s.cfg
        cfg = self.cfg
        self.training_barrier = cfg.get("phase_barriers", False) or not early_training
        self.n = len(s.groups)
        self.lanes = cfg.get("group_size", 1) // cfg.get("inference_tp_size", cfg.get("group_size", 1))
        self.classes = classes
        self.profile = cfg.get("sim_worker_profile", {})
        fallback = cfg["costs"]["score_base"], cfg["costs"]["score_token"]
        self.score_coeffs = {role: s.cost.score_coefficients(role, self.profile.get("score", {}).get(role) or fallback)
                             for role in cfg["teachers"]}
        self.now = s.now()
        self.prepare_s = s.cost.values.get("teacher_prepare", self.profile.get("teacher_prepare_s",
                                                                              cfg.get("sim_teacher_prepare_s", 0.)))
        if not cfg.get("preload_teacher_imports", True):
            self.prepare_s = 0.
        self.dedicated = {r for r in cfg["teachers"] if classes[r] != classes["rollout"]} if (
            cfg.get("engine") == "sglang_fsdp2" and cfg.get("sim_teacher_lifecycle", "lease") == "lease") else set()
        self.live = [[] for _ in s.groups]
        self.left = {}
        self.row_order = {rid: i for i, rid in enumerate(s.rows)}
        self.backfills = {}
        self.starts = {r.rid: len(r.labels) for r in s.rows.values()}
        self.available = [0.] * self.n
        self.score_done = {}
        self.inflight_tokens = {}
        self.score_cache = [{} for _ in s.groups]
        self.service_pending = any(not r.finish or (not r.discarded and len(r.labels) < len(r.tokens))
                                   for r in s.rows.values())
        self.role = [s.target_role(g) for g in s.groups]
        self.step = s.cost.get("decode_step")
        self.roof = decode_roofline(cfg)
        self.kv_pages = cfg.get("sim_blocks", cfg.get("engine_kv_tokens", 131072) // 16)
        self.switch_cache = {}
        for r in s.rows.values():
            for gpu, end in r.kv.items():
                self.score_cache[gpu][r.rid] = self.lane(r), len(r.prompt)+end
            if not r.finish:
                self.live[r.owner].append(r)
                self.left[r.rid] = max(1, round(s.remaining(r)))
        if cfg.get("length_forecast") == "conditional_quantiles" and sum(map(len, self.live)) > 1:
            cohorts = ([list(chain.from_iterable(self.live))] if
                       cfg.get("policy_options", {}).get("cluster_quantiles", True) else self.live)
            for rows in cohorts:
                self.estimate_quantiles(rows, s.prior_lengths)
        self.train_widths = {}
        self.train_rows = {True: [], False: []}
        self.batch_cache = {}
        for r in s.rows.values():
            if not (r.trained or r.training or r.discarded):
                self.train_rows[r.ready].append(r.rid)
                self.train_widths[r.rid] = len(r.prompt)+len(r.tokens)+self.left.get(r.rid, 0)
        for g in s.groups:
            if not g.busy:
                continue
            action = g.busy
            work = s.queued_work.get(g.gpu, action)
            elapsed = self.now - g.busy_since
            cost = self.switch(g.gpu, g.role, self.role[g.gpu], at=-elapsed) if action.kind == "switch" else 0.
            items = work.payload.get("items", [])
            if work.kind == "decode":
                rows = [s.rows[x["rid"]] for x in items]
                for r in rows:
                    self.inflight_tokens[r.rid] = min(self.left[r.rid], work.payload["steps"])
                if work.peer is not None and s.tp_model:
                    r, = rows
                    cost += s.tp_model.predict(len(r.prompt)+len(r.tokens), min(work.payload["steps"], self.left[r.rid]),
                                               s.tp_model.wide)
                else:
                    cost += self.decode_wave(rows, self.left, work.payload["steps"],
                                             {r.rid for r in rows if r.deferred_prefill})
            elif work.kind == "score":
                widths = [(s.rows[x["rid"]], len(x["tokens"])-x["start"] if
                           s.rows[x["rid"]].kv.get(g.gpu) == x["start"] else len(x["prompt"])+len(x["tokens"])-1)
                          for x in items]
                cost += self.score_time(widths, work.payload["teacher"])
            elif work.kind == "train":
                cost += self.train([len(x["prompt"])+len(x["tokens"]) for x in items])
            elif work.kind == "move":
                cost += self.move_cost(action.gpu, action.peer, action.payload.get("rebuild") == "on_decode")
            self.available[g.gpu] = max(0., cost-elapsed)
            if work.kind == "score":
                for x in items:
                    self.starts[x["rid"]] = len(x["tokens"])
                    self.score_done[x["rid"]] = self.available[g.gpu]
                    self.score_cache[g.gpu].pop(x["rid"], None)
                    if not x["final"]:
                        self.score_cache[g.gpu][x["rid"]] = self.lane(s.rows[x["rid"]]), len(x["prompt"])+len(x["tokens"])
                trim_score_cache(self.score_cache[g.gpu], cfg["teacher_kv_tokens"])

    def estimate_quantiles(self, rows, history):
        # A batch of conditional medians systematically underestimates its
        # slowest lane. Approximate the order statistics of each live cohort.
        cohorts = defaultdict(list)
        for row in rows:
            cohorts[(row.teacher, len(row.tokens), row.limit)].append(row)
        for (role, emitted, limit), members in cohorts.items():
            lengths = history.get((role, limit), history.get(role, ()))
            start = bisect_right(lengths, emitted)
            count = len(lengths)-start
            if count:
                members.sort(key=lambda row: blake2s(row.rid.encode()).digest())
                for i, row in enumerate(members, 1):
                    index = min(len(lengths)-1, start + count*i//(len(members)+1))
                    self.left[row.rid] = min(limit, lengths[index])-emitted

    def state(self, gpu):
        g = self.s.groups[gpu]
        state = SwitchState(self.cfg, self.classes)
        state.engine = self.classes[g.last_inference_role]
        state.resident = g.trainer_resident
        state.teacher_ready_at = g.teacher_ready_at - self.now
        if g.wake_calibrated is not None:
            state.wake_calibrated = g.wake_calibrated
        return state

    def transition(self, gpu, old, new, publish=False, owner=None, state=None, at=0., advance=False):
        if old == new and not publish:
            return 0., None
        if owner is None:
            owner = gpu == self.s.trainer
        obs = state or self.state(gpu)
        cache_key = gpu, old, new, publish, owner, obs.engine, obs.resident, obs.wake_calibrated
        if cache_key not in self.switch_cache:
            g = self.s.groups[gpu]
            _, cell = fitted_switch(self.profile, obs.keys(old, new, publish, owner),
                                    obs.release_key(old, publish, owner))
            activation = g.teacher_activation_s.get(new)
            if activation is None and new in self.dedicated:
                prior = self.profile.get("teacher_activation_s", {}).get(new,
                        self.cfg.get("sim_teacher_activation_s", {}).get(new))
                activation = self.s.cost.values.get(("teacher_gpu_activation", new), prior)
            if activation is not None and not publish and self.cfg.get("sim_teacher_lifecycle", "lease") == "lease":
                release_key = ("before_teacher", old, new, obs.resident, bool(owner))
                unload = self.s.cost.values.get(release_key, cell["unload"] if cell else
                                                self.cfg.get("sim_teacher_release_s", {}).get(old, 0.))
                t = unload + activation
            elif cell:
                unload = cell["unload"]
                t = cell["unload"] + (cell.get("local_load", cell["load"]) if publish else cell["load"])
            elif old == g.role and new in self.cfg["teachers"]:
                unload = 0.
                t = self.s.teacher_switch_s(g, new)
            else:
                t = self.s.cost.switch(old, new)
                unload = self.cfg.get("sim_teacher_release_s", {}).get(old, 0.)
            if self.s.cost.measuring and not publish:
                values = self.s.cost.values
                t = values.get(("switch_group",gpu,old,new),values.get((old,new),values.get(("switch_to",new),t)))
            self.switch_cache[cache_key] = t, cell.get("device") if cell else None, unload
        t, device, unload = self.switch_cache[cache_key]
        if self.dedicated and old != new:
            release = self.cfg.get("sim_teacher_release_s", {}).get(old, unload)
            ready, wait = teacher_lease_window(obs.teacher_ready_at, at, unload, release, self.prepare_s,
                                               old in self.dedicated, new in self.dedicated)
            if advance:
                obs.teacher_ready_at = ready
            t += wait
        return t, device

    def switch(self, *args, **kwargs):
        return self.transition(*args, **kwargs)[0]

    def train(self, lengths):
        if "train" in self.profile:
            return train_seconds(self.cfg, lengths)
        return sum(lengths) * self.s.cost.get("train_token")

    def train_batches(self, ordered, batch_size):
        key = batch_size, tuple(ordered)
        if key not in self.batch_cache:
            widths = self.train_widths.__getitem__
            groups = batches(ordered, batch_size, self.cfg["train_budget"], widths,
                             self.cfg.get("trainer_attention") == "flash_attention_2")
            self.batch_cache[key] = [(ids, self.train(list(map(widths, ids)))) for ids in groups]
        return self.batch_cache[key]

    def lane(self, row):
        return row.seed // self.n % self.lanes

    def decode_wave(self, rows, left, steps, cold=()):
        lanes = [[] for _ in range(self.lanes)]
        for r in rows:
            emitted = len(r.tokens) + self.left[r.rid] - left[r.rid]
            context = len(r.prompt) + emitted
            features = (context, min(steps, left[r.rid]), context-1 if emitted and r.rid not in cold else 0)
            lanes[self.lane(r)].append((features, context, min(steps, r.limit-emitted)))
        def lane_seconds(lane):
            duration = 0.
            if "decode_rpc" in self.profile:
                ticks = [max(item[1] for item, _ in wave) for wave in decode_windows(lane, self.kv_pages, 16)]
                features = rpc_decode_features((item for item, _, _ in lane), ticks)
                return decode_duration(self.profile["decode_rpc"], features)
            for window in decode_windows(lane, self.kv_pages, 16):
                features = decode_features(item for item, _ in window)
                if "decode" in self.profile:
                    duration += decode_duration(self.profile["decode"], features)
                else:
                    context = sum(item[0] for item, _ in window)
                    rate = max(self.step, self.roof["F"], self.roof["W"] + self.roof["G"]*len(window)
                               + self.roof["A_d"]*context)
                    duration += features[1] * rate
                    duration += self.s.cost.get("prefill_token") * features[4]
            return duration
        return max(map(lane_seconds, lanes))

    def score_time(self, widths, role):
        by_lane = [0] * self.lanes
        for r, width in widths:
            by_lane[self.lane(r)] += max(1, width)
        base, slope = self.score_coeffs[role]
        return base + slope * max(by_lane)

    def move_cost(self, src, dst, deferred=False):
        tokens = sum(len(r.prompt)+len(r.tokens)+self.inflight_tokens.get(r.rid, 0) for r in self.live[src]
                     if self.left[r.rid] > self.inflight_tokens.get(r.rid, 0))
        base = self.profile.get("move_base", self.s.cost.get("move_base"))
        a, b = self.profile.get("move_install", [0., self.s.cost.get("prefill_token")])
        return base + (0. if deferred else a + b*tokens) + self.s.transfer_penalty(src, dst)

    def pairs(self):
        s = self.s
        for src, rows in enumerate(self.live):
            if not rows or any(not r.tokens and r.rid not in self.inflight_tokens for r in rows):
                continue
            for dst, other in enumerate(self.live):
                if not other or src == dst:
                    continue
                if any(s.groups[i].busy and s.groups[i].busy.kind != "decode" for i in (src, dst)):
                    continue
                ids = frozenset(r.rid for r in rows), frozenset(r.rid for r in other)
                if s.rejected_moves.get((src, dst)) == ids:
                    continue
                used = [0] * self.lanes
                survives = False
                for r in chain(rows, other):
                    advanced = self.inflight_tokens.get(r.rid, 0)
                    if self.left[r.rid] > advanced:
                        used[self.lane(r)] += len(r.prompt)+len(r.tokens)+advanced+self.cfg["quantum"]
                        survives |= r.owner == src
                if survives and max(used) <= self.cfg.get("engine_kv_tokens", float("inf")):
                    yield src, dst

    def merge_backfill(self, pair):
        """Find the largest decode chunk predicted to fit before the peer's safe point."""
        ready = [g for g in pair if not self.s.groups[g].busy]
        if len(ready) != 1:
            return None
        gpu, = ready
        other = pair[1] if gpu == pair[0] else pair[0]
        rows = self.live[gpu][:self.cfg["decode_batch"]]
        cold = {r.rid for r in rows if r.deferred_prefill}
        lo, hi = 0, min(self.cfg["quantum"], max(self.left[r.rid] for r in rows))
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self.decode_wave(rows, self.left, mid, cold) <= self.available[other]:
                lo = mid
            else:
                hi = mid-1
        return (gpu, lo) if lo else None

    def layout(self, pair, deferred=False, backfill=None):
        cfg = self.cfg
        free = self.available.copy()
        live = [rows.copy() for rows in self.live]
        left = self.left.copy()
        eos = {}
        next_prefix = {}
        cold = {r.rid for rows in live for r in rows if r.deferred_prefill and r.rid not in self.inflight_tokens}
        for g in self.s.groups:
            if g.busy and g.busy.kind == "decode" and g.gpu == g.busy.gpu:
                for x in g.busy.payload["items"]:
                    rid = x["rid"]
                    left[rid] -= self.inflight_tokens[rid]
                    next_prefix[rid] = (len(x["tokens"])+self.inflight_tokens[rid], free[g.gpu])
                    if not left[rid]:
                        eos[rid] = free[g.gpu]
        if pair:
            src, dst = pair
            if backfill:
                i, steps = backfill
                wave = live[i][:cfg["decode_batch"]]
                free[i] += self.decode_wave(wave, left, steps, cold)
                for r in wave:
                    cold.discard(r.rid)
                    left[r.rid] = max(0, left[r.rid]-steps)
                    next_prefix[r.rid] = len(r.tokens)+self.left[r.rid]-left[r.rid], free[i]
                    if not left[r.rid]:
                        eos[r.rid] = free[i]
                capacity = [0] * self.lanes
                for r in chain(live[src], live[dst]):
                    if left[r.rid]:
                        capacity[self.lane(r)] += len(r.prompt)+len(r.tokens)+self.left[r.rid]-left[r.rid]+cfg["quantum"]
                if max(capacity) > cfg.get("engine_kv_tokens", float("inf")):
                    return None
            install = self.move_cost(src, dst, deferred)
            if backfill and backfill[0] == src and not deferred:
                slope = self.profile.get("move_install", [0., self.s.cost.get("prefill_token")])[1]
                before = sum(len(r.prompt)+len(r.tokens) for r in live[src])
                after = sum(len(r.prompt)+len(r.tokens)+self.left[r.rid]-left[r.rid] for r in live[src] if left[r.rid])
                install += slope*(after-before)
            free[src] = free[dst] = max(free[src], free[dst]) + max(0.,install)
            for r in live[src]:
                if deferred:
                    cold.add(r.rid)
                else:
                    cold.discard(r.rid)
            live[dst] = sorted(chain(live[dst], live[src]), key=lambda r: self.row_order[r.rid])
            live[src] = []
        for i, rows in enumerate(live):
            active = [r for r in rows if left[r.rid]]
            # The controller admits at most decode_batch rows per quantum. Refill the
            # batch as rows finish; independent TP lanes finish at the slowest lane.
            while active:
                wave = active[:cfg["decode_batch"]]
                ticks = min(max(left[r.rid] for r in wave), cfg["quantum"])
                free[i] += self.decode_wave(wave, left, ticks, cold)
                for r in wave:
                    cold.discard(r.rid)
                    left[r.rid] = max(0, left[r.rid]-ticks)
                    next_prefix.setdefault(r.rid, (len(r.tokens)+self.left[r.rid]-left[r.rid], free[i]))
                    if not left[r.rid]:
                        eos[r.rid] = free[i]
                active = [r for r in active if left[r.rid]]
        return free, eos, next_prefix

    def layouts(self, pairs):
        s, model = self.s, self.s.tp_model
        base, eos, prefixes = self.layout(None)
        if model and "decode" not in self.profile and "decode_rpc" not in self.profile:
            for i, rows in enumerate(self.live):
                if len(rows) == 1 and not s.groups[i].busy and rows[0].tokens:
                    r, = rows
                    context = len(r.prompt)+len(r.tokens)
                    base[i] = eos[r.rid] = model.predict(context, self.left[r.rid], model.base)
                    count = min(self.left[r.rid], s.cfg["quantum"])
                    prefixes[r.rid] = len(r.tokens)+count, model.predict(context, count, model.base)
        self.tp_checks = defaultdict(int)
        if not s.tp_pair:
            yield "natural", None, base, eos, prefixes
            for pair in pairs:
                fill = self.merge_backfill(pair) if self.allow_backfill else None
                deferred_ok = s.cfg.get("deferred_migration_prefill", False) and s.cfg.get("engine") == "sglang_fsdp2"
                for deferred in ([False, True] if deferred_ok else [False]):
                    mode = "merge_deferred" if deferred else "merge"
                    yield mode, pair, *self.layout(pair, deferred)
                    if fill:
                        extra = self.layout(pair, deferred, fill)
                        if extra is not None:
                            self.backfills[(mode+"_backfill", pair)] = fill
                            yield mode+"_backfill", pair, *extra
        if not model or not self.allow_wide:
            self.tp_checks["profile_missing" if not model else "disabled"] += 1
            return
        if not s.tp_pair and (s.holds or any(g.busy and g.busy.kind in ("tp_expand", "tp_shrink") for g in s.groups)):
            self.tp_checks["handoff_in_progress"] += 1
            return
        pairs = ([s.tp_pair] if s.tp_pair else
                 ((leader, peer) for leader, rows in enumerate(self.live) if rows for peer in model.partners[leader]))
        for leader, peer in pairs:
            rows, donor = self.live[leader], s.groups[peer]
            leave = model.transition(leader, peer, donor.role, 0, False)["transition_s"]
            if s.tp_pair and not rows:
                free = base.copy()
                free[leader] = free[peer] = max(self.available[leader], self.available[peer])+leave
                yield "shrink", (leader, peer), free, eos, prefixes
                continue
            if len(rows) != 1 or self.live[peer]:
                self.tp_checks["pair_owns_multiple"] += 1
                continue
            r, = rows
            if not r.tokens:
                self.tp_checks["no_committed_prefix"] += 1
                continue
            owner = s.groups[leader]
            if not s.tp_pair:
                if ((owner.busy and owner.busy.kind != "decode")
                        or (donor.busy and donor.busy.kind not in ("score", "train"))):
                    self.tp_checks["role_transition"] += 1
                    continue
                trainer = peer in s.training_groups or peer in (s.trainer, s.required_trainer)
                allow_train = getattr(model, "borrow_trainer", False)
                roles = {"rollout", "standby", *s.cfg["teachers"]}
                if allow_train:
                    roles.add("train")
                if donor.role not in roles or (trainer and not allow_train) or peer in s.queued_work:
                    self.tp_checks["donor_role"] += 1
                    continue
                sizes = s.cfg.get("sim_role_bytes", {})
                extra = (0 if getattr(model, "preserves_donor", False) else
                         max(0, sizes.get("rollout", 16*2**30)-sizes.get(donor.role, 0 if donor.role == "standby" else 16*2**30)))
                if (min(owner.free_bytes, donor.free_bytes-extra) < s.cfg["memory_margin"]+model.scratch or
                        len(r.prompt)+r.limit-1 > model.context_capacity):
                    self.tp_checks["capacity"] += 1
                    continue
            advanced = self.inflight_tokens.get(r.rid, 0)
            remaining = self.left[r.rid] - advanced
            if not remaining and not s.tp_pair:
                self.tp_checks["predicted_finish_in_flight"] += 1
                continue
            context = len(r.prompt)+len(r.tokens)+advanced
            start = max(self.available[leader], self.available[peer])
            enter = 0. if s.tp_pair else model.transition(leader, peer, donor.role, context-1, True)["transition_s"]
            end = start+enter+(model.predict(context, remaining, model.wide) if remaining else 0.)
            free, wide_eos, next_prefix = base.copy(), eos.copy(), prefixes.copy()
            free[leader] = free[peer] = end+leave
            wide_eos[r.rid] = end
            count = min(s.cfg["quantum"], self.left[r.rid])
            if not advanced:
                next_prefix[r.rid] = len(r.tokens)+count, start+enter+model.predict(context, count, model.wide)
            self.tp_checks["evaluated"] += 1
            yield "wide", (leader, peer), free, wide_eos, next_prefix
            if s.tp_pair and not owner.busy:
                suffix = max(0, context-1-s.tp_prefix_tokens)
                handoff = model.transition(leader, peer, donor.role, suffix, False)["transition_s"]
                free, narrow_eos, next_prefix = base.copy(), eos.copy(), prefixes.copy()
                free[peer] = handoff
                free[leader] = narrow_eos[r.rid] = handoff+model.predict(context, remaining, model.base)
                next_prefix[r.rid] = len(r.tokens)+count, handoff+model.predict(context, count, model.base)
                yield "shrink", (leader, peer), free, narrow_eos, next_prefix

    def jobs(self, eos, next_prefix, score_batch, defer):
        cfg = self.cfg
        groups = defaultdict(list)
        for r in self.s.rows.values():
            if r.discarded or r.trained or r.training:
                continue
            start = self.starts[r.rid]
            visible = len(r.tokens)
            end = visible if r.finish else min(visible, start+cfg["score_chunk"])
            if end > start and (r.finish or not defer):
                groups[(r.teacher, 0)].append(Segment(r, start, end, self.score_done.get(r.rid, 0.),
                                                         not r.scoring))
                start = end
            elif not defer and r.rid in next_prefix:
                published, at = next_prefix[r.rid]
                end = min(published, start+cfg["score_chunk"])
                if start < end < visible+self.left[r.rid]:
                    groups[(r.teacher, 1)].append(Segment(r, start, end, at, False))
                    start = end
            end = visible + self.left.get(r.rid, 0)
            if end > start:
                groups[(r.teacher, 2)].append(Segment(r, start, end, eos[r.rid], False))
        jobs = []
        for (role, stage), rows in groups.items():
            rows.sort(key=lambda x: (x.release, not x.row.finish))
            for batch in batches(rows, score_batch, cfg["score_budget"],
                                 lambda x: len(x.row.prompt)+x.end-1, False):
                loads = [0] * self.lanes
                credits, retains = [], {}
                for x in batch:
                    r, lane = x.row, self.lane(x.row)
                    context = len(r.prompt)+x.end
                    loads[lane] += max(1, context-1)
                    if x.start:
                        credits.append((r.rid, lane, len(r.prompt)+x.start))
                    if x.end < len(r.tokens)+self.left.get(r.rid, 0):
                        retains[r.rid] = (lane, context)
                base, slope = self.score_coeffs[role]
                cold = base + slope * max(loads)
                items = ([dict(rid=x.row.rid, end=x.end) for x in batch]
                         if all(x.visible and not x.row.scoring for x in batch) else None)
                jobs.append(ScoreJob(max(x.release for x in batch), stage, role, batch, cold, loads, credits, retains, items))
        return sorted(jobs, key=lambda x: (x.release, x.stage))

    def project(self, free, eos, jobs, scorers, train_batch, invalid_kv, preload=False, home=None):
        s, cfg = self.s, self.cfg
        free, roles = free.copy(), self.role.copy()
        states = [self.state(i) for i in range(self.n)]

        def set_role(i, role, at, owner=None):
            _, device = self.transition(i, roles[i], role, owner=owner, state=states[i], at=at, advance=True)
            states[i].advance(role, device)
            roles[i] = role

        for g in s.groups:
            if g.role != roles[g.gpu]:
                roles[g.gpu] = g.role
                set_role(g.gpu, s.target_role(g), g.busy_since-self.now)
        first = {}
        ready = dict(self.score_done)
        caches = [cache.copy() if i not in invalid_kv and roles[i] == s.groups[i].role else {}
                  for i, cache in enumerate(self.score_cache)]
        generation_end = max(eos.values(), default=0.)
        for job in jobs:
            role, batch = job.role, job.batch
            release = max(job.release, max((ready.get(x.row.rid, 0.) for x in batch), default=0.))
            if cfg.get("phase_barriers"):
                release = max(release, generation_end)
            def switch_start(g):
                return free[g] if preload else max(release, free[g])
            def finish(g):
                at = switch_start(g)
                service = job.cold_s
                if roles[g] == role and caches[g] and job.credits:
                    loads = job.lane_tokens.copy()
                    for rid, lane, context in job.credits:
                        if caches[g].get(rid) == (lane, context):
                            loads[lane] -= context-1
                    base, slope = self.score_coeffs[role]
                    service = base + slope*max(loads)
                activated = at + self.switch(g, roles[g], role, state=states[g], at=at)
                return max(release, activated) + service
            candidates = home.get(role, scorers) if home else scorers
            end, gpu = min((finish(g), g) for g in candidates)
            if gpu not in first:
                first[gpu] = (Action("score", gpu, s.version, dict(teacher=role, items=job.action_items))
                              if job.action_items and not (cfg.get("phase_barriers") and self.live_generation) else None)
                if first[gpu] is None and preload and roles[gpu] != role and free[gpu] < release:
                    first[gpu] = Action("prepare_teacher", gpu, s.version, dict(teacher=role))
            if roles[gpu] != role:
                caches[gpu].clear()
            set_role(gpu, role, switch_start(gpu))
            free[gpu] = end
            for x in batch:
                ready[x.row.rid] = end
                caches[gpu].pop(x.row.rid, None)
            if job.retains:
                caches[gpu].update(job.retains)
                trim_score_cache(caches[gpu], cfg["teacher_kv_tokens"])

        participants = set(s.training_groups)
        owner = s.trainer
        service_end = max(ready.values(), default=0.)
        release_at = {rid: max(t, eos.get(rid, 0.)) for rid, t in ready.items()}
        for rid, t in eos.items():
            release_at.setdefault(rid, t)
        for ready_now in (True, False):
            ordered = self.train_rows[ready_now]
            if not ready_now:
                ordered = sorted(ordered, key=lambda rid: release_at.get(rid, 0.))
            for batch, cost in self.train_batches(ordered, train_batch):
                release = 0. if ready_now else release_at.get(batch[-1], 0.)
                if self.training_barrier:
                    release = max(release, service_end)
                candidates = []
                reserved = participants if owner is None or owner in participants else participants | {owner}
                for gpu in range(self.n):
                    if gpu not in reserved and len(reserved) >= cfg.get("max_train_groups", 1):
                        continue
                    at = max(release, free[gpu])
                    end = at + self.switch(gpu, roles[gpu], "train", owner=owner in (None,gpu), state=states[gpu], at=at) + cost
                    candidates.append((end, gpu))
                end, gpu = min(candidates)
                if gpu not in first:
                    first[gpu] = (Action("train", gpu, s.version, dict(items=[dict(rid=rid) for rid in batch]))
                                  if ready_now and not (self.training_barrier and self.service_pending) else None)
                participants.add(gpu)
                owner = gpu if owner is None else owner
                set_role(gpu, "train", max(release, free[gpu]), owner=gpu == owner)
                free[gpu] = end
        if owner is not None and owner not in participants:
            duration = self.switch(owner, roles[owner], "train", owner=True, state=states[owner], at=free[owner])
            set_role(owner, "train", free[owner], owner=True)
            free[owner] += duration
        end = max(free)
        if s.cost.measuring:
            if participants - {owner}:
                end += s.cost.values.get("replica_reduce", s.cost.get("rollout_publication"))
            return end + s.cost.get("optimizer_step") + s.cost.get("rollout_publication"), first
        if participants - {owner}:
            end += self.profile.get("reduce", s.cost.get("replica_reduce"))
        publish = [self.switch(i, role, "rollout", publish=True, owner=i == owner, state=states[i]) for i, role in enumerate(roles)]
        # A leased teacher can release in slack time before publication; retain it if cheaper.
        for g in s.groups:
            i = g.gpu
            if roles[i] not in g.teacher_activation_s or self.cfg.get("sim_teacher_lifecycle") == "restart":
                continue
            retired = max(end, free[i]+self.switch(i, roles[i], "standby"))
            with_standby = retired-end + self.switch(i, "standby", "rollout", publish=True, owner=i == owner)
            if with_standby < publish[i]:
                publish[i] = with_standby
                first.setdefault(i, Action("standby", i, s.version))
        return end + max(publish) + self.profile.get("tensor_collective_s", 0.), first

    @property
    def live_generation(self):
        return any(self.live)


class CostSearchPolicy:
    plans_standby = True

    def __init__(self, options):
        self.allow_wide = options.get("allow_tp_expansion", True)
        self.allow_backfill = options.get("merge_backfill", True)
        self.live_prefix_scoring = options.get("live_prefix_scoring", True)
        self.early_training = options.get("early_training", True)
        self.placement = options.get("rollout_placement", "round_robin")
        self.route_pools = options.get("route_pools", True)
        if self.placement not in ("balanced", "round_robin"):
            raise ValueError("rollout_placement must be balanced or round_robin")
        self.max_pairs = options.get("max_pairs", 12)
        self.batch_sizes = options.get("train_batch_sizes", [])
        self.score_sizes = options.get("score_batch_sizes", [])
        if type(self.max_pairs) is not int or self.max_pairs < 0 or any(type(n) is not int or n < 1 for n in [*self.batch_sizes, *self.score_sizes]):
            raise ValueError("migration and batch search bounds must be valid")
        self.group_sets = options.get("score_group_sets", [])
        self.classes = None

    def place_rollouts(self, rows, cfg, history):
        if self.placement == "round_robin" or (cfg.get("first_round_calibration") and not cfg.get("sim_worker_profile")):
            return
        groups = cfg["groups"]
        width = cfg.get("group_size", 1)
        replicas = width // cfg.get("inference_tp_size", width)
        profile = cfg.get("sim_worker_profile", {})
        coeffs = profile.get("decode_rpc", profile.get("decode"))
        means = {}
        def demand(row):
            key = row.teacher, row.limit
            if key not in means:
                prior = history.get(key) or history.get(row.teacher) or (cfg.get("tail_prior",row.limit),)
                means[key] = sum(min(n,row.limit) for n in prior) / len(prior)
            n, prompt = means[key], len(row.prompt)
            if coeffs:
                return coeffs[2]*n + coeffs[3]*n*(prompt+(n-1)/2) + coeffs[4]*prompt
            return cfg["costs"]["decode_step"]*n + cfg["costs"]["prefill_token"]*prompt

        lanes = [[] for _ in range(replicas)]
        for row in rows:
            lanes[row.seed // groups % replicas].append((demand(row), row))
        for entries in lanes:
            slots, before, after = [0]*groups, [0.]*groups, [0.]*groups
            for work, row in entries:
                slots[row.owner] += 1
                before[row.owner] += work
            assignment = []
            for work, row in sorted(entries, key=lambda item: -item[0]):
                gpu = min((i for i in range(groups) if slots[i]), key=after.__getitem__)
                after[gpu] += work
                slots[gpu] -= 1
                assignment.append((row, gpu))
            if max(after) < max(before):
                for row, gpu in assignment:
                    row.owner = gpu

    def plan(self, s):
        if all(g.busy for g in s.groups):
            return
        missing_move = s.cost.measuring and "move_base" not in s.cost.values
        if s.cost.measuring and (not s.cfg.get("sim_worker_profile") or (self.max_pairs and missing_move)):
            from .calibration import calibration_round
            if (yield from calibration_round(s, self.max_pairs > 0, self.early_training, self.live_prefix_scoring)):
                return
        started = perf_counter()
        if self.classes is None:
            self.classes = engine_classes(s.cfg)
        f = Forecast(s, self.classes, early_training=self.early_training)
        f.allow_wide = self.allow_wide
        f.allow_backfill = self.allow_backfill
        eligible = tuple(g.gpu for g in s.groups if g.gpu not in s.training_groups)
        pools = list(dict.fromkeys([eligible, *((i,) for i in eligible),
                                    *(tuple(j for j in eligible if j != i) for i in eligible)]))
        if s.cost.measuring:
            # Earliest-finish assignment already compares every eligible worker.
            # Re-enumerating all singleton and leave-one-out pools stalls dispatch.
            pools = [eligible]
        for preset in self.group_sets:
            if not preset or any(type(i) is not int or not 0 <= i < f.n for i in preset):
                raise ValueError("score_group_sets must contain valid group ids")
            pool = tuple(i for i in dict.fromkeys(preset) if i in eligible)
            if pool and pool not in pools:
                pools.append(pool)
        sizes = sorted({s.cfg["microbatch"], *(n for n in self.batch_sizes if n <= s.cfg["microbatch"])})
        if s.cost.measuring:
            # Include a schedule that shares the currently ready tail among
            # all legally idle trainers, beside the normal full microbatch.
            idle_trainers = sum(not g.busy and not f.live[g.gpu] for g in s.groups)
            ready = len(f.train_rows[True])
            if ready and idle_trainers > 1:
                split = min(s.cfg["microbatch"], (ready+idle_trainers-1)//idle_trainers)
                sizes = sorted(set(sizes) | {split})
        holding = next(iter(s.holds.items()), None)
        pairs = list(f.pairs()) if self.max_pairs and not missing_move else []
        if s.cost.measuring and not holding:
            # Bound placement alternatives as the number of groups grows. Rank
            # both the displaced work and the destination's remaining decode.
            def merge_rank(pair):
                src, dst = pair
                return (f.move_cost(src,dst) + max(f.available[src],f.available[dst])
                        + f.step * (max(f.left[r.rid] for r in f.live[dst])
                                    - max(f.left[r.rid] for r in f.live[src])))
            pairs = sorted(pairs,key=merge_rank)[:min(self.max_pairs,isqrt(len(s.groups)))]
        else:
            pairs = sorted(pairs,key=lambda p:f.move_cost(*p))[:self.max_pairs]
        if holding:
            pairs = [holding] if holding in f.pairs() else []
        idle = [g.gpu for g in s.groups if not g.busy]
        wide_possible = s.tp_model and self.allow_wide and any(
            len(rows)==1 and rows[0].tokens and any(not f.live[j] for j in s.tp_model.partners[i])
            for i, rows in enumerate(f.live))
        if not (pairs or holding or s.tp_pair or wide_possible) and all(f.live[i] for i in idle):
            s.record_decode_only(groups=idle, tp_profile=s.tp_model is not None, planner_wall_s=perf_counter()-started)
            for i in idle:
                yield Action("decode", i, s.version, dict(items=[dict(rid=r.rid) for r in
                             f.live[i][:s.cfg["decode_batch"]]], steps=s.cfg["quantum"]))
            return
        # Every executable layout keeps decoding these groups. Launch the RPCs
        # before the combinatorial service search, preserving its settled view.
        reserved = set(chain.from_iterable(pairs)) | set(holding or ()) | set(s.tp_pair or ())
        if wide_possible:
            for i, rows in enumerate(f.live):
                if len(rows) == 1 and rows[0].tokens:
                    for j in s.tp_model.partners[i]:
                        if not f.live[j]:
                            reserved.update((i,j))
        early = [i for i in idle if f.live[i] and i not in reserved]
        if early:
            snapshot = copy(s)
            snapshot.groups = tuple(copy(g) for g in s.groups)
            f.s = snapshot
            for i in early:
                yield Action("decode", i, s.version, dict(items=[dict(rid=r.rid) for r in
                             f.live[i][:s.cfg["decode_batch"]]], steps=s.cfg["quantum"]))
        score_sizes = sorted({s.cfg["score_batch"], *(n for n in self.score_sizes if n <= s.cfg["score_batch"])})
        if s.cost.measuring:
            backlog = defaultdict(int)
            for row in s.rows.values():
                if not row.scoring and not row.discarded and len(row.tokens)>len(row.labels):
                    backlog[row.teacher] += 1
            available = [g for g in s.groups if not g.busy and not f.live[g.gpu] and g.gpu in eligible]
            if backlog and len(available)>1:
                split = (max(backlog.values())+len(available)-1)//len(available)
                score_sizes = sorted(set(score_sizes) | {min(split,s.cfg["score_batch"])})
        best = None
        estimates = []
        home_choice = {}
        for mode, devices, free, eos, prefixes in f.layouts(pairs):
            local_best = None
            best_queue = None
            for score_size in score_sizes:
                deferrals = (False, True) if f.live_generation else (False,)
                for defer in (deferrals if self.live_prefix_scoring else (True,)):
                    jobs = f.jobs(eos, prefixes, score_size, defer)
                    grouped = sorted(jobs, key=lambda x: (x.role not in f.role, x.role, x.stage, x.release))
                    orders = [("release", jobs)]
                    if any(a is not b for a, b in zip(jobs, grouped)):
                        orders.append(("route", grouped))
                    backlog = defaultdict(float)
                    for job in jobs:
                        backlog[job.role] += job.cold_s
                    drain = sorted(jobs, key=lambda x: (x.role not in f.role, backlog[x.role], x.role, x.stage, x.release))
                    if all(any(a is not b for a, b in zip(drain, prior)) for _, prior in orders):
                        orders.append(("drain", drain))
                    for order, queue in orders:
                        queue = ordered_segments(queue)
                        for pool in pools:
                            if queue and not pool:
                                continue
                            for size in sizes:
                                invalid = devices[1:] if mode == "wide" else ()
                                alternatives = ((False, True) if not s.cfg.get("phase_barriers")
                                                and any(j.release > 0 for j in queue) else (False,))
                                for preload in alternatives:
                                    estimate, actions = f.project(free, eos, queue, pool, size, invalid, preload)
                                    if local_best is None or estimate < local_best[0]:
                                        local_best = estimate, actions, pool, size, score_size, defer, order, preload
                                        best_queue = queue
            if local_best is None:
                raise RuntimeError("no feasible group for remaining supervision")
            layout_home = None
            if self.route_pools and not s.cost.measuring and len({j.role for j in best_queue}) > 1:
                # Refine the best unrestricted plan for this layout. Compare the cost of
                # retaining a group for a route against allowing every group to switch.
                _, _, pool, size, score_size, defer, order, preload = local_best
                if len(pool)>1:
                    for role in sorted({j.role for j in best_queue}):
                        for gpu in pool:
                            other = tuple(i for i in pool if i != gpu)
                            home = {j.role: ((gpu,) if j.role == role else other) for j in best_queue}
                            estimate, actions = f.project(free, eos, best_queue, pool, size, invalid,
                                                          preload, home)
                            if estimate < local_best[0]:
                                local_best = estimate, actions, pool, size, score_size, defer, order, preload
                                layout_home = home
            estimates.append(dict(layout=mode, pair=devices, finish_s=local_best[0]))
            if best is None or local_best[0] < best[0]:
                best = *local_best, mode, devices
                home_choice = layout_home
        end, actions, pool, size, score_size, defer, order, preload, mode, devices = best
        pair = devices if mode.startswith("merge") else None
        fill = f.backfills.get((mode, devices))
        s.record_search(layouts=estimates, tp_eligibility=dict(f.tp_checks), early_decode_groups=early,
                        chosen_layout=mode, chosen_pair=devices, score_groups=pool, score_route_groups=home_choice,
                        train_batch=size, score_batch=score_size, defer_live_prefix=defer, score_order=order,
                        preload_in_slack=preload, merge_backfill=fill, predicted_remaining_s=end, planner_wall_s=perf_counter() - started,
                        service={i: a.kind if a else "wait" for i,a in actions.items()})
        if holding and (pair != holding or fill):
            yield Action("unhold", holding[0], s.version, peer=holding[1])
        reserved = set(devices or ())
        if pair:
            src, dst = pair
            if fill:
                gpu, steps = fill
                rows = f.live[gpu][:s.cfg["decode_batch"]]
                yield Action("decode", gpu, s.version, dict(items=[dict(rid=r.rid) for r in rows], steps=steps))
            if not holding or fill:
                yield Action("hold", src, s.version, peer=dst)
            if not any(s.groups[i].busy for i in pair):
                rebuild = "on_decode" if mode == "merge_deferred" else "eager"
                yield Action("move", src, s.version, dict(items=[dict(rid=r.rid) for r in f.live[src]], rebuild=rebuild), peer=dst)
        elif devices and not any(s.groups[i].busy for i in devices):
            src, dst = devices
            if mode == "shrink":
                reason = "service_pressure" if f.live[src] else "tail_finished"
                yield Action("tp_shrink", src, s.version, dict(reason=reason), peer=dst)
            else:
                items = [dict(rid=f.live[src][0].rid)]
                if s.tp_pair:
                    yield Action("decode", src, s.version, dict(items=items, steps=s.cfg["quantum"]), peer=dst)
                else:
                    yield Action("tp_expand", src, s.version, dict(items=items), peer=dst)
        idle_service = []
        for g in s.groups:
            i = g.gpu
            if g.busy or i in reserved:
                continue
            if f.live[i]:
                yield Action("decode", i, s.version, dict(items=[dict(rid=r.rid) for r in
                             f.live[i][:s.cfg["decode_batch"]]], steps=s.cfg["quantum"]))
            elif action := actions.get(i):
                if action.kind == "train" and i not in s.training_groups and len(s.training_groups)+1 == f.n and any(
                        not r.finish or (not r.discarded and len(r.labels) < len(r.tokens)) for r in s.rows.values()):
                    continue
                yield action
            elif s.cost.measuring:
                idle_service.append(g)
        if idle_service and s.cost.measuring and not s.holds:
            from .calibration import share_ready
            yield from share_ready(s,idle_service,early_training=self.early_training,
                                   live_prefix=self.live_prefix_scoring,calibrated=True)
