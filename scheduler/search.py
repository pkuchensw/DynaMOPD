"""Receding-horizon list scheduling over a bounded set of alternative resource assignments."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import chain

from .protocol import Action
from ..backends.sim_profile import SwitchState, decode_roofline, engine_classes, fitted_switch, train_seconds


@dataclass(frozen=True)
class Segment:
    row: object
    start: int
    end: int
    release: float
    visible: bool


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
    def __init__(self, s, classes):
        self.s, self.cfg = s, s.cfg
        cfg = self.cfg
        self.n = len(s.groups)
        self.lanes = cfg.get("group_size", 1) // cfg.get("inference_tp_size", cfg.get("group_size", 1))
        self.classes = classes
        self.profile = cfg.get("sim_worker_profile", {})
        self.live = [[] for _ in s.groups]
        self.left = {}
        self.starts = {r.rid: len(r.labels) for r in s.rows.values()}
        self.available = [0.] * self.n
        self.score_done = {}
        self.service_pending = any(not r.finish or (not r.discarded and len(r.labels) < len(r.tokens))
                                   for r in s.rows.values())
        self.role = [s.target_role(g) for g in s.groups]
        self.step = s.cost.get("decode_step")
        self.roof = decode_roofline(cfg)
        self.switch_cache = {}
        for r in s.rows.values():
            if not r.finish:
                self.live[r.owner].append(r)
                self.left[r.rid] = max(1, round(s.remaining(r)))
        for g in s.groups:
            if not g.busy:
                continue
            action = g.busy
            work = s.queued_work.get(g.gpu, action)
            elapsed = s.now() - g.busy_since
            cost = self.switch(g.gpu, g.role, self.role[g.gpu]) if action.kind == "switch" else 0.
            items = work.payload.get("items", [])
            if work.kind == "decode":
                cost += self.step * min(cfg["quantum"], max(self.left[x["rid"]] for x in items))
            elif work.kind == "score":
                widths = [(s.rows[x["rid"]], len(x["tokens"])-x["start"] if
                           s.rows[x["rid"]].kv.get(g.gpu) == x["start"] else len(x["prompt"])+len(x["tokens"])-1)
                          for x in items]
                cost += self.score_time(widths)
            elif work.kind == "train":
                cost += self.train([len(x["prompt"])+len(x["tokens"]) for x in items])
            elif work.kind == "move":
                cost += self.move_cost(action.gpu, action.peer)
            self.available[g.gpu] = max(0., cost-elapsed)
            if work.kind == "score":
                for x in items:
                    self.starts[x["rid"]] = len(x["tokens"])
                    self.score_done[x["rid"]] = self.available[g.gpu]

    def state(self, gpu):
        g = self.s.groups[gpu]
        state = SwitchState(self.cfg, self.classes)
        state.engine = self.classes[g.last_inference_role]
        state.resident = g.trainer_resident
        if g.wake_calibrated is not None:
            state.wake_calibrated = g.wake_calibrated
        return state

    def transition(self, gpu, old, new, publish=False, owner=None, state=None):
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
            if new in g.teacher_activation_s and not publish and self.cfg.get("sim_teacher_lifecycle", "lease") == "lease":
                # Rank launch was measured by this worker, including this backend's import path.
                # Use the same measured component for subsequent transitions in a projected plan.
                release_key = ("before_teacher", old, new, obs.resident, bool(owner))
                unload = self.s.cost.values.get(release_key, cell["unload"] if cell else 0.)
                t = unload + g.teacher_activation_s[new]
            elif cell:
                t = cell["unload"] + (cell.get("local_load", cell["load"]) if publish else cell["load"])
            elif old == g.role and new in self.cfg["teachers"]:
                t = self.s.teacher_switch_s(g, new)
            else:
                t = self.s.cost.switch(old, new)
            self.switch_cache[cache_key] = t, cell.get("device") if cell else None
        return self.switch_cache[cache_key]

    def switch(self, *args, **kwargs):
        return self.transition(*args, **kwargs)[0]

    def train(self, lengths):
        if "train" in self.profile:
            return train_seconds(self.cfg, lengths)
        return sum(lengths) * self.s.cost.get("train_token")

    def lane(self, row):
        return row.seed // self.n % self.lanes

    def score_time(self, widths):
        by_lane = [0] * self.lanes
        for r, width in widths:
            by_lane[self.lane(r)] += max(1, width)
        costs = self.cfg["costs"]
        return costs["score_base"] + costs["score_token"] * max(by_lane)

    def move_cost(self, src, dst):
        tokens = sum(len(r.prompt)+len(r.tokens) for r in self.live[src])
        base = self.profile.get("move_base", self.s.cost.get("move_base"))
        a, b = self.profile.get("move_install", [0., self.s.cost.get("prefill_token")])
        return base + a + b*tokens + self.s.transfer_penalty(src, dst)

    def pairs(self):
        s = self.s
        for src, rows in enumerate(self.live):
            if not rows or any(not r.tokens for r in rows):
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
                for r in chain(rows, other):
                    used[self.lane(r)] += len(r.prompt)+len(r.tokens)+self.cfg["quantum"]
                if max(used) <= self.cfg.get("engine_kv_tokens", float("inf")):
                    yield src, dst

    def layout(self, pair):
        cfg = self.cfg
        free = self.available.copy()
        live = [rows.copy() for rows in self.live]
        left = self.left.copy()
        eos = {}
        for g in self.s.groups:
            if g.busy and g.busy.kind == "decode" and g.gpu == g.busy.gpu:
                for x in g.busy.payload["items"]:
                    rid = x["rid"]
                    left[rid] = max(0, left[rid]-cfg["quantum"])
                    if not left[rid]:
                        eos[rid] = free[g.gpu]
        if pair:
            src, dst = pair
            free[src] = free[dst] = max(free[src], free[dst]) + self.move_cost(src, dst)
            live[dst].extend(live[src])
            live[src] = []
        for i, rows in enumerate(live):
            active = [r for r in rows if left[r.rid]]
            # The controller admits at most decode_batch rows per quantum. Refill the
            # batch as rows finish; independent TP lanes finish at the slowest lane.
            while active:
                wave = active[:cfg["decode_batch"]]
                ticks = min(max(left[r.rid] for r in wave), cfg["quantum"])
                loads = [0] * self.lanes
                context = [0] * self.lanes
                for r in wave:
                    lane = self.lane(r)
                    loads[lane] += 1
                    context[lane] += len(r.prompt)+len(r.tokens)+self.left[r.rid]-left[r.rid]
                rate = max(self.step, self.roof["F"], max(self.roof["W"] + self.roof["G"]*n +
                           self.roof["A_d"]*ctx for n, ctx in zip(loads, context)))
                free[i] += ticks * rate
                for r in wave:
                    left[r.rid] = max(0, left[r.rid]-ticks)
                    if not left[r.rid]:
                        eos[r.rid] = free[i]
                active = [r for r in active if left[r.rid]]
        return free, eos

    def layouts(self, pairs):
        s, model = self.s, self.s.tp_model
        rows = list(chain.from_iterable(self.live))
        base, eos = self.layout(None)
        if not s.tp_pair:
            if model and len(rows) == 1 and not s.groups[rows[0].owner].busy:
                r = rows[0]
                end = model.predict(len(r.prompt)+len(r.tokens), self.left[r.rid], model.base)
                base[r.owner] = eos[r.rid] = end
            yield "natural", None, base, eos
            for pair in pairs:
                free, ends = self.layout(pair)
                yield "merge", pair, free, ends
        if not model or s.holds:
            return
        if s.tp_pair and not rows:
            leader, peer = s.tp_pair
            delay = model.transition(leader, peer, s.groups[peer].role, 0, False)["transition_s"]
            base[leader] = base[peer] = max(base[leader], base[peer])+delay
            yield "shrink", s.tp_pair, base, eos
            return
        # Each executable wide layout serves one live trajectory on a local pair.
        if len(rows) != 1 or not rows[0].tokens:
            return
        r = rows[0]
        leader = r.owner
        owner = s.groups[leader]
        context = len(r.prompt)+len(r.tokens)
        for peer in ([s.tp_pair[1]] if s.tp_pair else model.partners[leader]):
            donor = s.groups[peer]
            if not s.tp_pair:
                trainer_donor = peer in s.training_groups or peer in (s.trainer, s.required_trainer)
                retains_donor = getattr(model, "preserves_donor", False)
                eligible_roles = {"rollout", "standby", *s.cfg["teachers"]}
                if getattr(model, "borrow_trainer", False):
                    eligible_roles.add("train")
                elif trainer_donor:
                    continue
                if owner.busy or donor.busy or donor.role not in eligible_roles:
                    continue
                sizes = s.cfg.get("sim_role_bytes", {})
                extra = max(0, sizes.get("rollout", 16*2**30)-sizes.get(donor.role, 0 if donor.role == "standby" else 16*2**30))
                if (min(owner.free_bytes, donor.free_bytes-(0 if retains_donor else extra)) < s.cfg["memory_margin"]+model.scratch or
                        len(r.prompt)+r.limit-1 > model.context_capacity):
                    continue
            start = max(self.available[leader], self.available[peer])
            remaining = self.left[r.rid]
            if owner.busy and owner.busy.kind == "decode":
                remaining = max(0, remaining-s.cfg["quantum"])
            enter = 0. if s.tp_pair else model.transition(leader, peer, donor.role, context-1, True)["transition_s"]
            leave = model.transition(leader, peer, donor.role, 0, False)["transition_s"]
            end = start+enter+(model.predict(context, remaining, model.wide) if remaining else 0.)
            free = base.copy()
            free[leader] = free[peer] = end+leave
            yield "wide", (leader, peer), free, {r.rid: end}
            if s.tp_pair and not owner.busy:
                moved = max(0, context-1-s.tp_prefix_tokens)
                handoff = model.transition(leader, peer, donor.role, moved, False)["transition_s"]
                free = base.copy()
                free[peer] = handoff
                free[leader] = handoff+model.predict(context, remaining, model.base)
                yield "shrink", s.tp_pair, free, {r.rid: free[leader]}

    def jobs(self, eos, score_batch, defer):
        cfg = self.cfg
        groups = defaultdict(list)
        for r in self.s.rows.values():
            if r.discarded or r.trained or r.training:
                continue
            start = self.starts[r.rid]
            visible = len(r.tokens)
            end = visible if r.finish else min(visible, start+cfg["score_chunk"])
            if end > start and (r.finish or not defer):
                groups[(r.teacher, True)].append(Segment(r, start, end, self.score_done.get(r.rid, 0.),
                                                         not r.scoring))
                start = end
            end = visible + self.left.get(r.rid, 0)
            if end > start:
                groups[(r.teacher, False)].append(Segment(r, start, end, eos[r.rid], False))
        jobs = []
        for (role, visible), rows in groups.items():
            rows.sort(key=lambda x: (x.release, not x.row.finish))
            for batch in batches(rows, score_batch, cfg["score_budget"],
                                 lambda x: len(x.row.prompt)+x.end-1, False):
                jobs.append((max(x.release for x in batch), not visible, role, batch))
        return sorted(jobs, key=lambda x: (x[0], x[1]))

    def project(self, free, eos, jobs, scorers, train_batch, invalid_kv):
        s, cfg = self.s, self.cfg
        free, roles = free.copy(), self.role.copy()
        states = [self.state(i) for i in range(self.n)]

        def set_role(i, role, owner=None):
            _, device = self.transition(i, roles[i], role, owner=owner, state=states[i])
            states[i].advance(role, device)
            roles[i] = role

        for g in s.groups:
            if g.role != roles[g.gpu]:
                roles[g.gpu] = g.role
                set_role(g.gpu, s.target_role(g))
        first = {}
        ready = dict(self.score_done)
        invalid_kv = set(invalid_kv)
        generation_end = max(eos.values(), default=0.)
        for release, future, role, batch in jobs:
            release = max(release, max((ready.get(x.row.rid, 0.) for x in batch), default=0.))
            if cfg.get("phase_barriers"):
                release = max(release, generation_end)
            def finish(g):
                widths = [(x.row, x.end-x.start if g not in invalid_kv and roles[g] == role and x.row.kv.get(g) == x.start
                           else len(x.row.prompt)+x.end-1) for x in batch]
                return max(release, free[g]) + self.switch(g, roles[g], role, state=states[g]) + self.score_time(widths)
            end, gpu = min((finish(g), g) for g in scorers)
            actionable = all(x.visible and not x.row.scoring for x in batch)
            first.setdefault(gpu, Action("score", gpu, s.version, dict(teacher=role, items=[
                dict(rid=x.row.rid, end=x.end) for x in batch])) if actionable and not (
                    cfg.get("phase_barriers") and self.live_generation) else None)
            if roles[gpu] != role:
                invalid_kv.add(gpu)
            free[gpu] = end
            set_role(gpu, role)
            for x in batch:
                ready[x.row.rid] = end

        pending = [r for r in s.rows.values() if not (r.trained or r.training or r.discarded)]
        pending.sort(key=lambda r: (not r.ready, max(eos.get(r.rid, 0.), ready.get(r.rid, 0.))))
        participants = set(s.training_groups)
        owner = s.trainer
        width = lambda r: len(r.prompt)+len(r.tokens)+self.left.get(r.rid, 0)
        service_end = max(ready.values(), default=0.)
        for ready_now in (True, False):
            for batch in batches((r for r in pending if r.ready == ready_now), train_batch, cfg["train_budget"],
                                 width, cfg.get("trainer_attention") == "flash_attention_2"):
                release = max(max(eos.get(r.rid, 0.), ready.get(r.rid, 0.)) for r in batch)
                if cfg.get("phase_barriers"):
                    release = max(release, service_end)
                cost = self.train([width(r) for r in batch])
                candidates = []
                for gpu in range(self.n):
                    reserved = participants | {gpu} | ({owner} if owner is not None else set())
                    if len(reserved) > cfg.get("max_train_groups", 1):
                        continue
                    end = max(release, free[gpu]) + self.switch(gpu, roles[gpu], "train", owner=owner in (None,gpu), state=states[gpu]) + cost
                    candidates.append((end, gpu))
                end, gpu = min(candidates)
                first.setdefault(gpu, Action("train", gpu, s.version, dict(items=[dict(rid=r.rid) for r in batch]))
                                 if ready_now and not (cfg.get("phase_barriers") and self.service_pending) else None)
                participants.add(gpu)
                owner = gpu if owner is None else owner
                free[gpu] = end
                set_role(gpu, "train", owner=gpu == owner)
        if owner is not None and owner not in participants:
            free[owner] += self.switch(owner, roles[owner], "train", owner=True, state=states[owner])
            set_role(owner, "train", owner=True)
        end = max(free)
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
        self.max_pairs = options.get("max_pairs", 12)
        self.batch_sizes = options.get("train_batch_sizes", [])
        self.score_sizes = options.get("score_batch_sizes", [])
        if type(self.max_pairs) is not int or self.max_pairs < 0 or any(type(n) is not int or n < 1 for n in [*self.batch_sizes, *self.score_sizes]):
            raise ValueError("migration and batch search bounds must be valid")
        self.group_sets = options.get("score_group_sets", [])
        self.classes = None

    def plan(self, s):
        if self.classes is None:
            self.classes = engine_classes(s.cfg)
        f = Forecast(s, self.classes)
        eligible = tuple(g.gpu for g in s.groups if g.gpu not in s.training_groups)
        pools = list(dict.fromkeys([eligible, *((i,) for i in eligible),
                                    *(tuple(j for j in eligible if j != i) for i in eligible)]))
        for preset in self.group_sets:
            if not preset or any(type(i) is not int or not 0 <= i < f.n for i in preset):
                raise ValueError("score_group_sets must contain valid group ids")
            pool = tuple(i for i in dict.fromkeys(preset) if i in eligible)
            if pool and pool not in pools:
                pools.append(pool)
        sizes = sorted({s.cfg["microbatch"], *(n for n in self.batch_sizes if n <= s.cfg["microbatch"])})
        holding = next(iter(s.holds.items()), None)
        pairs = sorted(f.pairs(), key=lambda p: f.move_cost(*p))[:self.max_pairs]
        if holding:
            pairs = [holding] if holding in f.pairs() else []
        score_sizes = sorted({s.cfg["score_batch"], *(n for n in self.score_sizes if n <= s.cfg["score_batch"])})
        best = None
        estimates = []
        for mode, devices, free, eos in f.layouts(pairs):
            local_best = None
            for score_size in score_sizes:
                for defer in ((False, True) if f.live_generation else (False,)):
                    jobs = f.jobs(eos, score_size, defer)
                    grouped = sorted(jobs, key=lambda x: (x[2] not in f.role, x[2], x[1], x[0]))
                    orders = [("release", jobs)]
                    if any(a is not b for a, b in zip(jobs, grouped)):
                        orders.append(("route", grouped))
                    for order, queue in orders:
                        for pool in pools:
                            if queue and not pool:
                                continue
                            for size in sizes:
                                invalid = devices[1:] if mode == "wide" else ()
                                estimate, actions = f.project(free, eos, queue, pool, size, invalid)
                                if local_best is None or estimate < local_best[0]:
                                    local_best = estimate, actions, pool, size, score_size, defer, order
            if local_best is None:
                raise RuntimeError("no feasible group for remaining supervision")
            estimates.append(dict(layout=mode, pair=devices, finish_s=local_best[0]))
            if best is None or local_best[0] < best[0]:
                best = *local_best, mode, devices
        end, actions, pool, size, score_size, defer, order, mode, devices = best
        pair = devices if mode == "merge" else None
        s.record_search(layouts=estimates, chosen_layout=mode, chosen_pair=devices, score_groups=pool,
                        train_batch=size, score_batch=score_size, defer_live_prefix=defer, score_order=order,
                        predicted_remaining_s=end,
                        service={i: a.kind if a else "wait" for i,a in actions.items()})
        if holding and pair != holding:
            yield Action("unhold", holding[0], s.version, peer=holding[1])
        reserved = set(devices or ())
        if pair:
            src, dst = pair
            if not holding:
                yield Action("hold", src, s.version, peer=dst)
            if not any(s.groups[i].busy for i in pair):
                yield Action("move", src, s.version, dict(items=[dict(rid=r.rid) for r in f.live[src]]), peer=dst)
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
