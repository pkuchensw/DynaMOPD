"""An online greedy joint policy over acknowledged state; each yield claims its resources."""
from collections import defaultdict

from .protocol import Action
from .base import PlanningState
from ..components import resolve


def build_policy(name, options):
    if name == "cost_search":
        from .search import CostSearchPolicy
        return CostSearchPolicy(options)
    if name in ("natural", "merge", "elastic_tp"):
        if options:
            raise ValueError("built-in policies read their tuning parameters from the scheduling config")
        return AdaptivePolicy(allow_merge=name != "natural", elastic_tp=name == "elastic_tp")
    return resolve(name, {})(options)


class AdaptivePolicy:
    def __init__(self, allow_merge=True, elastic_tp=False):
        self.allow_merge = allow_merge
        self.elastic_tp = elastic_tp

    def plan(self, s: PlanningState):
        if s.cost.measuring and not s.cfg.get("sim_worker_profile"):
            from .calibration import calibration_round
            yield from calibration_round(s, self.allow_merge, True, True)
            return
        cfg, v = s.cfg, s.version
        active = defaultdict(list)
        for row in s.rows.values():
            if not row.finish:
                active[row.owner].append(row)

        # Resolve a pending two-party handoff only after BOTH decode operations finish.
        for src, dst in list(s.holds.items()):
            if s.groups[src].busy or s.groups[dst].busy:
                continue
            if active[src] and active[dst] and len(active[src])+len(active[dst]) <= cfg["decode_batch"]:
                yield Action("move", src, v, {"items": [{"rid": r.rid} for r in active[src]]}, peer=dst)
            else:
                yield Action("unhold", src, v, peer=dst)

        held = set(s.holds) | set(s.holds.values())
        tp_reserved = set()
        if self.elastic_tp:
            from .tail_tp import tail_action, wait_for_tail_boundary
            tp_reserved.update(g.gpu for g in s.groups if g.busy and g.busy.kind == "tp_expand")
            action = tail_action(s, active)
            if action is not None:
                yield action
                tp_reserved.update(action.devices)
            elif getattr(s.tp_model,"borrow_trainer",False):
                donor = wait_for_tail_boundary(s,active)
                if donor is not None:
                    tp_reserved.add(donor)
            if s.tp_pair:
                tp_reserved.update(s.tp_pair)
            held.update(tp_reserved)
        idle = [g for g in s.groups if not active[g.gpu] and not g.busy and g.gpu not in held]
        phase_barriers = cfg.get("phase_barriers", False)
        generating = any(active.values())

        def waiting():
            if phase_barriers and generating:
                return []
            return [r for r in s.rows.values() if not r.scoring and not r.discarded
                    and len(r.tokens) > len(r.labels)]

        def scoring_options(g):
            by_teacher = defaultdict(list)
            for r in waiting():
                by_teacher[r.teacher].append(r)
            for teacher, rows in by_teacher.items():
                def priority(r):
                    return (r.finish is not None, r.kv.get(g.gpu, -1) == len(r.labels),
                            len(r.tokens)-len(r.labels))
                ordered = sorted(rows, key=priority, reverse=True)
                selected, work, padded = [], 0, 0
                for r in ordered:
                    start = len(r.labels)
                    hit = r.kv.get(g.gpu, -1) == start
                    chunk = cfg["score_chunk"]
                    if r.finish:
                        # Score the remaining terminal suffix in one pass when
                        # it fits, so the completed sample can enter training.
                        prefix = 0 if hit else len(r.prompt) + start - 1
                        chunk = max(chunk, cfg["score_budget"] - prefix)
                    end = min(len(r.tokens), start + chunk)
                    width = end-start if hit else len(r.prompt)+end-1
                    if selected and max(padded, width)*(len(selected)+1) > cfg["score_budget"]:
                        continue
                    x = dict(rid=r.rid, end=end, final=r.finish is not None and end == len(r.tokens))
                    selected.append(x)
                    padded = max(padded, width)
                    work += end-start
                    if len(selected) == cfg["score_batch"]:
                        break
                cold = g.role != teacher
                switch = s.teacher_switch_s(g, teacher)
                if g.prepared_teachers is not None and teacher not in g.prepared_teachers:
                    # A prepared replica can serve as soon as its current tail
                    # drains. Start a new process only when that wait costs more.
                    alternatives = [h for h in s.groups if h.gpu not in tp_reserved and h.gpu not in s.training_groups
                                    and h.prepared_teachers
                                    and teacher in h.prepared_teachers and s.target_role(h) != "train"]
                    wait = min((max((s.remaining(r) for r in active[h.gpu]), default=0)
                                * s.cost.get("decode_step") + s.teacher_switch_s(h, teacher)
                                for h in alternatives), default=float("inf"))
                    if wait < switch:
                        continue
                now = s.now()
                age = now-min(r.pending_since for r in rows)
                live_routes = [r for r in s.rows.values() if r.finish is None]
                next_prefix = cfg["quantum"]*s.cost.get("decode_step")
                if cold and any(r.teacher==g.role for r in live_routes):
                    amortization = switch + max(0.,switch-(now-g.resident_since))
                    unlocks_trainer = not s.training_groups and any(x["final"] for x in selected)
                    if not unlocks_trainer and next_prefix < switch and work*s.cost.get("score_token")+age < amortization:
                        continue
                if (not cold and len(selected)<cfg["score_batch"] and not any(x["final"] for x in selected)
                        and age < min(next_prefix,s.cost.get("score_base"))
                        and any(r.teacher==teacher and len(r.tokens)==len(r.labels) for r in live_routes)):
                    continue
                warm_elsewhere = any(h.gpu != g.gpu and h.gpu not in tp_reserved and s.target_role(h) == teacher for h in s.groups)
                # An unloaded replica may wait for a small queue already served by a resident one.
                # Never defer the last serviceable route or an EOS that would unblock training.
                if cold and warm_elsewhere and active and not any(r.finish for r in rows):
                    if work*s.cost.get("score_token") < switch:
                        continue
                delay = s.cost.get("score_base") + padded*len(selected)*s.cost.get("score_token")
                if cold:
                    cost_per_batch = switch
                    other_owners = [h for h in s.groups if h.gpu != g.gpu and s.target_role(h) == teacher]
                    if teacher in g.teacher_activation_s and not other_owners:
                        # One activation serves the visible queue. Charging its full price to
                        # the first microbatch alone can defer an entire route until after EOS.
                        batches = max(1., sum(len(r.tokens)-len(r.labels) for r in rows) / work)
                        cost_per_batch /= batches
                    if g.role in g.teacher_activation_s and any(r.teacher == g.role for r in waiting()):
                        if not any(h.gpu != g.gpu and s.target_role(h) == g.role for h in s.groups):
                            cost_per_batch += g.teacher_activation_s[g.role]
                    delay += cost_per_batch
                benefit = work + cfg["quantum"] * sum(x["final"] for x in selected)
                yield benefit/delay, Action("score", g.gpu, v, {"teacher":teacher, "items":selected})

        # Compare assignments globally, including keeping a teacher resident while a peer changes role.
        while idle:
            options = []
            ready = [r for r in s.rows.values() if r.ready]
            replicas = s.training_groups
            owner_available = any(g.gpu == s.trainer for g in idle)
            backlog = sum(len(r.prompt) + len(r.tokens) for r in ready)
            needs_service = any(not r.finish or (not r.discarded and len(r.labels) < len(r.tokens))
                                for r in s.rows.values())
            if phase_barriers and needs_service:
                ready = []
            batch, width, tokens = [], 0, 0
            packed = cfg.get("trainer_attention") == "flash_attention_2"
            for row in ready:
                length = len(row.prompt) + len(row.tokens)
                new_width = max(width, length)
                cost = tokens + length if packed else new_width * (len(batch) + 1)
                if batch and cost > cfg["train_budget"]:
                    continue
                batch.append({"rid": row.rid})
                width = new_width
                tokens += length
                if len(batch) == cfg["microbatch"]:
                    break
            backward_time = s.cost.get("train_token") * (tokens if packed else width * len(batch))
            for g in idle:
                if g.gpu not in replicas:
                    options.extend(scoring_options(g))
                if not batch:
                    continue
                if not replicas and owner_available and g.gpu != s.trainer:
                    # Let an idle optimizer take the first microbatch. If it
                    # takes a teacher batch instead, peers can start immediately.
                    continue
                if g.gpu not in replicas:
                    k = len(replicas)
                    claimed = replicas | {g.gpu}
                    if s.trainer is not None:
                        claimed = claimed | {s.trainer}
                    if len(claimed) > cfg.get("max_train_groups", 1) or (needs_service and k + 1 == len(s.groups)):
                        continue
                    if generating and waiting() and not any(
                        h.gpu != g.gpu and h.gpu not in replicas and h.gpu not in held and not active[h.gpu]
                        for h in s.groups
                    ):
                        # A decoding group cannot serve teacher batches. Keep
                        # a drained group scoring while the last rollout runs.
                        continue
                if g.gpu not in replicas and replicas:
                    # Saving the difference between k and k+1 replicas must
                    # repay activation and the end-of-round gradient reduction.
                    saving = backlog * s.cost.get("train_token") / (k * (k + 1))
                    if saving <= s.cost.switch(g.role, "train") + s.cost.get("replica_reduce"):
                        continue
                    # Preserve a ready teacher if its outstanding work costs
                    # more to relocate than we expect to save by training here.
                    if any(r.teacher == g.role for r in waiting()):
                        alternatives = [h for h in s.groups if h.gpu != g.gpu and h.gpu not in replicas
                                        and h.gpu not in tp_reserved]
                        if not alternatives or min(s.teacher_switch_s(h, g.role) for h in alternatives) > saving:
                            continue
                duration = backward_time
                if g.role != "train":
                    duration += s.cost.switch(g.role, "train")
                benefit = (backlog + cfg["quantum"]*len(ready)) / (1 + len(replicas - {g.gpu}))
                options.append((benefit/max(duration, 1e-6), Action("train", g.gpu, v, {"items": batch})))
            if not options:
                break
            _, best = max(options, key=lambda pair: pair[0])
            idle = [g for g in idle if g.gpu != best.gpu]
            yield best

        # Request a safe point on both engines. The real path sees progress and historical
        # durations only. The executor can reject an install if device memory has changed.
        if self.allow_merge and not s.holds and (not s.cost.measuring or "move_base" in s.cost.values):
            sparse = [g for g in s.groups if active[g.gpu] and g.role == "rollout" and g.gpu not in held
                      and (not g.busy or g.busy.kind == "decode")
                      and all(r.tokens for r in active[g.gpu])]
            candidates = []
            service_backlog = sum(len(r.tokens)-len(r.labels) for r in waiting())
            ready_count = sum(r.ready for r in s.rows.values())
            for src in sparse:
                for dst in sparse:
                    if src == dst or len(active[src.gpu])+len(active[dst.gpu]) > cfg["decode_batch"]:
                        continue
                    if dst.gpu == s.required_trainer and (ready_count or service_backlog):
                        continue
                    rejected = s.rejected_moves.get((src.gpu,dst.gpu))
                    if rejected and rejected == (frozenset(r.rid for r in active[src.gpu]),
                                                  frozenset(r.rid for r in active[dst.gpu])):
                        continue
                    items = active[src.gpu]
                    context = sum(len(r.prompt)+len(r.tokens) for r in items)
                    if dst.free_bytes < cfg["memory_margin"] + 3*cfg["kv_bytes_per_token"]*context:
                        continue
                    remaining = max(s.remaining(r) for r in items)
                    saved = remaining*s.cost.get("decode_step")
                    rebuild = s.cost.get("move_base") + s.transfer_penalty(src.gpu, dst.gpu) + context*s.cost.get("prefill_token")
                    pressure = service_backlog*s.cost.get("score_token") + ready_count*s.cost.get("switch")
                    # Freeing a card buys at most its expected tail, even under large backlog.
                    value = min(saved, pressure) - rebuild
                    if value > 0:
                        candidates.append((value, -context, src.gpu, dst.gpu))
            if candidates:
                _, _, src, dst = max(candidates)
                yield Action("hold", src, v, peer=dst)
                held.update((src, dst))

        for g in s.groups:
            if not g.busy and g.gpu not in held and active[g.gpu]:
                items = [{"rid": r.rid} for r in active[g.gpu][:cfg["decode_batch"]]]
                yield Action("decode", g.gpu, v, dict(items=items, steps=cfg["quantum"]))
