"""Work-conserving execution while the allocation is collecting its first timing samples."""
from collections import Counter, defaultdict
from math import ceil

from .protocol import Action


def update_profile(controller, events):
    """Fit just the completed round on the current allocation, before the next decision."""
    from ..tools.worker_costs import observations, fit_profile
    cfg, cost = controller.cfg, controller.cost
    cfg["sim_teacher_lifecycle"] = "lease"
    points = list(observations(cfg, events))
    measured = fit_profile(cfg, points)
    profile = cfg.setdefault("sim_worker_profile", {})
    for k, value in measured.items():
        if isinstance(value, dict):
            profile.setdefault(k, {}).update(value)
        else:
            profile[k] = value
    # These scalars are used only for shapes with no fitted curve yet.
    # Prefill is estimated from real cold decode batches until a handoff measures it directly.
    if "prefill_token" not in cost.values:
        cold = [(p["dt"], max(lane[4] for lane in p["lanes"])) for p in points if "lanes" in p]
        cold = [(elapsed, tokens) for elapsed,tokens in cold if tokens > 0]
        if cold:
            cost.observe("prefill_token",sum(t for t,_ in cold)/sum(n for _,n in cold))
    coeffs = list(profile.get("score", {}).values())
    if coeffs:
        cfg["costs"]["score_base"] = sum(x[0] for x in coeffs)/len(coeffs)
        cfg["costs"]["score_token"] = sum(x[1] for x in coeffs)/len(coeffs)
    for k in ("score_base", "score_token"):
        if k in cfg["costs"]:
            cost.values[k] = cfg["costs"][k]
    if "switch" not in cost.values:
        changes=[p["dt"] for p in points if p["op"]=="switch"]
        if changes:
            cost.observe("switch", sum(changes)/len(changes))
    cost.values.setdefault("teacher_wake", cost.values.get("switch", 0.))
    cfg["costs"].update({k:v for k,v in cost.values.items() if isinstance(k,str)})
    controller.emit("cost_calibration", source_version=controller.version,
                    observations=len(points), costs={k:v for k,v in cost.values.items() if isinstance(k,str)},
                    profile=profile)


def share_ready(s, idle, *, early_training=True, live_prefix=True, calibrated=False):
    """Partition real, unclaimed work. Each yield acquires its batch before the next choice."""
    from .search import batches
    cfg = s.cfg
    generating = any(not r.finish for r in s.rows.values())
    if cfg.get("phase_barriers") and generating:
        return
    while idle:
        ready = [r for r in s.rows.values() if r.ready]
        outstanding = any(not r.finish or (not r.discarded and len(r.labels) < len(r.tokens))
                          for r in s.rows.values())
        score = defaultdict(list)
        for r in s.rows.values():
            if (not r.scoring and not r.discarded and len(r.tokens) > len(r.labels)
                    and (live_prefix or r.finish)):
                score[r.teacher].append(r)
        scoring = [g for g in idle if g.gpu not in s.training_groups]
        trainers = []
        if ready and (not outstanding or (early_training and not cfg.get("phase_barriers"))):
            for g in idle:
                reserved = s.training_groups | {g.gpu}
                if s.trainer is not None:
                    reserved |= {s.trainer}
                if len(reserved) <= cfg["max_train_groups"] and (g.gpu in s.training_groups
                        or not outstanding or len(s.training_groups)+1 < len(s.groups)):
                    trainers.append(g)
        if trainers:
            g = min(trainers, key=lambda g: (g.gpu not in s.training_groups, g.role in score,
                                             g.gpu != s.trainer, g.gpu))
            share = min(cfg["microbatch"], ceil(len(ready)/len(trainers)))
            block = next(iter(batches(ready, share, cfg["train_budget"], lambda r: len(r.prompt)+len(r.tokens),
                                      cfg.get("trainer_attention") == "flash_attention_2")))
            action = Action("train", g.gpu, s.version, dict(items=[dict(rid=r.rid) for r in block]))
        elif scoring and score:
            serving = Counter(s.target_role(h) for h in s.groups if h.busy)
            def priority(pair):
                g, role = pair
                backlog = sum(len(r.tokens)-len(r.labels) for r in score[role])
                return (g.role != role, serving[role], -backlog, g.gpu)
            g, role = min(((g,k) for g in scoring for k in score), key=priority)
            rows = sorted(score[role], key=lambda r: (not r.finish, r.kv.get(g.gpu) != len(r.labels)))
            if calibrated and g.role != role:
                # The main cost search already considered cold replicas. Only
                # backfill otherwise idle *resident* engines with visible work.
                break
            peers = sum(h.role == role for h in scoring) or len(scoring)
            share = min(cfg["score_batch"], ceil(len(rows)/peers))
            end = lambda r: len(r.tokens) if r.finish else min(len(r.tokens),len(r.labels)+cfg["score_chunk"])
            block = next(iter(batches(rows, share, cfg["score_budget"], lambda r: len(r.prompt)+end(r)-1, False)))
            action = Action("score",g.gpu,s.version,dict(teacher=role,items=[dict(rid=r.rid,end=end(r)) for r in block]))
        else:
            return
        idle.remove(g)
        yield action


def calibration_round(s, allow_merge, early_training, live_prefix):
    live = defaultdict(list)
    for r in s.rows.values():
        if not r.finish:
            live[r.owner].append(r)
    # Measure one real, feasible handoff. Acknowledged ownership and the worker's
    # memory admission are identical to a normal cost-search migration.
    if allow_merge and not s.holds and "move_base" not in s.cost.values and not any(
            g.busy and g.busy.kind == "move" for g in s.groups):
        candidates = [(a,b) for a in live for b in live if a != b and live[a] and live[b]
                      and not s.groups[a].busy and all(r.tokens for r in live[a])
                      and s.groups[b].role == "rollout" and (not s.groups[b].busy or s.groups[b].busy.kind=="decode")
                      and len(live[a])+len(live[b]) <= s.cfg["decode_batch"]
                      and (frozenset(r.rid for r in live[a]),frozenset(r.rid for r in live[b]))
                      != s.rejected_moves.get((a,b))]
        if candidates:
            a,b=min(candidates,key=lambda pair: (len(live[pair[0]]),len(live[pair[1]])))
            yield Action("hold",a,s.version,peer=b)
    held=set(s.holds)|set(s.holds.values())
    for src,dst in list(s.holds.items()):
        if s.groups[src].busy or s.groups[dst].busy:
            continue
        if live[src] and live[dst]:
            yield Action("move",src,s.version,dict(items=[dict(rid=r.rid) for r in live[src]]),peer=dst)
        else:
            yield Action("unhold",src,s.version,peer=dst)
            held.difference_update((src,dst))
    idle=[]
    for g in s.groups:
        if g.busy or g.gpu in held:
            continue
        if live[g.gpu]:
            yield Action("decode",g.gpu,s.version,dict(steps=s.cfg["quantum"],
                         items=[dict(rid=r.rid) for r in live[g.gpu][:s.cfg["decode_batch"]]]))
        else:
            idle.append(g)
    yield from share_ready(s,idle,early_training=early_training,live_prefix=live_prefix)
