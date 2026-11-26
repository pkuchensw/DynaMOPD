"""Compare narrow and wide decoding of the final live request at a token boundary."""
from .protocol import Action


def deferred_service(s, groups, horizon, scoring, training, now=None):
    """Approximate queued work left uncovered during the predicted decode window."""
    available = {}
    if now is None:
        now = s.now()
    for g in groups:
        busy = g.busy
        cost = 0.
        if busy:
            items = busy.payload.get("items", [])
            if busy.kind == "score":
                cost = s.cost.get("score_base") + sum(len(x["tokens"])-x["start"] for x in items)*s.cost.get("score_token")
            elif busy.kind == "train":
                cost = sum(len(x["prompt"])+len(x["tokens"]) for x in items)*s.cost.get("train_token")
            elif busy.kind == "switch":
                cost = s.cost.switch(g.role, busy.payload["role"])
            else:
                available[g.gpu] = 0.
                continue
            cost = max(0., cost - max(0., now - g.busy_since))
        available[g.gpu] = max(0., horizon - cost)
    trainer = s.trainer if s.trainer is not None else s.required_trainer
    if trainer is None and training and available:
        trainer = max(available, key=available.get)
    trainers = set(s.training_groups)
    if trainer is not None:
        trainers.add(trainer)
    trained = min(training, sum(available.get(gpu, 0.) for gpu in trainers))
    # Once activated this round, the accumulator cannot be lent to a teacher.
    score_capacity = sum(seconds for gpu, seconds in available.items() if gpu not in trainers)
    return training - trained + max(0., scoring - score_capacity)


def tail_action(s, active):
    rows = [row for group in active.values() for row in group]
    if s.tp_pair:
        leader, peer = s.tp_pair
        return pair_action(s, rows, leader, peer)
    if len(rows) != 1 or not rows[0].tokens or s.holds:
        return
    leader = rows[0].owner
    options = (pair_action(s, rows, leader, peer) for peer in s.tp_model.partners.get(leader, ())
               if getattr(s.tp_model,"borrow_trainer",False) or (peer not in s.training_groups and peer not in (s.trainer,s.required_trainer)))
    return max((a for a in options if a is not None), key=lambda a: a.payload["predicted_gain_s"], default=None)


def pair_action(s, rows, leader, peer, *, lookahead=False):
    model = s.tp_model
    pair = s.tp_pair
    owner, donor = s.groups[leader], s.groups[peer]
    if (owner.busy and not lookahead) or donor.busy:
        return
    if not rows:
        return Action("tp_shrink", leader, s.version, {"reason": "tail_finished"}, peer)
    row, = rows
    preserved = getattr(model,"preserves_donor",False)
    if owner.role != "rollout" or (donor.role not in {"rollout","standby",*s.cfg["teachers"]}
                                   and not (preserved and donor.role=="train")):
        return
    if not pair:
        role_bytes = s.cfg.get("sim_role_bytes", {})
        donor_bytes = role_bytes.get(donor.role, 0 if donor.role == "standby" else 16*2**30)
        extra = max(0, role_bytes.get("rollout", 16*2**30) - donor_bytes)
        free = min(owner.free_bytes, donor.free_bytes-(0 if preserved else extra))
        if free < s.cfg["memory_margin"] + model.scratch:
            if preserved:
                s.record_tp_candidate(leader=leader,peer=peer,rid=row.rid,reason="capacity",
                                      free_bytes=free,required_bytes=s.cfg["memory_margin"]+model.scratch)
            return
        if len(row.prompt) + row.limit - 1 > model.context_capacity:
            return
    remaining = s.remaining(row)
    context = len(row.prompt) + len(row.tokens)
    narrow = model.predict(context, remaining, model.base)
    wide = model.predict(context, remaining, model.wide)
    enter = 0. if pair else model.transition(leader, peer, donor.role, context-1, True)["transition_s"]
    # At EOS no live student KV has to return to the original TP width.
    leave = model.transition(leader, peer, donor.role, 0, False)["transition_s"]
    other = [g for g in s.groups if g.gpu not in (leader, peer)]
    pending = sum(len(r.tokens)-len(r.labels) for r in s.rows.values() if not r.discarded and not r.scoring)
    score_s = (pending + remaining) * s.cost.get("score_token")
    train_s = sum(len(r.prompt)+len(r.tokens) for r in s.rows.values()
                  if r.finish and not (r.discarded or r.training or r.trained)) * s.cost.get("train_token")
    now = s.now()
    blocked = max(0., deferred_service(s, other, wide, score_s, train_s, now)
                  - deferred_service(s, [*other, donor], wide, score_s, train_s, now))
    lost_kv = sum(len(r.prompt)+r.kv[peer]-1 for r in s.rows.values()
                  if not pair and peer in r.kv and not r.discarded and (not r.finish or len(r.labels)<len(r.tokens)))
    reprefill = (0 if preserved else lost_kv) * s.cost.get("prefill_token")
    gain = narrow - wide - enter - leave - blocked - reprefill
    estimates = dict(narrow_decode_s=narrow, wide_decode_s=wide, enter_s=enter, exit_s=leave,
                     blocked_service_s=blocked, lost_teacher_prefill_s=reprefill,
                     predicted_gain_s=gain, remaining_estimate=remaining)
    if preserved and not pair:
        s.record_tp_candidate(leader=leader,peer=peer,rid=row.rid,lookahead=lookahead,
                              donor_role=donor.role,**estimates)
    if not pair:
        if gain > 0:
            return Action("tp_expand", leader, s.version, dict(items=[{"rid":row.rid}], **estimates), peer)
        return
    # Shrinking a live layout must pay for the inverse KV repartition too.
    suffix = max(0, context-1-s.tp_prefix_tokens)
    live_exit = model.transition(leader, peer, donor.role, suffix, False)["transition_s"]
    if gain < -live_exit:
        return Action("tp_shrink", leader, s.version, dict(reason="service_pressure", **estimates), peer)
    return Action("decode", leader, s.version,
                  dict(items=[{"rid":row.rid}], steps=s.cfg["quantum"]), peer)


def wait_for_tail_boundary(s, active):
    """Keep a profitable idle donor available until the in-flight narrow chunk returns."""
    rows = [r for group in active.values() for r in group]
    if len(rows)!=1 or s.tp_pair or s.holds or not rows[0].tokens:
        return None
    row, = rows
    owner = s.groups[row.owner]
    if not owner.busy or owner.busy.kind != "decode":
        return None
    steps = min(owner.busy.payload["steps"],row.limit-len(row.tokens))
    until_safe = max(0.,s.tp_model.predict(len(row.prompt)+len(row.tokens),steps,s.tp_model.base)
                     - (s.now()-owner.busy_since))
    useful_work = any(r.ready or (not r.discarded and not r.scoring and len(r.labels)<len(r.tokens))
                      for r in s.rows.values())
    for peer in s.tp_model.partners.get(owner.gpu,()):
        if peer in s.queued_work:
            continue
        candidate = pair_action(s,rows,owner.gpu,peer,lookahead=True)
        if candidate and candidate.payload["predicted_gain_s"] > (until_safe if useful_work else 0.):
            return peer
    return None
