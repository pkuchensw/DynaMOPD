"""Single writer for trajectory progress, GPU leases, action receipts, and version commits."""
from collections import defaultdict
from bisect import bisect_right, insort
import json
import math
from pathlib import Path

from .protocol import Action, Costs, Group, validate_batch_receipt
from .base import PlanningState, SchedulingPolicy
from ..backends.base import Executor


class Trace:
    def __init__(self, path):
        self.file = Path(path).open("x", buffering=1)
        self.seq = 0

    def emit(self, event, **fields):
        self.seq += 1
        self.file.write(json.dumps(dict(seq=self.seq, event=event, **fields), allow_nan=False)+"\n")

    def close(self):
        self.file.close()


class Controller:
    def __init__(self, cfg, backend: Executor, policy: SchedulingPolicy, trace):
        self.cfg, self.backend, self.policy, self.trace = cfg, backend, policy, trace
        self.groups = [Group(i) for i in range(cfg["groups"])]
        self.cost = Costs(cfg)
        self.history = defaultdict(list)
        self.capped_lengths = defaultdict(list)
        self.prior_lengths = {}
        self.max_length_ema = None
        self.live_count = 0
        self.next_work = {}
        self.version = -1
        self.image_probe = None
        self.required_trainer = None
        self.training_groups = set()
        self.tp_pair = None
        self.tp_prefix_tokens = 0
        self.round_loss_sum = 0.
        self.decision_id = 0

    def target_role(self, g):
        return g.busy.payload["role"] if g.busy and g.busy.kind == "switch" else g.role

    def release_finished_teachers(self):
        # A route can have future work even with an empty prefix queue.
        # Retire only when every trajectory on it has terminated and settled.
        needed = {r.teacher for r in self.rows.values()
                  if not r.finish or (not r.discarded and len(r.labels) < len(r.tokens))}
        reserved = set(self.holds) | set(self.holds.values()) | set(self.tp_pair or ()) | self.next_work.keys()
        for g in self.groups:
            if not g.busy and g.gpu not in reserved and g.role in g.teacher_activation_s and g.role not in needed:
                self.submit(Action("switch", g.gpu, self.version, {"role": "standby"}))

    def remaining(self, row):
        current = len(row.tokens)
        if row.finish:
            return 0.
        # Only the last live request uses the batch maximum as its length forecast.
        # A request can outlive the forecast. Then use its conditional residual again.
        if self.live_count == 1 and self.max_length_ema is not None and current < self.max_length_ema:
            return min(row.limit-current, max(1., self.max_length_ema-current))
        seen = self.prior_lengths.get((row.teacher, row.limit)) or self.history[row.teacher]
        start = bisect_right(seen, current)
        middle = (start + len(seen)) // 2
        if start < len(seen):
            lower = (start + len(seen) - 1) // 2
            estimate = (seen[lower]+seen[middle])/2-current
        else:
            estimate = row.limit-current if self.cost.measuring else self.cfg["tail_prior"]
        return min(row.limit-current, max(1., estimate))

    def emit(self, event, **fields):
        self.trace.emit(event, t=self.backend.now(), version=self.version, **fields)

    def submit(self, a):
        for gpu in a.devices:
            if self.groups[gpu].busy:
                raise RuntimeError("overlapping exclusive leases")
        for gpu in a.devices:
            self.groups[gpu].busy = a
            self.groups[gpu].busy_since = self.backend.now()
        self.emit("dispatch", decision_id=self.decision_id,gpu=a.gpu, peer=a.peer, op=a.kind,
                  role=self.groups[a.gpu].role, next_role=a.payload.get("role"),
                  load_version=a.payload.get("load_version"),
                  ids=[x["rid"] for x in a.payload.get("items", [])])
        self.backend.submit(a)

    def materialize(self, action):
        """Resolve policy-selected IDs against the controller's versioned trajectory journal."""
        ids = [item["rid"] for item in action.payload["items"]]
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("a batch needs distinct trajectories")
        items = []
        for rid, selection in zip(ids, action.payload["items"], strict=True):
            row = self.rows[rid]
            if action.kind in ("decode", "move", "tp_expand"):
                if row.owner != action.gpu or row.finish:
                    raise ValueError(f"{rid} is not an active request on group {action.gpu}")
                item = row.item()
                if action.kind == "tp_expand":
                    item["last_logp"] = row.behavior[-1]
            elif action.kind == "score":
                start, end = len(row.labels), selection["end"]
                if (row.scoring or row.discarded or row.teacher != action.payload["teacher"]
                        or not start < end <= len(row.tokens)):
                    raise ValueError(f"invalid teacher interval for {rid}")
                item = row.item(end)
                item.update(start=start, final=row.finish is not None and end==len(row.tokens))
            else:
                if not row.ready:
                    raise ValueError(f"{rid} lacks completed supervision")
                item = dict(**row.item(), labels=row.labels.copy(), behavior=row.behavior.copy())
                if row.top_k:
                    item["teacher_topk"] = row.teacher_topk.copy()
            items.append(item)
        if action.kind == "move":
            active = {r.rid for r in self.rows.values() if r.owner == action.gpu and not r.finish}
            if set(ids) != active:
                raise ValueError("a released rollout group must hand off all its live requests")
        return Action(action.kind, action.gpu, action.version, dict(action.payload, items=items), action.peer)

    def assign(self, a):
        if a.version != self.version or any(not 0 <= i < len(self.groups) for i in a.devices):
            raise ValueError("invalid group or version in policy action")
        if a.kind == "decode" and (type(a.payload.get("steps")) is not int or a.payload["steps"] < 1):
            raise ValueError("decode steps must be a positive integer")
        if a.kind in ("decode", "hold", "move"):
            rollout_groups = (a.gpu,) if a.kind == "decode" else a.devices
            if any(self.groups[i].role != "rollout" or self.groups[i].version != self.version for i in rollout_groups):
                raise RuntimeError("action requires the current rollout image")
        if self.cfg.get("phase_barriers") and a.kind in ("score", "train"):
            if any(not row.finish for row in self.rows.values()):
                raise RuntimeError("rollout phase has unfinished trajectories")
            if a.kind == "train" and any(not r.discarded and len(r.labels) != len(r.tokens) for r in self.rows.values()):
                raise RuntimeError("teacher phase has undelivered supervision")
        tp_action = a.kind in ("tp_expand", "tp_shrink") or (a.kind == "decode" and a.peer is not None)
        if tp_action:
            model = getattr(self.backend, "tail_tp", None)
            if model is None or a.peer not in model.partners.get(a.gpu, ()):
                raise ValueError("TP handoff requires a supported same-node pair")
            if any(gpu in self.holds or gpu in self.holds.values() for gpu in a.devices):
                raise RuntimeError("TP handoff conflicts with a pending request migration")
            if a.kind == "tp_expand":
                live = [row for row in self.rows.values() if row.owner in a.devices and not row.finish]
                allow_trainer = getattr(model,"borrow_trainer",False)
                resizing = any(g.busy and g.busy.kind in ("tp_expand", "tp_shrink") for g in self.groups)
                if (self.tp_pair or resizing or len(live) != 1 or live[0].owner != a.gpu or not live[0].tokens
                        or [x["rid"] for x in a.payload["items"]] != [live[0].rid]
                        or self.groups[a.gpu].role != "rollout" or (not allow_trainer and
                        (a.peer in self.training_groups or a.peer in (self.trainer,self.required_trainer)))
                        or (self.groups[a.peer].role not in {"rollout","standby",*self.cfg["teachers"]}
                            and not (allow_trainer and self.groups[a.peer].role == "train"))
                        or any(self.groups[i].version != self.version for i in a.devices)):
                    raise RuntimeError("TP expansion requires a single owned rollout on the pair and a free donor")
            elif self.tp_pair != a.devices:
                raise RuntimeError("operation does not own the wide TP layout")
            self.submit(a if a.kind == "tp_shrink" else self.materialize(a))
            return
        if self.tp_pair and set(a.devices).intersection(self.tp_pair):
            raise RuntimeError("wide TP layout must be released before reassigning either group")
        if a.kind == "prepare_teacher":
            g, role = self.groups[a.gpu], a.payload["teacher"]
            if (role not in self.cfg["teachers"] or a.peer is not None or g.busy or g.role == role
                    or a.gpu in self.training_groups or a.gpu in self.next_work
                    or a.gpu in self.holds or a.gpu in self.holds.values()
                    or any(r.owner == a.gpu and not r.finish for r in self.rows.values())
                    or (self.cfg.get("phase_barriers") and any(not r.finish for r in self.rows.values()))):
                raise RuntimeError("teacher preparation requires a free drained group")
            self.submit(Action("switch", a.gpu, a.version, {"role": role}))
            return
        if a.kind == "standby":
            g = self.groups[a.gpu]
            if (a.peer is not None or g.busy or g.role not in g.teacher_activation_s or
                    a.gpu in self.holds or a.gpu in self.holds.values() or a.gpu in self.next_work):
                raise RuntimeError("standby requires an available leased teacher")
            self.submit(Action("switch", a.gpu, a.version, {"role": "standby"}))
            return
        if a.kind not in ("decode", "score", "train", "hold", "move", "unhold"):
            raise ValueError(f"{a.kind} belongs to the controller")
        if a.kind in ("hold", "move", "unhold"):
            if a.peer is None or a.gpu == a.peer:
                raise ValueError("migration requires two distinct groups")
        elif a.peer is not None:
            raise ValueError("only migration can address a peer group")
        if a.kind == "hold":
            if any(self.groups[i].busy and self.groups[i].busy.kind != "decode" for i in a.devices):
                raise RuntimeError("cannot reserve groups during another handoff")
            reserved = set(self.holds) | set(self.holds.values())
            if reserved.intersection(a.devices):
                raise RuntimeError("conflicting migration reservation")
            self.holds[a.gpu] = a.peer
            self.emit("migration_planned", decision_id=self.decision_id,gpu=a.gpu, peer=a.peer)
            return
        if a.kind in ("move", "unhold"):
            if self.holds.get(a.gpu) != a.peer:
                raise RuntimeError("migration does not own both reservations")
            if a.kind == "unhold":
                del self.holds[a.gpu]
                return
        if a.kind != "move" and set(a.devices).intersection(set(self.holds) | set(self.holds.values())):
            raise RuntimeError("group reserved for a migration")
        if any(self.groups[i].busy for i in a.devices):
            raise RuntimeError("policy selected a busy group")
        if a.kind in ("score", "train") and any(r.owner==a.gpu and not r.finish for r in self.rows.values()):
            raise RuntimeError("role switch requires a drained rollout group")
        a = self.materialize(a)
        if a.kind == "move":
            del self.holds[a.gpu]
        if a.kind in ("score", "train"):
            role = a.payload.get("teacher", "train")
            if a.kind == "score" and a.gpu in self.training_groups:
                raise RuntimeError("cannot release uncommitted gradients for scoring")
            if a.kind == "score" and self.groups[a.gpu].role != role and not self.cost.measuring:
                # Acquire only the engine lease. A resident replica can still serve
                # these prefixes while this group loads; replan on its ready receipt.
                self.submit(Action("switch", a.gpu, a.version, {"role": role}))
                return
            if a.kind == "train" and a.gpu not in self.training_groups:
                claimed = self.training_groups | {a.gpu}
                if self.trainer is not None:
                    claimed.add(self.trainer)
                if len(claimed) > self.cfg.get("max_train_groups", 1):
                    raise RuntimeError("training replica limit exceeded")
                if len(self.training_groups) + 1 == len(self.groups) and any(
                    not r.finish or (not r.discarded and len(r.labels) < len(r.tokens)) for r in self.rows.values()
                ):
                    raise RuntimeError("unfinished supervision still requires a service group")
                if self.trainer is None:
                    self.trainer = a.gpu
                self.training_groups.add(a.gpu)
            for item in a.payload["items"]:
                row = self.rows[item["rid"]]
                if a.kind == "score":
                    if row.scoring or len(row.labels) != item["start"] or row.teacher != role:
                        raise RuntimeError("duplicate or misrouted score claim")
                    row.scoring = True
                else:
                    if not row.ready:
                        raise RuntimeError("ineligible training claim")
                    row.training = True
            if self.groups[a.gpu].role != role:
                self.next_work[a.gpu] = a
                switch = {"role": role}
                if role == "train":
                    switch["optimizer_owner"] = a.gpu == self.trainer
                self.submit(Action("switch", a.gpu, a.version, switch))
                return
        self.submit(a)

    def settle(self, a, receipt):
        if a.version != self.version or any(self.groups[i].busy is not a for i in a.devices):
            raise RuntimeError("stale worker receipt")
        value = receipt["value"]
        duration = receipt["end"]-receipt["start"]
        if not math.isfinite(duration) or duration < 0:
            raise RuntimeError("invalid worker action duration")
        if a.kind in ("score", "decode"):
            validate_batch_receipt(a, value)
        if a.kind == "switch" and value["new"] != a.payload["role"]:
            raise RuntimeError("worker acknowledged a different role")
        observe = getattr(getattr(self.backend,"tail_tp",None),"observe",None)
        if observe:
            observe(a,value,duration,self.cfg["quantum"])
        g = self.groups[a.gpu]
        for i in a.devices:
            self.groups[i].busy = None
        g.free_bytes = receipt["free_bytes"]
        if a.peer is not None and "peer_free_bytes" in value:
            self.groups[a.peer].free_bytes = value["peer_free_bytes"]
        summary = {k:v for k,v in value.items() if k not in ("updates", "scores")}
        if a.kind == "decode":
            delayed = {x["rid"]: len(x["prompt"])+len(x["tokens"]) for x in a.payload["items"]
                       if self.rows[x["rid"]].deferred_prefill}
            if delayed:
                summary["migration_prefill_contexts"] = delayed
        self.emit("complete", op=a.kind, gpu=a.gpu, peer=a.peer, start=receipt["start"],
                  end=receipt["end"], role=g.role, free_bytes=g.free_bytes, **summary)
        for rid in value.get("evicted", []):
            if rid in self.rows:
                self.rows[rid].kv.pop(g.gpu, None)
        items = a.payload.get("items", [])
        if a.kind in ("tp_expand", "tp_shrink"):
            self.tp_pair = a.devices if a.kind == "tp_expand" and value.get("accepted",True) else None
            self.tp_prefix_tokens = value["kv_context_tokens"] if self.tp_pair else 0
            for rid in value.get("peer_evicted", []):
                self.rows[rid].kv.pop(a.peer, None)
        elif a.kind == "switch":
            if (self.cost.measuring and a.payload.get("publish_from") is None and
                    value["old"] != value["new"]):
                self.cost.observe(("switch_group", g.gpu, value["old"], value["new"]), duration)
                self.cost.observe(("switch_to", value["new"]), duration)
            if value.get("teacher_prepare_s") is not None:
                self.cost.observe("teacher_prepare", value["teacher_prepare_s"])
            if "teacher_rank_release_s" in value:
                prior = self.cfg.get("sim_worker_profile", {}).get("teacher_prepare_s",
                                                               self.cfg.get("sim_teacher_prepare_s", 0.))
                g.teacher_ready_at = (receipt["start"] + value["teacher_rank_release_s"]
                                      + self.cost.values.get("teacher_prepare", prior))
                if not value.get("teacher_prepare_enabled", self.cfg.get("preload_teacher_imports", True)):
                    g.teacher_ready_at = 0.
            first_load = (g.prepared_teachers is not None and value["new"] in self.cfg["teachers"]
                          and value["new"] not in g.prepared_teachers)
            if "prepared_teachers" in value:
                g.prepared_teachers = frozenset(value["prepared_teachers"])
                g.cached_teacher_weights = frozenset(value["cached_teacher_weights"])
                g.teacher_start_s = value["teacher_start_s"]
                g.teacher_activation_s = value.get("teacher_activation_s", {})
            if "teacher_rank_load_s" in value:
                self.cost.observe(("teacher_gpu_activation", value["new"]),
                                  value["teacher_rank_load_s"]-value.get("standby_wait_s", 0.))
                self.cost.observe(("before_teacher", value["old"], value["new"], g.trainer_resident,
                                   g.gpu == self.trainer), value["unload_s"])
            if value.get("engine_restarted"):
                g.wake_calibrated = False
            if "wake_calibrated" in value:
                g.wake_calibrated = value["wake_calibrated"]
            if value["new"] in {"rollout", *self.cfg["teachers"]}:
                g.last_inference_role = value["new"]
            if value["new"] == "train" or "trainer_state_device" in value:
                g.trainer_resident = value.get("trainer_state_device") == "cuda"
            g.role, g.version = value["new"], a.payload.get("load_version", self.version)
            if (a.payload.get("publish_from") is not None and self.cfg.get("max_train_groups", 1) > 1
                    and value.get("master_version") != g.version):
                raise RuntimeError("FSDP master did not acknowledge the new policy version")
            g.resident_since = self.backend.now()
            if g.role in ("train", "rollout") and value.get("image_probe"):
                if self.image_probe is None:
                    self.image_probe = value["image_probe"]
                elif value["image_probe"] != self.image_probe:
                    raise RuntimeError("worker loaded a different student weight image")
            if value.get("teacher_weights_reused"):
                self.cost.observe("teacher_wake", duration)
            elif not first_load and value["old"] not in ("cold",value["new"]):
                if not {value["old"], value["new"]}.intersection(g.teacher_activation_s):
                    self.cost.observe("switch", duration)
                self.cost.observe((value["old"], value["new"]), duration)
        elif a.kind == "decode":
            for rid, update in value["updates"].items():
                row = self.rows[rid]
                if row.owner != g.gpu or row.version != a.version:
                    raise RuntimeError("stale decode owner")
                row.deferred_prefill = False
                row.prefix(**update)
                if row.pending_since is None:
                    row.pending_since = self.backend.now()
                if row.finish:
                    self.live_count -= 1
                if row.finish == "eos":
                    insort(self.history[row.teacher], len(row.tokens))
                elif row.finish == "length":
                    insort(self.capped_lengths[row.teacher], len(row.tokens))
                self.emit("prefix", rid=rid, gpu=g.gpu, end=len(row.tokens), start=update["start"],
                          finish=row.finish, teacher=row.teacher)
            if a.peer is None:
                self.cost.observe("decode_step", duration/max(len(x["tokens"]) for x in value["updates"].values()))
            if not any(r.owner==g.gpu and not r.finish for r in self.rows.values()):
                self.emit("group_drained", gpu=g.gpu, cause="natural")
        elif a.kind == "score":
            group_size = self.cfg.get("group_size", 1)
            lanes = group_size // self.cfg.get("inference_tp_size", group_size)
            loads = [0] * lanes
            for rid, score in value["scores"].items():
                row = self.rows[rid]
                loads[row.seed // len(self.groups) % lanes] += max(1, len(row.prompt)+score["end"]-1-score["reused"])
                keep = score["retained"] and rid not in value.get("evicted", [])
                row.deliver(score["start"], score["end"], score["labels"], g.gpu, keep, score.get("topk"))
                if len(row.labels)==len(row.tokens):
                    row.pending_since = None
                self.emit("supervision", gpu=g.gpu, teacher=row.teacher, rid=rid, start=score["start"],
                          end=score["end"], before_eos=row.finish is None, reused_kv_tokens=score["reused"],
                          cached_on=row.kv, ready=row.ready)
            self.cost.observe("score_token", duration/sum(len(x["tokens"])-x["start"] for x in items))
            if not self.cost.measuring:
                self.cost.observe("score_base", min(duration, self.cost.get("score_base")*2))
            self.cost.observe_score(a.payload["teacher"], max(loads), duration)
        elif a.kind == "move":
            if value["accepted"]:
                install = value.get("install_end", receipt["end"]) - value.get("install_start", receipt["start"])
                if value.get("rebuilt_tokens",0):
                    self.cost.observe("prefill_token", install/value["rebuilt_tokens"])
                if "install_end" in value:
                    cross = value.get("source_node") != value.get("destination_node")
                    penalty = self.cfg["costs"].get("cross_node_move_s", 0.) if cross else 0.
                    self.cost.observe("move_base", max(0.,duration-install-penalty))
                for x in items:
                    self.rows[x["rid"]].owner = a.peer
                    self.rows[x["rid"]].deferred_prefill = a.payload.get("rebuild") == "on_decode"
                self.emit("group_drained", gpu=g.gpu, cause="migration", new_owner=a.peer,
                          ids=[x["rid"] for x in items])
            else:
                self.rejected_moves[(a.gpu,a.peer)] = tuple(
                    frozenset(r.rid for r in self.rows.values() if r.owner==i and not r.finish)
                    for i in (a.gpu,a.peer))
        elif a.kind == "train":
            expected = sum(len(x["tokens"]) for x in items)
            if expected != value["tokens"] or not math.isfinite(value["loss_sum"]):
                raise RuntimeError("invalid training receipt")
            self.round_loss_sum += value["loss_sum"]
            self.group_tokens[g.gpu] += expected
            for x in items:
                r = self.rows[x["rid"]]
                r.training, r.trained = False, True
            self.cost.observe("train_token", duration/sum(len(x["prompt"])+len(x["tokens"]) for x in items))
        elif a.kind == "reduce_grads":
            if (value["owner"] != self.trainer or value["group_tokens"] != self.group_tokens[g.gpu]
                    or value["global_tokens"] != sum(self.group_tokens)):
                raise RuntimeError("gradient reduction disagrees with settled training receipts")
            if not self.cost.measuring:
                self.cost.observe("replica_reduce", duration)
        elif a.kind == "commit":
            if value["published"] != self.version+1 or value["tokens"] != sum(self.group_tokens):
                raise RuntimeError("incorrect published version")
            self.commit_result = value
            self.cost.observe("optimizer_step", value.get("optimizer_and_probe_s", duration))
            if value.get("durable",True)!=a.payload["durable"]:
                raise RuntimeError("checkpoint durability receipt disagrees with commit")
            # Only this group holds current Adam moments while the job is live.
            self.required_trainer = g.gpu if self.cfg.get("max_train_groups", 1) > 1 or not a.payload["durable"] else None
            self.image_probe = value.get("image_probe")

    def receive(self):
        # Settle all currently available events before giving the policy another view.
        for a, receipt in self.backend.wait():
            self.settle(a, receipt)

    def run_round(self, rows, version):
        self.version = version
        self.rows = {r.rid:r for r in rows}
        # Early completions within the current round are right-censored towards
        # short responses. Freeze completed-round distributions at publication.
        self.prior_lengths = {role: tuple(lengths) for role, lengths in self.history.items()}
        for role, limit in {(row.teacher, row.limit) for row in rows}:
            # Reaching an old cap >= this limit tells us exactly how much work
            # that trajectory would have taken at this cap. Shorter censored
            # responses give no exact length for the new limit.
            capped = self.capped_lengths[role]
            count = len(capped) - bisect_right(capped, limit-1)
            if count:
                seen = self.prior_lengths.get(role, ())
                self.prior_lengths[role, limit] = tuple(min(n,limit) for n in seen) + (limit,)*count
        place = getattr(self.policy, "place_rollouts", None)
        if place:
            place(rows, self.cfg, self.prior_lengths)
        self.live_count = len(rows)
        if (len(self.rows) != len(rows) or set(r.owner for r in rows) != set(range(len(self.groups)))
                or any(r.version != version for r in rows)):
            raise ValueError("round must initialize unique trajectories on every group")
        self.trainer, self.holds, self.rejected_moves = self.required_trainer, {}, {}
        self.training_groups = set()
        self.group_tokens = [0] * len(self.groups)
        self.round_loss_sum = 0.
        self.commit_result = None
        self.decision_id += 1
        self.emit("round_start", samples=len(rows), max_length_ema=self.max_length_ema,
                  optimizer_owner=self.required_trainer,requests=[
            dict(rid=r.rid, owner=r.owner, teacher=r.teacher, seed=r.seed,
                 prompt_tokens=len(r.prompt), max_new_tokens=r.limit) for r in rows])
        for g in self.groups:
            if g.role!="rollout" or g.version!=version:
                self.submit(Action("switch", g.gpu, version, {"role":"rollout"}))
        while any(g.busy for g in self.groups):
            self.receive()
        while True:
            previous_holds = self.holds.copy()
            self.decision_id += 1
            for gpu, a in list(self.next_work.items()):
                if not self.groups[gpu].busy:
                    if self.groups[gpu].role != a.payload.get("teacher", "train"):
                        raise RuntimeError("missing role handoff")
                    del self.next_work[gpu]
                    self.submit(a)
            for a in self.policy.plan(PlanningState(self)):
                self.assign(a)
            if not getattr(self.policy, "plans_standby", False):
                self.release_finished_teachers()
            if any(g.busy for g in self.groups):
                self.receive()
                continue
            if self.tp_pair:
                raise RuntimeError("scheduler left a wide TP layout unreleased")
            if self.holds:
                # A new reservation can require one more decision. No receipt
                # can wake a policy that keeps the same reservation forever.
                if self.holds == previous_holds:
                    raise RuntimeError("scheduler stalled with unresolved migration reservations")
                continue
            if not all(r.trained or r.discarded for r in rows):
                raise RuntimeError("scheduler stalled with unfinished work")
            break
        effective = sum(len(r.tokens) for r in rows if r.trained)
        if not effective:
            raise RuntimeError("all requests reached the length limit; no EOS samples to update")
        if effective != sum(self.group_tokens) or self.training_groups != {i for i,n in enumerate(self.group_tokens) if n}:
            raise RuntimeError("trained trajectories and replica token counts disagree")
        if self.trainer not in self.training_groups:
            # The optimizer's group can finish a long rollout after helpers have
            # trained the entire batch. Only its optimizer and master are needed.
            self.decision_id += 1
            self.submit(Action("switch", self.trainer, version, {"role": "train", "optimizer_owner": True}))
            self.receive()
        if self.training_groups != {self.trainer}:
            # Every same-rank NCCL lane has the same fixed membership, including
            # groups which finished the round running inference and contribute zero.
            self.decision_id += 1
            reduction_start = self.backend.now()
            for g in self.groups:
                self.submit(Action("reduce_grads", g.gpu, version,
                                   {"owner": self.trainer, "counts": self.group_tokens.copy()}))
            while any(g.busy for g in self.groups):
                self.receive()
            if self.cost.measuring:
                self.cost.observe("replica_reduce", self.backend.now()-reduction_start)
        next_version = version+1
        interval = self.cfg.get("checkpoint_interval", 1)
        save_hf = next_version == self.cfg.get("final_version") or (interval > 0 and next_version % interval == 0)
        durable = save_hf and "training" in self.cfg.get("checkpoint_contents", ["hf_model", "training"])
        round_max = max(len(r.tokens) for r in rows)
        alpha = self.cfg["max_length_ema_alpha"]
        next_ema = None
        if alpha is not None:
            next_ema = (round_max if self.max_length_ema is None else
                        self.max_length_ema + alpha * (round_max - self.max_length_ema))
        commit = {"tokens": effective, "durable": durable, "save_hf": save_hf}
        if durable and next_ema is not None:
            commit["scheduler_state"] = {"max_length_ema": next_ema}
        self.decision_id += 1
        self.submit(Action("commit", self.trainer, version, commit))
        self.receive()
        self.decision_id += 1
        publish_start = self.backend.now()
        for g in self.groups:
            self.submit(Action("switch", g.gpu, version, {"role":"rollout","load_version":version+1,
                                                         "publish_from": self.trainer}))
        while any(g.busy for g in self.groups):
            self.receive()
        if self.cost.measuring:
            self.cost.observe("rollout_publication", self.backend.now()-publish_start)
        self.emit("rollout_barrier", loaded_version=version+1, groups=len(self.groups))
        self.max_length_ema = next_ema
        self.emit("round_end", eos=sum(r.finish=="eos" for r in rows), trained=sum(r.trained for r in rows), truncated=sum(r.finish=="length" for r in rows),
                  max_response_tokens=round_max, max_length_ema=self.max_length_ema, **self.commit_result)
        return self.commit_result
