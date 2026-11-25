"""Virtual-time executor using SimRL rollout/scoring costs and its KV block manager."""
import heapq
import math
from .score_cache import trim_score_cache
from .sim_profile import SwitchState, decode_roofline, engine_classes, fitted_switch, train_seconds
from itertools import count

from srl.component.rollout.engine.kv.manager import KvCacheManager
from srl.component.rollout.engine.request import Request
from srl.cost.rollout.roofline import UnifiedRoofline
from srl.cost.scoring import ScoringCost
from ..cluster.topology import GroupPlacement, requested_nodes, warm_teacher_routes


class Simulator:
    supervision = frozenset({"sampled_token_logprob", "teacher_topk"})
    supports_periodic_checkpoint = True
    supports_resume = False

    @staticmethod
    def validate_checkpoint(path):
        raise ValueError("SimRL does not restore model checkpoints")

    def __init__(self, cfg):
        self.cfg = cfg
        nodes = requested_nodes(cfg) or ("sim",)*cfg["groups"]
        width = cfg.get("group_size", 1)
        self.placements = tuple(GroupPlacement(i, node, tuple(f"sim-{i}-{j}" for j in range(width)))
                                for i, node in enumerate(nodes))
        self.clock = 0.
        self.queue = []
        self.serial = count()
        c = cfg["costs"]
        self.rollout = UnifiedRoofline(**decode_roofline(cfg))
        self.teacher = ScoringCost(launch_s=c["score_base"], token_s=c["score_token"])
        blocks = cfg.get("sim_blocks", cfg.get("engine_kv_tokens",131072)//16)
        self.tp_size = cfg.get("inference_tp_size", width)
        self.replicas = width // self.tp_size
        self.managers = [[KvCacheManager(blocks, enable_caching=False) for _ in range(self.replicas)]
                         for _ in range(cfg["groups"])]
        self.lanes = [{} for _ in self.managers]
        self.leased_teachers = (cfg.get("engine") == "sglang_fsdp2" and
                                cfg.get("sim_teacher_lifecycle", "lease") == "lease")
        self.profile = cfg.get("sim_worker_profile", {})
        self.tail_tp = None
        self.tp_pair = None
        self.tp_prefix_tokens = 0
        self.wide_idle_bytes = {}
        self.wide_managers = []
        self.wide_requests = []
        if cfg.get("rollout_tp_expansion"):
            from .elastic_profile import TailProfile
            self.tail_tp = TailProfile(cfg,self.placements)
            options = cfg["rollout_tp_expansion"]
            self.tail_tp.scratch = options.get("wake_bytes_per_gpu",0)
            self.wide_idle_bytes = {group:options.get("idle_bytes_per_gpu",0)
                                    for pair in self.tail_tp.pairs for group in pair}
            self.wide_managers = [KvCacheManager(blocks,enable_caching=False) for _ in range(self.tail_tp.wide)]
        elif cfg.get("sim_tp"):
            from .sim_tp import TailTP
            if self.replicas != 1:
                raise ValueError("wide-TP simulation requires one TP engine per base group")
            self.tail_tp = TailTP(cfg, self.placements, blocks*self.managers[0][0].block_size)
            self.rollout = self.tail_tp.models[width]
        self.reqs = [{} for _ in self.managers]
        self.caches = [{} for _ in self.managers]
        self.roles = ["cold"]*cfg["groups"]
        self.teacher_ready_at = [0.] * cfg["groups"]
        self.teacher_prepare_s = self.profile.get("teacher_prepare_s", cfg.get("sim_teacher_prepare_s", 0.))
        self.prepared = [set(warm_teacher_routes(cfg, i)) for i in range(cfg["groups"])]
        aliases = {name for name, path in cfg["teachers"].items() if path == cfg.get("student")
                   and cfg.get("share_student_teacher_engine", True)}
        self.signatures = engine_classes(cfg)
        self.switch_states = [SwitchState(cfg, self.signatures) for _ in self.roles]
        self.optimizer_owner = None
        self.publish_pending = []
        if cfg.get("inference_layout") == "single":
            self.engine_class = [self.signatures["rollout"]] * len(self.roles)
            aliases = {key for key in cfg["teachers"] if self.signatures[key] == self.signatures["rollout"]}
            self.prepared = [set(aliases) for _ in self.roles]
        self.dedicated = set(cfg["teachers"]) - aliases
        for names in self.prepared:
            names.update(aliases)
        self.parked = [set(names & self.dedicated) if cfg.get("prime_teacher_weights") else set()
                       for names in self.prepared]
        historical = self.profile.get("lifecycle") == "restart" and self.leased_teachers
        activation_prior = c["switch"] if historical else cfg.get("sim_teacher_start_s", c["switch"])
        self.activation_s = {name: cfg.get("sim_teacher_activation_s", {}).get(
            name, activation_prior) for name in self.dedicated}
        if self.leased_teachers:
            self.prepared = [set(cfg["teachers"]) for _ in self.roles]
            self.parked = [set() for _ in self.roles]
        self.targets = {}
        self.train_tokens = [0] * cfg["groups"]
        self.reduced_groups = set()

    def now(self):
        return self.clock

    def occupied(self, role):
        if role == "standby":
            return self.cfg.get("sim_role_bytes", {}).get("standby", 0)
        return self.cfg.get("sim_role_bytes", {}).get(role, (64 if role=="train" else 16)*2**30)

    def parked_bytes(self, gpu):
        sizes = self.cfg.get("sim_teacher_weight_bytes", {})
        return sum(sizes.get(role, self.occupied(role)) for role in self.parked[gpu])

    def free_bytes(self, group):
        widening = self.tp_pair and group in self.tp_pair
        footprint = self.occupied("rollout" if widening and not self.wide_managers else self.roles[group]) + self.parked_bytes(group)
        footprint += self.wide_idle_bytes.get(group,0)
        if widening:
            footprint += self.tail_tp.scratch
        return max(0, self.cfg.get("sim_gpu_bytes",96*2**30) - footprint)

    def grow(self, gpu, x, size, version):
        rid = x["rid"]
        if rid not in self.reqs[gpu]:
            self.reqs[gpu][rid] = Request(next(self.serial), len(x["prompt"]), self.targets[rid], version)
            self.lanes[gpu][rid] = self.lane(x)
        req = self.reqs[gpu][rid]
        if self.managers[gpu][self.lanes[gpu][rid]].allocate(req, size-req.prefilled, []) is None:
            raise MemoryError("simulated rollout KV exhausted")
        req.prefilled = size

    def free(self, gpu, rid):
        self.managers[gpu][self.lanes[gpu].pop(rid)].free(self.reqs[gpu].pop(rid))

    def lane(self, item):
        return item["seed"] // self.cfg["groups"] % self.replicas

    def decode(self, a):
        lanes = [[] for _ in range(self.replicas)]
        for item in a.payload["items"]:
            lanes[self.lane(item)].append(item)
        updates, durations = {}, []
        for items in lanes:
            if items:
                dt, changed = self.decode_lane(a, items)
                durations.append(dt)
                updates.update(changed)
        return max(durations), dict(updates=updates, tp_width=self.tail_tp.wide if a.peer is not None else self.tp_size,
                                    replica_requests=[len(items) for items in lanes])

    def decode_lane(self, a, active):
        ids, dt = {}, 0.
        model = self.rollout
        if a.peer is not None:
            if self.tp_pair != a.devices or len(active) != 1:
                raise RuntimeError("wide decode requires a single request and both TP groups")
            if not self.wide_managers:
                model = self.tail_tp.models[self.tail_tp.wide]
        for x in active:
            ids[x["rid"]] = dict(start=len(x["tokens"]), tokens=[], logp=[], finish=None)
        for _ in range(a.payload["steps"]):
            if not active:
                break
            prefill, decode = [], []
            for x in active:
                rid = x["rid"]
                u = ids[rid]
                n = len(x["prompt"])+u["start"]+len(u["tokens"])
                narrow = self.reqs[a.gpu].get(rid)
                (decode if narrow is not None else prefill).append(n)
                if self.wide_managers and a.peer is None and narrow is not None:
                    # The narrow engine kept its original KV while the wide
                    # engine generated. Replay its missing suffix on first use.
                    dt += max(0, n - 1 - narrow.prefilled) * self.tail_tp.prefill
                if a.peer is not None and self.wide_managers:
                    self.grow_wide(n)
                else:
                    self.grow(a.gpu,x,n,a.version)
                    if a.peer is not None:
                        self.grow(a.peer,x,n,a.version)
                u["tokens"].append(1)
                u["logp"].append(-1.)
                end = u["start"]+len(u["tokens"])
                if end == self.targets[rid]:
                    u["finish"] = "eos"
                elif end == x["limit"]:
                    u["finish"] = "length"
                if u["finish"]:
                    self.free(a.gpu, rid)
                    if a.peer is not None and not self.wide_managers:
                        self.free(a.peer, rid)
            if self.wide_managers and len(active)==1 and not prefill:
                width = self.tail_tp.wide if a.peer is not None else self.tp_size
                dt += self.tail_tp.predict(decode[0],1,width)
            else:
                dt += model.step_time(sum(prefill)+len(decode),sum(n*n for n in prefill),sum(decode),len(active))
            active = [x for x in active if not ids[x["rid"]]["finish"]]
        return dt, ids

    def grow_wide(self, size):
        for manager,req in zip(self.wide_managers,self.wide_requests,strict=True):
            if size > req.prefilled and manager.allocate(req,size-req.prefilled,[]) is None:
                raise MemoryError("wide TP rank has insufficient KV pages")
            req.prefilled = size

    def resize_tp(self, a):
        leader, peer = a.devices
        entering = a.kind == "tp_expand"
        donor_role = self.roles[peer]
        model = self.tail_tp
        if not model or peer not in model.partners.get(leader, ()):
            raise RuntimeError("unsupported TP slot")
        if entering:
            item, = a.payload["items"]
            rid = item["rid"]
            if self.tp_pair or self.reqs[peer] or set(self.reqs[leader]) != {rid}:
                raise RuntimeError("cannot repartition occupied TP slots")
            context = len(item["prompt"]) + len(item["tokens"]) - 1
            narrow_context = self.reqs[leader][rid].prefilled
            if narrow_context > context or (not self.wide_managers and narrow_context != context):
                raise RuntimeError("tail KV does not match the published prefix")
            if self.wide_managers:
                self.wide_requests = [Request(next(self.serial),len(item["prompt"]),self.targets[rid],a.version)
                                      for _ in self.wide_managers]
                self.grow_wide(context)
            else:
                self.grow(peer,item,context,a.version)
            self.tp_prefix_tokens = context
            evicted = [] if self.wide_managers else list(self.caches[peer])
            if not self.wide_managers:
                self.caches[peer].clear()
            self.tp_pair = a.devices
        else:
            if self.tp_pair != a.devices or (not self.wide_managers and self.reqs[peer].keys() != self.reqs[leader].keys()):
                raise RuntimeError("TP release does not match the active KV layout")
            if self.wide_managers:
                context = self.wide_requests[0].prefilled if self.reqs[leader] else 0
                for manager,req in zip(self.wide_managers,self.wide_requests,strict=True):
                    manager.free(req)
                self.wide_requests.clear()
            else:
                context = sum(r.prefilled for r in self.reqs[peer].values())
            for rid in list(self.reqs[peer]):
                self.free(peer, rid)
            evicted = []
            self.tp_pair = None
        # The original narrow KV prefix remains allocated on the owner. Only
        # heads written on the peer during wide decode are missing on return.
        transfer_tokens = context if entering else max(0, context-self.tp_prefix_tokens)
        # A separate SGLang mesh sleeps on release. Rebuilding a narrow
        # suffix costs time only if this request subsequently decodes there.
        switch_tokens = 0 if self.wide_managers and not entering else transfer_tokens
        parts = model.transition(leader, peer, donor_role, switch_tokens, entering)
        if not self.wide_managers:
            model.communicators.add(tuple(sorted(a.devices)))
        estimates = {k:v for k,v in a.payload.items() if k != "items"}
        value = dict(**estimates, **parts, donor_role=donor_role, peer_evicted=evicted,
                     kv_context_tokens=context, kv_reshard_tokens=0 if self.wide_managers else transfer_tokens,
                     from_tp=model.base if entering else model.wide, to_tp=model.wide if entering else model.base)
        if not entering:
            self.tp_prefix_tokens = 0
        return parts["transition_s"], value

    def submit(self, a):
        cfg, gpu, c = self.cfg, a.gpu, self.cfg["costs"]
        p, items = a.payload, a.payload.get("items", [])
        cache = self.caches[gpu]
        if a.kind == "decode":
            dt, value = self.decode(a)
        elif a.kind in ("tp_expand", "tp_shrink"):
            dt, value = self.resize_tp(a)
        elif a.kind == "switch":
            old = self.roles[gpu]
            if self.reqs[gpu]:
                raise RuntimeError("role change with live rollout KV")
            self.roles[gpu] = p["role"]
            unload, load = cfg.get("sim_switch_s",{}).get(old,{}).get(
                p["role"], (c["switch"]*.2, c["switch"]*.8))
            target = p["role"]
            start_s = cfg.get("sim_teacher_start_s", c["switch"])
            cold = target in cfg["teachers"] and target not in self.prepared[gpu]
            if self.leased_teachers:
                if target == "standby":
                    unload, load = cfg.get("sim_switch_s", {}).get(old, {}).get(target, (unload, 0.))
                elif target in self.dedicated and old != target:
                    load = cfg.get("sim_switch_s", {}).get(old, {}).get(target, (unload, self.activation_s[target]))[1]
                hit = False
            else:
                if cfg.get("inference_layout") == "single" and target in self.signatures:
                    cold = self.engine_class[gpu] != self.signatures[target]
                    self.engine_class[gpu] = self.signatures[target]
                hit = target in self.parked[gpu]
                if cfg.get("cache_teacher_weights") and old in self.dedicated and target!=old:
                    self.parked[gpu].add(old)
                if target=="train" or cold:
                    self.parked[gpu].clear()
                if cfg.get("inference_layout") == "single" and target in self.signatures:
                    self.prepared[gpu] = {key for key in cfg["teachers"] if self.signatures[key] == self.signatures[target]}
                elif target in cfg["teachers"]:
                    self.prepared[gpu].add(target)
                self.parked[gpu].discard(target)
                capacity = cfg.get("sim_gpu_bytes",96*2**30) - cfg.get("memory_margin",0)
                for name in sorted(self.parked[gpu]):
                    if self.occupied(target) + self.parked_bytes(gpu) <= capacity:
                        break
                    self.parked[gpu].remove(name)
                if cold:
                    load = start_s
                elif hit:
                    load = cfg.get("sim_teacher_wake_s", c["score_base"])
            publishing = p.get("publish_from") is not None
            if target == "train" and p.get("optimizer_owner", True):
                self.optimizer_owner = gpu
            state = self.switch_states[gpu]
            keys = state.keys(old, target, publishing, gpu == self.optimizer_owner)
            release_key = state.release_key(old, publishing, gpu == self.optimizer_owner)
            profile_key, cell = fitted_switch(self.profile, keys, release_key)
            if self.profile and not cell:
                unload, load = c["switch"]*.2, c["switch"]*.8
                if self.leased_teachers and old in self.dedicated:
                    unload = cfg.get("sim_teacher_release_s", {}).get(old, unload)
                if self.leased_teachers and target in self.dedicated:
                    load = self.activation_s[target]
                if target == "standby":
                    load = 0.
            if cell:
                unload, load = cell["unload"], cell["load"]
            standby_wait = 0.
            if self.leased_teachers and self.teacher_prepare_s and old != target:
                if old in self.dedicated:
                    released = cfg.get("sim_teacher_release_s", {}).get(old, unload)
                    self.teacher_ready_at[gpu] = self.clock + released + self.teacher_prepare_s
                if target in self.dedicated:
                    standby_wait = max(0., self.teacher_ready_at[gpu] - self.clock - unload)
                    load += standby_wait
            state.advance(target, cell.get("device") if cell else None)
            dt = unload+load
            value = dict(old=old, new=target, evicted=list(cache), unload_s=unload, load_s=load,
                         prepared_teachers=sorted(self.prepared[gpu]), cached_teacher_weights=sorted(self.parked[gpu]),
                         teacher_start_s=start_s, teacher_weights_reused=hit)
            value["wake_calibrated"] = state.wake_calibrated
            if cell:
                value.update(cost_profile_key=profile_key, cost_profile_samples=cell["n"])
                if cell.get("device"):
                    value["trainer_state_device"] = cell["device"]
            elif self.profile:
                value["cost_profile_fallback"] = keys[0]
            if not self.leased_teachers:
                value["engine_restarted"] = "|restart|" in keys[0]
            if self.leased_teachers:
                value["teacher_activation_s"] = dict(self.activation_s)
                if old != target and target in self.dedicated and self.teacher_prepare_s:
                    value.update(standby_wait_s=standby_wait, teacher_rank_load_s=load)
                value["rollout_engine_restarted"] = False
            if p.get("publish_from") is not None and cfg.get("max_train_groups", 1) > 1:
                sync = cfg.get("sim_replica_sync_s", cfg.get("sim_publish_s", 1.))
                if cell or target in cfg.get("sim_switch_s", {}).get(old, {}):
                    # The native worker measures refresh_parameters inside unload_s.
                    sync = min(unload, cell["sync"] if cell else sync)
                else:
                    dt += sync
                    value["unload_s"] += sync
                value.update(master_version=p["load_version"], replica_sync_s=sync)
            cache.clear()
        elif a.kind == "score":
            lengths, scores = [[] for _ in range(self.replicas)], {}
            for x in items:
                rid, start, end = x["rid"], x["start"], len(x["tokens"])
                base = len(x["prompt"])+start-1
                lane = self.lane(x)
                cached_lane, cached_end = cache.pop(rid, (lane, 0))
                hit = base if cached_lane == lane and cached_end == base + 1 else 0
                size = len(x["prompt"])+end-1
                lengths[self.lane(x)].append(size-hit)
                if not x["final"]:
                    cache[rid] = (lane, size + 1)
                scores[rid] = dict(start=start, end=end, labels=[-1.2]*(end-start), reused=hit,
                                   retained=not x["final"])
                k = cfg.get("teacher_top_k", 0)
                if k:
                    position = [(j, -math.log(k)) for j in range(k)]
                    scores[rid]["topk"] = [position]*(end-start)
            lanes, evicted = trim_score_cache(cache, cfg["teacher_kv_tokens"])
            dt = max(self.teacher.predict(tuple(lane), 1) for lane in lengths if lane)
            dt += c.get("topk_value_s", 0.) * cfg.get("teacher_top_k", 0) * sum(len(x["labels"]) for x in scores.values())
            value = dict(scores=scores, evicted=evicted, flushed_score_lanes=lanes,
                         inference_tp_size=self.tp_size, replica_requests=[len(lane) for lane in lengths])
        elif a.kind == "move":
            counts = [len(x["prompt"])+len(x["tokens"]) for x in items]
            by_lane = [[] for _ in range(self.replicas)]
            for x,n in zip(items, counts, strict=True):
                by_lane[self.lane(x)].append(n)
            headroom = [sum(n + cfg["quantum"] for n in sizes) for sizes in by_lane]
            for rid, req in self.reqs[a.peer].items():
                # A decode request's last sampled token has not entered the KV yet.
                headroom[self.lanes[a.peer][rid]] += req.prefilled + 1 + cfg["quantum"]
            managers = self.managers[a.peer]
            limits = [min(cfg.get("engine_kv_tokens", 131072),
                          (m.num_free_blocks + m.num_used_blocks)*m.block_size) for m in managers]
            accepted = all(need <= cap and sum((n+m.block_size-1)//m.block_size for n in sizes) <= m.num_free_blocks
                           for need, cap, m, sizes in zip(headroom, limits, managers, by_lane, strict=True))
            value = dict(accepted=accepted, ids=[x["rid"] for x in items],
                         required_kv_tokens=sum(headroom), capacity_tokens=sum(limits))
            dt = self.profile.get("move_base", c["move_base"]) if accepted else (self.profile.get("move_reject") or c["move_base"])
            if self.placements[gpu].node != self.placements[a.peer].node:
                dt += c.get("cross_node_move_s", 0.)
            if accepted:
                for x,n in zip(items, counts, strict=True):
                    self.grow(a.peer, x, n-1, a.version)
                    self.free(gpu, x["rid"])
                install = max(self.rollout.step_time(sum(sizes), sum(n*n for n in sizes), 0, len(sizes))
                              for sizes in by_lane if sizes)
                if "move_install" in self.profile:
                    base, slope = self.profile["move_install"]
                    install = base + slope * sum(counts)
                value.update(install_start=self.clock + dt, install_end=self.clock + dt + install,
                             source_node=self.placements[gpu].node, destination_node=self.placements[a.peer].node)
                dt += install
                value.update(rebuilt_tokens=sum(counts), kv_bytes=sum(counts)*cfg["kv_bytes_per_token"],
                             kv_transfer_bytes=0, transfer="seed_position_and_reprefill")
        elif a.kind == "train":
            tokens = sum(len(x["tokens"]) for x in items)
            self.train_tokens[gpu] += tokens
            dt = train_seconds(cfg, [len(x["prompt"]) + len(x["tokens"]) for x in items])
            value = dict(tokens=tokens, loss_sum=0.)
        elif a.kind == "reduce_grads":
            if (p["counts"] != self.train_tokens or gpu in self.reduced_groups
                    or ((self.train_tokens[gpu] or gpu == p["owner"]) and self.roles[gpu] != "train")):
                raise RuntimeError("simulated gradient exchange has inconsistent contributions")
            self.reduced_groups.add(gpu)
            dt = self.profile.get("reduce", cfg.get("sim_replica_reduce_s", c.get("replica_reduce", c["switch"])))
            value = dict(owner=p["owner"], group_tokens=self.train_tokens[gpu], global_tokens=sum(self.train_tokens))
        elif a.kind == "commit":
            if sum(self.train_tokens) != p["tokens"] or (
                any(n and i != gpu for i,n in enumerate(self.train_tokens)) and len(self.reduced_groups) != cfg["groups"]
            ):
                raise RuntimeError("incomplete simulated gradient accumulation")
            self.train_tokens = [0] * cfg["groups"]
            self.reduced_groups.clear()
            dt = cfg.get("sim_publish_s", 1.)
            if not p["durable"]:
                dt = cfg.get("sim_export_s",dt)
                if cfg.get("weight_sync") == "tensor" and not p["save_hf"]:
                    dt = cfg.get("sim_optimizer_s", 1.)
            mode = "durable" if p["durable"] else "export" if p["save_hf"] else "step"
            dt = self.profile.get(mode, dt)
            value = dict(tokens=p["tokens"], published=a.version+1,durable=p["durable"], hf_checkpoint=p["save_hf"])
        else:
            raise ValueError(a.kind)
        # SGLang reserves its KV pool up front. Free CUDA bytes and free
        # radix blocks are independent constraints on an incoming handoff.
        value["kv_used_blocks"] = sum(m.num_used_blocks for m in self.managers[gpu])
        value["kv_free_blocks"] = sum(m.num_free_blocks for m in self.managers[gpu])
        if a.peer is not None:
            value.update(peer_free_bytes=self.free_bytes(a.peer),
                         peer_kv_used_blocks=sum(m.num_used_blocks for m in self.managers[a.peer]),
                         peer_kv_free_blocks=sum(m.num_free_blocks for m in self.managers[a.peer]))
        receipt = dict(start=self.clock, end=self.clock+dt, value=value,
                       free_bytes=self.free_bytes(gpu))
        if (a.kind == "switch" and p.get("publish_from") is not None and
                cfg.get("max_train_groups", 1) > 1 and "tensor_collective_s" in self.profile):
            local_load = cell.get("local_load", load) if cell else load
            arrival = self.clock + value["unload_s"] + local_load
            self.publish_pending.append((arrival, a, receipt))
            if len(self.publish_pending) == cfg["groups"]:
                latest = max(t for t, _, _ in self.publish_pending)
                end = latest + self.profile["tensor_collective_s"]
                for t, action, report in self.publish_pending:
                    fields = report["value"]
                    fields.update(publish_wait_s=latest-t, tensor_publish_s=end-t,
                                  load_s=end-report["start"]-fields["unload_s"])
                    report["end"] = end
                    heapq.heappush(self.queue, (end, next(self.serial), action, report))
                self.publish_pending.clear()
            return
        heapq.heappush(self.queue, (self.clock+dt, next(self.serial), a, receipt))

    def wait(self):
        self.clock = self.queue[0][0]
        done = []
        while self.queue and self.queue[0][0] == self.clock:
            _, _, a, receipt = heapq.heappop(self.queue)
            done.append((a, receipt))
        return done

    def close(self, failed=False):
        pass
