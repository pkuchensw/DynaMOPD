"""A sleeping SGLang TP mesh leased by two local FSDP groups at token boundaries."""
import math
from threading import Barrier
from time import monotonic

from .sglang_engine import Server


class WideRollout:
    def __init__(self, cfg, workers, profile):
        self.groups = tuple(w.index for w in workers)
        self.devices = [d for w in workers for d in w.devices]
        self.barrier = Barrier(len(workers),timeout=cfg.get("action_timeout_s",180))
        self.handles = {}
        self.server = None
        self.active = None
        self.profile = profile
        self.cfg = cfg

    def prepare(self):
        import pynvml
        pynvml.nvmlInit()
        devices = [pynvml.nvmlDeviceGetHandleByUUID(d.encode()) if d.startswith("GPU-")
                   else pynvml.nvmlDeviceGetHandleByIndex(int(d)) for d in self.devices]
        def free():
            return [pynvml.nvmlDeviceGetMemoryInfo(h).free for h in devices]
        cfg = dict(self.cfg,engine_kv_tokens=self.profile.context_capacity+128)
        cfg["sglang_limits"] = dict(cfg.get("sglang_limits",{}),max_running_requests=4,cuda_graph_max_bs=1)
        before = free()
        start = monotonic()
        self.server = Server(cfg,self.devices,"rollout",cfg["staged"][cfg["student"]],"v0",cpu_weight_backup=True)
        loaded = free()
        ready_at = monotonic()
        self.server.sleep()
        parked = free()
        self.wake_bytes = max(max(0,p-q) for p,q in zip(parked,loaded,strict=True))
        self.startup = dict(groups=self.groups, devices=self.devices, bootstrap_s=monotonic()-start,
                            engine_ready_s=ready_at-start, park_s=monotonic()-ready_at,
                            wake_bytes=self.wake_bytes, idle_device_bytes=[p-q for p,q in zip(before,parked,strict=True)])
        if self.profile.measuring:
            start = monotonic()
            self.server.wake()
            awake = monotonic()
            size = min(cfg["quantum"], cfg["max_prompt"])
            item = dict(rid="tp-calibration",prompt=[1]*size,tokens=[])
            self.server.batch([item],dict(max_new_tokens=0,temperature=1.),return_logprob=True,logprob_start_len=[size-1])
            prefilled = monotonic()
            (_,result,_), = self.server.batch([item],dict(max_new_tokens=min(cfg["quantum"],cfg["max_new_tokens"]),
                                                          temperature=1.,ignore_eos=True))
            decoded = monotonic()
            self.server.sleep()
            self.startup["measured_costs"] = dict(wake_s=awake-start,sleep_s=monotonic()-decoded,
                  prefill_token_s=(prefilled-awake)/size, decode_token_s=(decoded-prefilled)/len(result["output_ids"]))
        return self.startup

    def publish(self, group, handles, version, index, last):
        """Both narrow groups keep the current FSDP IPC bucket alive until every wide rank acks."""
        self.handles[group] = (version,index,handles)
        try:
            self.barrier.wait()
            if group == self.groups[0]:
                ordered = [self.handles[g] for g in self.groups]
                if any((v,i)!=(version,index) for v,i,_ in ordered) or self.active is not None:
                    raise RuntimeError("out-of-order wide mesh publication")
                if index == 0:
                    self.server.wake(["weights"])
                self.server.publish_bucket([h for _,_,part in ordered for h in part],version,last)
                if last:
                    self.server.sleep(["weights"])
            self.barrier.wait()
        except BaseException:
            self.barrier.abort()
            raise

    def execute(self, action, leader, donor):
        s, v, p = self.server, action.version, action.payload
        receipt = dict(kv_transfer_bytes=0,physical_gpus=self.devices,donor_role=donor.role,peer_evicted=[],
                       peer_free_bytes=donor.available(),engine_weight_version=s.version)
        if leader.version != v or donor.version != v or leader.role != "rollout":
            raise RuntimeError("stale wide worker lease")
        if action.kind == "tp_expand":
            item, = p["items"]
            rid = item["rid"]
            if (self.active is not None or set(leader.infer.owned)!={rid} or donor.infer.owned
                    or item["student_version"] != v or leader.infer.owned[rid] != leader.infer.sampling_state(item)):
                raise RuntimeError("cannot take sampling ownership at this boundary")
            context = len(item["prompt"])+len(item["tokens"])
            common = dict(from_tp=self.profile.base,to_tp=self.profile.wide,kv_context_tokens=context-1,
                          **{k:val for k,val in p.items() if k!="items"})
            free = min(leader.available(),donor.available())
            if free < self.wake_bytes+self.cfg["memory_margin"]:
                return dict(receipt,**common,accepted=False,capacity_free_bytes=free,required_bytes=self.wake_bytes)
            start = monotonic()
            s.wake()
            try:
                if s.version != f"v{v}":
                    # A fresh executor resuming a durable checkpoint has no earlier publication.
                    s.publish(leader.student_image(v),v)
                s.confirm_version(f"v{v}")
                awake = monotonic()
                (_,_,meta), = s.batch([item],dict(max_new_tokens=0, temperature=1.),return_logprob=True,
                                      logprob_start_len=[context-2])
                entries = meta["input_token_logprobs"]
                if len(entries)!=2 or entries[-1][1]!=item["tokens"][-1]:
                    raise RuntimeError("wide prefix replay lost the last committed token")
                error = abs(entries[-1][0]-item["last_logp"])
                if not math.isfinite(error) or error > self.cfg.get("tp_logprob_tolerance",.25):
                    raise RuntimeError(f"wide student disagrees with rollout at handoff: {error}")
                self.active = (leader.index,donor.index,v,rid)
            except BaseException:
                s.sleep()
                raise
            return dict(receipt,**common,accepted=True,engine_weight_version=s.version,
                        wake_s=awake-start,prefill_s=monotonic()-awake,rebuilt_tokens=context-meta["cached_tokens"],
                        handoff_logp_error=error,transfer="seed_position_and_reprefill",peer_free_bytes_after=donor.available())
        if self.active is None or self.active[:3] != (leader.index,donor.index,v):
            raise RuntimeError("missing or stale wide layout")
        if action.kind == "decode":
            item, = p["items"]
            if item["rid"] != self.active[3] or item["student_version"] != v or s.version!=f"v{v}":
                raise RuntimeError("stale wide decode")
            result = leader.infer.decode(s,p)
            return dict(receipt,**result,inference_tp_size=s.tp_size,replica_requests=[1])
        if action.kind != "tp_shrink":
            raise ValueError(action.kind)
        start = monotonic()
        s.sleep()
        self.active = None
        return dict(receipt,from_tp=self.profile.wide,to_tp=self.profile.base,reason=p["reason"],
                    kv_context_tokens=0,peer_kv_used_blocks=0,sleep_s=monotonic()-start,
                    peer_free_bytes_after=donor.available())

    def close(self):
        self.barrier.abort()
        if self.server is not None:
            self.server.close()
