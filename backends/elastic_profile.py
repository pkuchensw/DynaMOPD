"""Batch-one cost model shared by the native standby mesh and SimRL."""
import math

from ..cluster.topology import local_pairs


class TailProfile:
    """Measured batch-one latencies. Refine estimates using acknowledged operations."""
    borrow_trainer = True
    preserves_donor = True

    def __init__(self, cfg, placements):
        options = cfg["rollout_tp_expansion"]
        self.base = cfg.get("inference_tp_size",cfg["group_size"])
        self.wide = 2*cfg["group_size"]
        if options.get("source_tp",self.base) != self.base or options.get("target_tp",self.wide) != self.wide:
            raise ValueError("tail cost profile does not match the configured narrow and wide TP")
        self.context_capacity = cfg["max_prompt"]+cfg["max_new_tokens"]+1
        self.scratch = 0
        self.rates = {self.base: options["narrow_token_s"], self.wide: options["wide_token_s"]}
        self.enter, self.leave = options["enter_s"], options["exit_s"]
        self.prefill = options["prefill_token_s"]
        if not all(math.isfinite(v) and v > 0 for v in (*self.rates.values(),self.enter,self.leave,self.prefill)):
            raise ValueError("tail TP timings must be positive finite seconds")
        self.partners = {p.group: [] for p in placements}
        self.pairs = []
        for a,b in options.get("group_pairs", local_pairs(placements)):
            if (a == b or a not in self.partners or b not in self.partners or self.partners[a] or self.partners[b]
                    or placements[a].node != placements[b].node):
                raise ValueError("standby meshes require disjoint, node-local pairs of FSDP groups")
            self.pairs.append((a,b))
            self.partners[a].append(b)
            self.partners[b].append(a)
        if not self.pairs:
            raise ValueError("no local rollout group pair configured")

    def predict(self, context, remaining, width):
        return remaining*self.rates[width]

    def transition(self, leader, peer, role, context, entering):
        return {"transition_s": (self.enter if entering else self.leave)+self.prefill*context,
                "kv_transfer_bytes":0}

    def observe(self,action,value,duration,quantum):
        if action.kind == "tp_expand" and value.get("accepted") and "wake_s" in value:
            self.enter = .75*self.enter + .25*value["wake_s"]
            n = max(1,value["kv_context_tokens"])
            self.prefill = .75*self.prefill + .25*value["prefill_s"]/n
        if action.kind == "tp_shrink" and "sleep_s" in value:
            self.leave = .75*self.leave + .25*value["sleep_s"]
        if action.kind == "decode" and len(value["updates"]) == 1:
            n = len(next(iter(value["updates"].values()))["tokens"])
            width = self.wide if action.peer is not None else self.base
            alpha = .25*min(n/quantum,1.)
            self.rates[width] = (1.-alpha)*self.rates[width] + alpha*duration/n
