"""Transformers 5.x causal LM engine for dense full-attention Qwen2/3 replicas."""
import gc
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, DynamicCache

from ...algorithms import TokenBatch, build_algorithm


def selected_logp(logits, ids, temperature):
    scaled = logits.float() / temperature
    return scaled.gather(-1, ids.unsqueeze(-1)).squeeze(-1) - scaled.logsumexp(-1)


def kv_len(cache):
    return cache[0][0].size(2) if cache else 0


def packed_forward(model, rows, past, device, cache=True):
    """Left-pad past KV, right-pad query tokens, and use logical RoPE positions."""
    sizes = [len(row) for row in rows]
    old = [kv_len(entry) for entry in past]
    width, history = max(sizes), max(old)
    ids = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), history + width), dtype=torch.long, device=device)
    pos = torch.zeros_like(ids)
    for i, (row, size, n) in enumerate(zip(rows, sizes, old, strict=True)):
        ids[i, :size] = torch.as_tensor(row, device=device)
        mask[i, history-n:history+size] = 1
        pos[i, :size] = torch.arange(n, n+size, device=device)
    packed = None
    if history:
        example = next(entry for entry in past if entry)
        layers = []
        for layer in range(len(example)):
            pair = []
            for side in (0, 1):
                template = example[layer][side]
                blocks = template.new_zeros((len(rows), template.size(1), history, template.size(3)))
                for i, n in enumerate(old):
                    if n:
                        blocks[i:i+1, :, history-n:] = past[i][layer][side]
                pair.append(blocks)
            layers.append(tuple(pair))
        packed = DynamicCache(layers)
    output = model.model(input_ids=ids, attention_mask=mask, position_ids=pos,
                         past_key_values=packed, use_cache=cache)
    states = [output.last_hidden_state[i, :n] for i, n in enumerate(sizes)]
    return states, output.past_key_values, mask


def unpack_row(layers, i, mask):
    # Padding can occur between the prefill and the first incremental token.
    idx = mask[i].nonzero().flatten()
    return [(k[i:i+1].index_select(2, idx), v[i:i+1].index_select(2, idx)) for k, v, *_ in layers]


def forward_rows(model, rows, past, device, cache=True):
    states, packed, mask = packed_forward(model, rows, past, device, cache)
    result = [None] * len(rows)
    if cache:
        layers = list(packed)
        result = [unpack_row(layers, i, mask) for i in range(len(rows))]
    return states, result


class Worker:
    def __init__(self, rank, cfg):
        torch.set_num_threads(cfg.get("cpu_threads", 2))
        torch.cuda.set_device(rank)
        self.device = torch.device("cuda", rank)
        self.cfg = cfg
        self.algorithm = build_algorithm(cfg)
        self.role = "cold"
        self.version = -1
        self.model = None
        self.optimizer = None
        self.cache = {}
        self.generators = {}
        self.grad_tokens = 0
        self.temperature = cfg["temperature"]
        self.excluded = None
        self.templates = {}
        if cfg.get("stage_host_weights", True):
            for path in dict.fromkeys([cfg["student"], *cfg["teachers"].values()]):
                self.templates[path] = self.host_model(path)

    @staticmethod
    def host_model(path):
        model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"":"cpu"},
                                                     local_files_only=True, attn_implementation="sdpa").eval()
        # Materialize mmap pages before the first round; switch time then reflects H2D.
        for param in model.parameters():
            param.data = param.detach().clone()
        return model

    def replica(self, path):
        template = self.templates.get(str(path))
        if template is None:
            return AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"":self.device},
                                                        local_files_only=True, attn_implementation="sdpa").eval()
        memo = {id(w):torch.nn.Parameter(w.detach().to(self.device), requires_grad=False)
                for w in template.parameters()}
        memo.update({id(b):b.to(self.device) for b in template.buffers()})
        return copy.deepcopy(template, memo)

    def fence(self):
        torch.cuda.synchronize(self.device)

    def image_probe(self):
        pieces = [w.detach().reshape(-1)[::max(1, w.numel()//64)][:64]
                  for w in self.model.parameters()]
        raw = torch.cat(pieces).view(torch.uint8).cpu().numpy().tobytes()
        return hashlib.sha256(raw).hexdigest()

    def version_path(self, version):
        return Path(self.cfg["state_dir"]) / f"v{version}"

    def switch(self, p, version):
        role = p["role"]
        if self.grad_tokens or self.generators:
            raise RuntimeError("switch with live gradients or sampling ownership")
        old = self.role
        start = time.monotonic()
        lost = list(self.cache)
        self.cache.clear()
        self.optimizer = self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        self.fence()
        unloaded = time.monotonic()
        is_student = role in ("rollout", "train")
        if is_student:
            path = self.cfg["student"] if version == 0 else self.version_path(version)
        else:
            path = self.cfg["teachers"][role]
        self.model = self.replica(path)
        if any(t != "full_attention" for t in (getattr(self.model.config, "layer_types", None) or [])):
            raise ValueError("this engine requires full-attention KV layers")
        self.model.requires_grad_(role == "train")
        self.excluded = torch.tensor([i for i in self.cfg["excluded_ids"] if i < self.model.config.vocab_size],
                                     dtype=torch.long, device=self.device)
        if role == "train":
            if self.cfg.get("activation_checkpointing", False):
                if self.model.config.attention_dropout:
                    raise ValueError("checkpointed training requires zero attention dropout")
                self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
                self.model.train()
            # BF16 parameter image is identical for rollout and differentiable recomputation.
            # FP32 AdamW master and moments reside on CPU and survive role reassignment.
            self.master = [torch.nn.Parameter(w.detach().to("cpu", torch.float32))
                           for w in self.model.parameters()]
            self.optimizer = torch.optim.AdamW(self.master, lr=self.cfg["lr"], foreach=False,
                                               weight_decay=self.cfg.get("weight_decay", 0.0))
            if version:
                checkpoint = torch.load(self.version_path(version) / "adam.pt", mmap=True,
                                        map_location="cpu", weights_only=True)
                for master, saved in zip(self.master, checkpoint["master"], strict=True):
                    master.data.copy_(saved)
                self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.role, self.version = role, version
        self.fence()
        return dict(old=old, new=role, evicted=lost, unload_s=unloaded-start,
                    load_s=time.monotonic()-unloaded, image_probe=self.image_probe() if is_student else None)

    @torch.inference_mode()
    def decode(self, p):
        items = p["items"]
        updates = {x["rid"]: dict(start=len(x["tokens"]), tokens=[], logp=[], finish=None) for x in items}
        rows, caches = [], []
        for x in items:
            rid = x["rid"]
            ctx = x["prompt"] + x["tokens"]
            past = self.cache.pop(rid, None)
            if past and kv_len(past) != len(ctx)-1:
                raise RuntimeError("stale student KV")
            rows.append(ctx[kv_len(past):])
            caches.append(past)
            if rid not in self.generators:
                if x["tokens"]:
                    raise RuntimeError("missing migrated RNG")
                self.generators[rid] = torch.Generator(device=self.device).manual_seed(x["seed"])
        hidden, packed, mask = packed_forward(self.model, rows, caches, self.device)
        last = torch.stack([h[-1] for h in hidden])
        for step in range(p["steps"]):
            logits = self.model.lm_head(last).float() / self.temperature
            logits.index_fill_(-1, self.excluded, -torch.inf)
            lp = logits.log_softmax(-1)
            chosen = torch.zeros(len(items), dtype=torch.long, device=self.device)
            for i, x in enumerate(items):
                if not updates[x["rid"]]["finish"]:
                    chosen[i] = torch.multinomial(lp[i].exp(), 1, generator=self.generators[x["rid"]])[0]
            sampled = chosen.tolist()
            logp = lp.gather(1, chosen[:, None]).flatten().tolist()
            for i, x in enumerate(items):
                rid, token = x["rid"], sampled[i]
                u = updates[rid]
                if u["finish"]:
                    continue
                u["tokens"].append(token)
                u["logp"].append(logp[i])
                if token in self.cfg["eos_ids"]:
                    u["finish"] = "eos"
                elif u["start"] + len(u["tokens"]) == x["limit"]:
                    u["finish"] = "length"
                if u["finish"]:
                    del self.generators[rid]
            if step+1 == p["steps"] or all(u["finish"] for u in updates.values()):
                break
            pos = mask.sum(1, keepdim=True)
            mask = torch.cat((mask, torch.ones_like(pos)), dim=1)
            output = self.model.model(input_ids=chosen[:, None], past_key_values=packed,
                                      attention_mask=mask, position_ids=pos, use_cache=True)
            packed = output.past_key_values
            last = output.last_hidden_state[:, -1]
        layers = list(packed)
        for i, x in enumerate(items):
            if not updates[x["rid"]]["finish"]:
                self.cache[x["rid"]] = unpack_row(layers, i, mask)
        return {"updates": updates}

    @torch.inference_mode()
    def score(self, p):
        rows, caches, offsets = [], [], []
        for x in p["items"]:
            start = len(x["prompt"]) + x["start"] - 1
            past = self.cache.pop(x["rid"], None)
            if kv_len(past) != start:
                past = None
            ctx = x["prompt"] + x["tokens"][:-1]
            rows.append(ctx[kv_len(past):])
            offsets.append(start-kv_len(past))
            caches.append(past)
        hidden, next_cache = forward_rows(self.model, rows, caches, self.device)
        result = {}
        for i, x in enumerate(p["items"]):
            target = torch.tensor(x["tokens"][x["start"]:], device=self.device)
            scores = self.logprobs(hidden[i][offsets[i]:], target)
            rid = x["rid"]
            if not x["final"]:
                self.cache[rid] = next_cache[i]
            result[rid] = dict(start=x["start"], end=len(x["tokens"]), labels=scores.tolist(),
                               reused=kv_len(caches[i]), retained=not x["final"])
        dropped = []
        while sum(map(kv_len, self.cache.values())) > self.cfg["teacher_kv_tokens"]:
            victim = next(iter(self.cache))
            dropped.append(victim)
            del self.cache[victim]
        return {"scores": result, "evicted": dropped}

    def logprobs(self, hidden, target):
        width = self.cfg.get("head_chunk", 64)
        parts = []
        for i in range(0, len(target), width):
            logits = self.model.lm_head(hidden[i:i+width]).float()
            logits.index_fill_(-1, self.excluded, -torch.inf)
            parts.append(selected_logp(logits, target[i:i+width], self.temperature))
        return torch.cat(parts)

    def train(self, p):
        items = p["items"]
        inputs = [x["prompt"] + x["tokens"][:-1] for x in items]
        old_logps = [None] * len(items)
        if self.algorithm.recompute_old:
            with torch.no_grad():
                old_hidden, _ = forward_rows(self.model, inputs, [None] * len(inputs), self.device, cache=False)
                old_logps = [self.logprobs(h[len(x["prompt"])-1:], torch.tensor(x["tokens"], device=self.device))
                             for x,h in zip(items, old_hidden, strict=True)]
        hidden, _ = forward_rows(self.model, inputs, [None] * len(inputs), self.device, cache=False)
        terms = []
        max_error = 0.
        for x, h, old in zip(items, hidden, old_logps, strict=True):
            ids = torch.tensor(x["tokens"], device=self.device)
            lp = self.logprobs(h[len(x["prompt"])-1:], ids)
            teacher = torch.tensor(x["labels"], device=self.device)
            behavior = torch.tensor(x["behavior"], device=self.device)
            terms.append(self.algorithm.loss_sum(TokenBatch(lp, behavior, teacher, x, old_logp=old)))
            max_error = max(max_error, (lp.detach()-behavior).abs().max().item())
        loss = torch.stack(terms).sum()
        loss.backward()
        count = sum(len(x["tokens"]) for x in items)
        # Accumulate in FP32 even between microbatches. Normalize only at commit.
        for master, param in zip(self.master, self.model.parameters(), strict=True):
            grad = param.grad.detach().to("cpu", torch.float32)
            if master.grad is None:
                master.grad = grad
            else:
                master.grad.add_(grad)
            param.grad = None
        self.grad_tokens += count
        return dict(tokens=count, loss_sum=loss.item(), max_logp_error=max_error)

    def commit(self, p):
        n = p["tokens"]
        if self.grad_tokens != n or not n:
            raise RuntimeError("effective-token denominator disagrees with accumulated gradients")
        for param in self.master:
            param.grad.div_(n)
        norm = torch.nn.utils.clip_grad_norm_(self.master, self.cfg["clip"], error_if_nonfinite=True)
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        # A real parameter delta is checked on the bf16 image seen by subsequent rollout.
        delta = 0.
        for master, weight in zip(self.master, self.model.parameters(), strict=True):
            updated = master.detach().to(device=self.device, dtype=weight.dtype)
            delta = max(delta, (updated-weight.detach()).abs().max().item())
            weight.data.copy_(updated)
        path = self.version_path(self.version+1)
        staging = path.with_suffix(".partial")
        staging.mkdir()
        if "scheduler_state" in p:
            (staging / "scheduler.json").write_text(json.dumps(p["scheduler_state"], allow_nan=False))
        self.model.save_pretrained(staging)
        torch.save(dict(master=[w.detach() for w in self.master], optimizer=self.optimizer.state_dict()),
                   staging / "adam.pt")
        os.rename(staging, path)
        self.grad_tokens = 0
        self.master = []
        self.optimizer = None
        return dict(tokens=n, grad_norm=norm.item(), max_weight_delta=delta, published=self.version+1,
                    image_probe=self.image_probe())

    @torch.inference_mode()
    def install(self, p):
        if set(self.generators).intersection(p["rng"]):
            raise RuntimeError("duplicate request at destination")
        rows = [(x["prompt"] + x["tokens"])[:-1] for x in p["items"]]
        config = self.model.config
        head_dim = getattr(config, "head_dim", None) or config.hidden_size//config.num_attention_heads
        per_token = 4 * config.num_hidden_layers * config.num_key_value_heads * head_dim
        required = 3 * per_token * len(rows) * (max(map(len, rows)) + self.cfg["quantum"])
        if torch.cuda.mem_get_info(self.device)[0] < required + self.cfg["memory_margin"]:
            return {"accepted": False}
        try:
            _, caches = forward_rows(self.model, rows, [None]*len(rows), self.device)
        except torch.OutOfMemoryError:
            return {"accepted": False}
        for x, cache in zip(p["items"], caches, strict=True):
            rid = x["rid"]
            self.cache[rid] = cache
            self.generators[rid] = torch.Generator(device=self.device)
            self.generators[rid].set_state(torch.tensor(list(p["rng"][rid]), dtype=torch.uint8))
        return dict(accepted=True, rebuilt_tokens=sum(map(len, rows)), kv_bytes=per_token*sum(map(len, rows)))

    def execute(self, kind, p, version):
        if kind == "switch":
            return self.switch(p, p.get("load_version",version))
        if version != self.version:
            raise RuntimeError("stale command version")
        expected = {"decode": "rollout", "export": "rollout", "install": "rollout", "release": "rollout",
                    "score": p.get("teacher"), "train": "train", "commit": "train"}
        if self.role != expected[kind]:
            raise RuntimeError(f"{kind} on {self.role}")
        if kind == "export":
            return {"rng": {rid: bytes(self.generators[rid].get_state().tolist()) for rid in p["ids"]}}
        if kind == "release":
            for rid in p["ids"]:
                del self.cache[rid], self.generators[rid]
            return {"released": p["ids"]}
        return getattr(self, kind)(p)
