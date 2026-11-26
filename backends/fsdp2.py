"""One persistent FSDP2 shard per physical GPU in a group, with verl wrapping/offload."""
import datetime
import json
from itertools import accumulate
import os
import shutil
import tempfile
import time
from pathlib import Path
import traceback

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (get_model_state_dict, get_state_dict, set_model_state_dict,
                                                    set_state_dict, StateDictOptions)
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import FSDPModule, MixedPrecisionPolicy
from transformers import AutoModelForCausalLM

from ..algorithms import TokenBatch, build_algorithm
from ..scheduler.batching import training_waves
from ..utils.fsdp_utils import (
    apply_fsdp2, load_fsdp2_model_to_gpu, load_fsdp_optimizer, offload_fsdp2_model_to_cpu,
)


class Shard:
    def __init__(self, cfg, device, rank, world, rendezvous):
        self.cfg, self.rank, self.world = cfg, rank, world
        self.algorithm = build_algorithm(cfg)
        self.device = torch.device("cuda", device)
        torch.cuda.set_device(device)
        torch.set_num_threads(cfg.get("cpu_threads", 2))
        dist.init_process_group("nccl", init_method=rendezvous, rank=rank, world_size=world,
                                device_id=self.device, timeout=datetime.timedelta(minutes=15))
        mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("fsdp",))
        self.attention = cfg.get("trainer_attention", "sdpa")
        self.model = AutoModelForCausalLM.from_pretrained(cfg["staged"][cfg["student"]], dtype=torch.float32,
                                                         local_files_only=True, attn_implementation=self.attention)
        if self.model.config.attention_dropout:
            raise ValueError("nonzero attention dropout changes the rollout policy")
        if cfg.get("activation_checkpointing", True):
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
        self.model.train()
        policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        apply_fsdp2(self.model, dict(mesh=mesh, mp_policy=policy, reshard_after_forward=True), {})
        for module in self.model.modules():
            if isinstance(module, FSDPModule):
                # Every response position contributes once across the mesh.
                module.set_gradient_divide_factor(1.0)
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=cfg["lr"],
                                     weight_decay=cfg.get("weight_decay", 0.), fused=True)
        self.tokens = 0
        self.version = 0
        self.host_moments = {}
        self.host_moments_version = None
        self.host_params = {}
        self.host_params_version = None
        self.publisher = None
        if "worker_group" in cfg and cfg.get("weight_sync", "tensor") == "tensor":
            from .weight_sync import TensorPublisher
            self.publisher = TensorPublisher(cfg, rank, self.device, self.model)
        self.awake = True
        self.resident = True
        self.sleep()
        if self.publisher is not None:
            self.publisher.prepare()

    def path(self, v):
        return Path(self.cfg["state_dir"])/f"v{v}"

    @staticmethod
    def copy_to_host(cache, key, value):
        if key not in cache:
            cache[key] = value.detach().to("cpu", copy=True)
        else:
            host = cache[key]
            host_local = host.to_local() if hasattr(host, "to_local") else host
            local = value.to_local() if hasattr(value, "to_local") else value
            host_local.copy_(local, non_blocking=True)

    @torch.no_grad()
    def mirror_training_state(self):
        """Build a versioned CPU snapshot after acknowledging rollout publication."""
        if not self.resident:
            return
        moments = self.cfg.get("mirror_optimizer_on_rollout", False) and self.host_moments_version != self.version
        params = self.cfg.get("mirror_parameters_on_rollout", False) and self.host_params_version != self.version
        if moments:
            for param, state in self.opt.state.items():
                for key, value in state.items():
                    if torch.is_tensor(value) and value.device.type == "cuda":
                        self.copy_to_host(self.host_moments, (param, key), value)
        if params:
            for param in self.model.parameters():
                self.copy_to_host(self.host_params, param, param)
        torch.cuda.current_stream(self.device).synchronize()
        if params:
            self.host_params_version = self.version
        if moments:
            self.host_moments_version = self.version

    def sleep(self, retain=False):
        if self.tokens:
            raise RuntimeError("cannot release a trainer with outstanding gradients")
        mirrored = master_bytes = 0
        if not retain and self.resident:
            for param, state in self.opt.state.items():
                for k,v in state.items():
                    if torch.is_tensor(v):
                        host = self.host_moments.get((param,k))
                        if host is not None and self.host_moments_version == self.version:
                            state[k] = host
                            local = host.to_local() if hasattr(host, "to_local") else host
                            mirrored += local.numel()*local.element_size()
                        else:
                            state[k] = v.cpu()
            if self.host_params_version == self.version:
                # Use FSDPModule._apply so FSDP reshards and resets its local shards
                # exactly as it does for model.cpu(). Buffers still use the native path.
                def on_cpu(tensor):
                    nonlocal master_bytes
                    host = self.host_params.get(tensor)
                    if host is None:
                        return tensor.cpu()
                    local = host.to_local() if hasattr(host, "to_local") else host
                    master_bytes += local.numel() * local.element_size()
                    return host.detach()
                self.model._apply(on_cpu)
            else:
                offload_fsdp2_model_to_cpu(self.model, empty_cache=False)
            self.resident = False
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        self.awake = False
        return dict(trainer_state_device="cuda" if self.resident else "cpu", reused_host_optimizer_bytes=mirrored, reused_host_parameter_bytes=master_bytes)

    def activate(self, version, optimizer_owner=True):
        reused = self.resident and self.version == version
        if not self.resident:
            load_fsdp2_model_to_gpu(self.model)
            load_fsdp_optimizer(self.opt, self.device)
            self.resident = True
        if self.version != version:
            checkpoint = self.path(version) / "training"
            if optimizer_owner:
                weights, optimizer = get_state_dict(self.model, self.opt)
                dcp.load({"model":weights, "optimizer":optimizer}, checkpoint_id=checkpoint)
                set_state_dict(self.model, self.opt, model_state_dict=weights, optim_state_dict=optimizer)
            else:
                weights = get_model_state_dict(self.model)
                dcp.load({"model": weights}, checkpoint_id=checkpoint)
                set_model_state_dict(self.model, model_state_dict=weights)
            self.version = version
        self.awake = True
        return dict(world=self.world, reused_trainer_state=reused,
                    local_numel=sum(p.to_local().numel() for p in self.model.parameters()),
                    global_numel=sum(p.numel() for p in self.model.parameters()))

    def response_logits(self, batch):
        lengths = [len(x["prompt"])+len(x["tokens"])-1 for x in batch]
        if self.attention == "flash_attention_2":
            offsets = list(accumulate(lengths, initial=0))
            inputs, positions, targets = [], [], []
            for i, x in enumerate(batch):
                inputs.extend(x["prompt"] + x["tokens"][:-1])
                positions.extend(range(lengths[i]))
                targets.extend(range(offsets[i]+len(x["prompt"])-1, offsets[i+1]))
            cu = torch.tensor(offsets, dtype=torch.int32, device=self.device)
            longest = max(lengths)
            output = self.model(input_ids=torch.tensor([inputs], device=self.device),
                                position_ids=torch.tensor([positions], device=self.device),
                                cu_seq_lens_q=cu, cu_seq_lens_k=cu, max_length_q=longest, max_length_k=longest,
                                logits_to_keep=torch.tensor(targets, device=self.device), use_cache=False)
            return output.logits[0].split([len(x["tokens"]) for x in batch])
        ids = torch.zeros((len(batch), max(lengths)), dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, x in enumerate(batch):
            ids[i,:lengths[i]] = torch.tensor(x["prompt"]+x["tokens"][:-1], device=self.device)
            mask[i,:lengths[i]] = 1
        output = self.model(input_ids=ids, attention_mask=mask, use_cache=False)
        return [output.logits[i,len(x["prompt"])-1:lengths[i]] for i,x in enumerate(batch)]

    def sampled_logp(self, logits, labels):
        if self.attention == "flash_attention_2":
            from ..utils.torch_functional import logprobs_from_logits_flash_attn
            return logprobs_from_logits_flash_attn(logits, labels, inplace_backward=False)
        return logits.float().log_softmax(-1).gather(1, labels[:, None]).flatten()

    def train(self, items):
        results = []
        inputs, samples = [], []
        waves = list(training_waves(items, self.cfg, lambda x: len(x["prompt"])+len(x["tokens"])))
        for block, ranks, loads in waves:
            part = [block[i] for i in ranks[self.rank]]
            results.append(self.train_part(part, block))
            inputs.append(loads[self.rank])
            samples.append(len(part))
        return dict(tokens=sum(r["tokens"] for r in results), loss_sum=sum(r["loss_sum"] for r in results),
                    max_logp_error=max(r["max_logp_error"] for r in results),
                    input_tokens_per_microbatch=inputs, samples_per_microbatch=samples)

    def train_part(self, part, items):
        # Empty ranks enter the same collectives, with an exactly zero objective.
        batch = part or items[:1]
        old_logps = [None] * len(batch)
        if self.algorithm.recompute_old:
            with torch.no_grad():
                old_logps = [self.sampled_logp(response, torch.tensor(x["tokens"], device=self.device))
                             for x, response in zip(batch, self.response_logits(batch), strict=True)]
        terms, errors = [], []
        for x, response, old in zip(batch, self.response_logits(batch), old_logps, strict=True):
            ids_y = torch.tensor(x["tokens"], device=self.device)
            if "teacher_topk" not in x:
                student = self.sampled_logp(response, ids_y)
            else:
                logits = response.float()
                log_z = logits.logsumexp(-1)
                student = logits.gather(1, ids_y[:,None]).flatten()-log_z
            behavior = torch.tensor(x["behavior"], device=self.device)
            target = torch.tensor(x["labels"], device=self.device)
            student_at_ids = teacher_topk = None
            if "teacher_topk" in x:
                positions = x["teacher_topk"]
                selected = torch.tensor([[idx for idx,_ in row] for row in positions], device=self.device)
                teacher_topk = torch.tensor([[lp for _,lp in row] for row in positions], device=self.device)
                student_at_ids = logits.gather(1, selected) - log_z[:,None]
            terms.append(self.algorithm.loss_sum(TokenBatch(student, behavior, target, x, student_at_ids, teacher_topk, old)))
            errors.append((student.detach()-behavior).abs().max())
        loss = torch.stack(terms).sum() * int(bool(part))
        loss.backward()
        n = sum(len(x["tokens"]) for x in part)
        self.tokens += n
        return dict(tokens=n, loss_sum=loss.item(), max_logp_error=torch.stack(errors).amax().item() if part else 0.)

    def commit(self, n, durable=True, save_hf=None, scheduler_state=None):
        begin = time.monotonic()
        count = torch.tensor(self.tokens, device=self.device)
        dist.all_reduce(count)
        if n <= 0 or count.item() != n:
            raise RuntimeError("FSDP2 token denominator mismatch")
        norm2 = torch.zeros((), device=self.device)
        probes = []
        params = [p for p in self.model.parameters() if p.requires_grad]
        for p in params:
            g = p.grad.to_local()
            g.div_(n)
            norm2 += g.norm().square()
            w = p.to_local().detach().reshape(-1)
            probes.append(w[::max(1,w.numel()//2048)].to(torch.bfloat16))
        dist.all_reduce(norm2)
        norm = norm2.sqrt().item()
        if not torch.isfinite(norm2):
            raise FloatingPointError("nonfinite FSDP2 gradient")
        for p in params:
            p.grad.to_local().mul_(min(1., self.cfg["clip"]/(norm+1e-6)))
        self.opt.step()
        self.opt.zero_grad(set_to_none=True)
        delta = torch.zeros((), device=self.device)
        for before, p in zip(probes, params, strict=True):
            if not before.numel():
                continue
            w = p.to_local().detach().reshape(-1)
            after = w[::max(1,w.numel()//2048)].to(torch.bfloat16)
            delta = torch.maximum(delta, (after-before).abs().max().float())
        dist.all_reduce(delta, op=dist.ReduceOp.MAX)
        torch.cuda.synchronize(self.device)
        stepped = time.monotonic()
        self.version += 1
        save_hf = durable if save_hf is None else save_hf
        save_s = gather_s = export_s = 0.
        disk_sync = self.cfg.get("weight_sync", "tensor") == "disk"
        if save_hf or durable or disk_sync:
            dest = self.path(self.version) if save_hf or durable else Path(self.cfg["export_dir"])/f"v{self.version}"
            dtype = self.cfg.get("hf_checkpoint_dtype", "bfloat16") if save_hf else "bfloat16"
            save_s, gather_s, export_s = self.save_checkpoint(dest, durable, dtype, scheduler_state)
        self.tokens = 0
        return dict(tokens=n, published=self.version, durable=durable, hf_checkpoint=save_hf,
                    grad_norm=norm, probe_delta=delta.item(), optimizer_and_probe_s=stepped-begin,
                    dcp_write_s=save_s, gather_weights_s=gather_s, hf_export_s=export_s, fsdp_world=self.world)

    def save_checkpoint(self, dest, durable, dtype, scheduler_state=None):
        begin = time.monotonic()
        names = [tempfile.mkdtemp(prefix=f".v{self.version}-", dir=dest.parent) if self.rank==0 else None]
        dist.broadcast_object_list(names, src=0, device=self.device)
        temporary = Path(names[0])
        if durable:
            model_sd, optim_sd = get_state_dict(self.model, self.opt)
            # The overlapping DCP writer retains page-locked copies of an entire
            # shard on this torch release. Pageable staging bounds pinned memory.
            writer = dcp.FileSystemWriter(temporary/"training", per_thread_copy_ahead=0)
            dcp.save({"model":model_sd, "optimizer":optim_sd}, storage_writer=writer)
        saved = time.monotonic()
        full = get_model_state_dict(self.model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        gathered = time.monotonic()
        if self.rank == 0:
            if durable and scheduler_state is not None:
                (temporary / "scheduler.json").write_text(json.dumps(scheduler_state, allow_nan=False))
            inference = {k:(v.to(dtype=getattr(torch, dtype)) if v.is_floating_point() else v) for k,v in full.items()}
            self.model.save_pretrained(temporary, state_dict=inference, max_shard_size="4GB")
            source = Path(self.cfg["staged"][self.cfg["student"]])
            for name in ("tokenizer.json","tokenizer_config.json","special_tokens_map.json","vocab.json",
                         "merges.txt","chat_template.jinja","tokenizer.model"):
                if (source/name).is_file():
                    shutil.copyfile(source/name,temporary/name)
            os.rename(temporary, dest)
        dist.barrier()
        return saved-begin if durable else 0., gathered-saved, time.monotonic()-gathered

    def execute(self, op, payload, version):
        if op in ("reduce_grads", "refresh_parameters"):
            from .replica_sync import reduce_gradients, refresh_parameters
            if self.publisher is None:
                raise RuntimeError("replica exchange requires the trainer NCCL lanes")
            if op == "reduce_grads":
                return reduce_gradients(self, payload["owner"], payload["counts"], version)
            return refresh_parameters(self, payload["owner"], version)
        if op == "activate":
            return self.activate(version, payload.get("optimizer_owner", True))
        if op in ("capture_weights", "begin_weights", "begin_teacher", "weight_bucket", "finish_weights"):
            if self.publisher is None:
                raise RuntimeError("tensor publication is unavailable on this shard")
            if op == "capture_weights":
                if not self.awake or version != self.version or self.tokens:
                    raise RuntimeError("only the committed trainer may capture rollout weights")
                self.publisher.capture(self.model, version)
                torch.cuda.empty_cache()
                return {"scratch_bytes": self.publisher.scratch_bytes}
            if self.awake:
                raise RuntimeError("release the trainer before exposing rollout weight buffers")
            if op == "begin_teacher":
                return self.publisher.begin_teacher(version)
            if op == "begin_weights":
                local = self.cfg.get("node_local_rollout_publication", False)
                if local and self.version != version:
                    raise RuntimeError("rollout publication requires refreshed FSDP master parameters")
                return self.publisher.begin(payload["source"], version, self.model if local else None)
            if op == "weight_bucket":
                return self.publisher.bucket(payload["index"], version, wide_receiver=payload.get("wide_receiver",False))
            return self.publisher.finish(version, collect_ipc=payload.get("wide_receiver",False))
        if op == "evict_trainer" and not self.awake and version == self.version:
            return self.sleep()
        if not self.awake or version != self.version:
            raise RuntimeError("inactive or stale FSDP2 shard")
        if op == "train":
            return self.train(payload["items"])
        if op == "commit":
            return self.commit(payload["tokens"], payload["durable"], payload.get("save_hf"),
                               payload.get("scheduler_state"))
        if op == "sleep":
            return self.sleep(payload.get("retain", False))
        raise ValueError(op)


def serve_shard(cfg, device, rank, world, rendezvous, connection, *, preload_s=0., gate_wait_s=0.):
    try:
        shard = Shard(cfg, device, rank, world, rendezvous)
        properties = torch.cuda.get_device_properties(shard.device)
        attrs = {k:v for k in dir(properties) if not k.startswith("_")
                 and isinstance((v := getattr(properties, k)), (int, float, bool, str))}
        connection.send(dict(ready=True, publish_scratch_bytes=shard.publisher.scratch_bytes if shard.publisher else 0,
                             device_properties=attrs, cpu_preload_s=preload_s, gpu_gate_wait_s=gate_wait_s))
        while (msg := connection.recv()) is not None:
            value = shard.execute(*msg)
            torch.cuda.synchronize()
            connection.send(value)
            kind, _, version = msg
            if (kind == "finish_weights" and type(version) is int and
                    (cfg.get("mirror_optimizer_on_rollout") or cfg.get("mirror_parameters_on_rollout"))
                    and shard.version == version < cfg.get("final_version", cfg["rounds"])):
                shard.mirror_training_state()
        if shard.publisher is not None:
            shard.publisher.close()
        dist.destroy_process_group()
    except BaseException:
        connection.send({"error": traceback.format_exc()})
    finally:
        connection.close()
