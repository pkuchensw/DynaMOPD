"""Publish FSDP2 parameters over NCCL and same-device SGLang CUDA IPC."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor


def bucket_plan(state, limit):
    from sglang.srt.weight_sync.tensor_bucket import FlattenedTensorMetadata

    buckets, entries, offset = [], [], 0
    for name, tensor in state.items():
        dtype = torch.bfloat16 if tensor.is_floating_point() else tensor.dtype
        size = tensor.numel() * dtype.itemsize
        if entries and (offset + size > limit or dtype != entries[-1].dtype):
            buckets.append(entries)
            entries, offset = [], 0
        # SGLang's flattened bucket metadata stores byte offsets and byte counts.
        entries.append(FlattenedTensorMetadata(name, tensor.shape, dtype, offset, offset+size, size))
        offset += size
    if entries:
        buckets.append(entries)
    if not buckets:
        raise ValueError("empty inference state")
    return buckets


class TensorPublisher:
    """One endpoint per persistent FSDP process, on the physical GPU of its rollout TP rank."""

    def __init__(self, cfg, rank, device, model):
        self.cfg, self.rank, self.device = cfg, rank, device
        self.group = cfg["worker_group"]
        self.buckets = bucket_plan(model.state_dict(), cfg.get("weight_bucket_mib", 2048) * 2**20)
        # A flat IPC bucket, the gathered parameter and collective workspace.
        self.scratch_bytes = max(b[-1].end_idx for b in self.buckets) + 2 * max(
            entry.numel for bucket in self.buckets for entry in bucket
        )
        schema = [[(m.name, list(m.shape), str(m.dtype)) for m in bucket] for bucket in self.buckets]
        self.digest = hashlib.sha256(json.dumps(schema).encode()).hexdigest()
        self.lane = self.snapshot = self.buffer = self.active_version = None
        self.snapshot_version = None
        self.allocator_settings = None
        self.cursor = 0
        self.frozen = {}
        self.local_state = None
        self.local_parameter_names = {name for name, p in model.named_parameters(remove_duplicate=False)
                                      if p.requires_grad}
        self.cross_node_payload_bytes = 0

    @torch.no_grad()
    def capture(self, model, version):
        if self.snapshot is not None or self.active_version is not None:
            raise RuntimeError("unacknowledged rollout snapshot")
        # Retain only this rank's inference-precision shard so that the trainer
        # can offload the FP32 parameters and Adam state before SGLang wakes up.
        self.snapshot = {k: t.detach().to(dtype=torch.bfloat16 if t.is_floating_point() else t.dtype, copy=True)
                         for k,t in model.state_dict().items()}
        self.snapshot_version = version

    def prepare(self):
        from sglang.srt.utils.patch_torch import monkey_patch_torch_reductions

        monkey_patch_torch_reductions()
        # Connect at executor startup, when all ranks are available. NCCL creates
        # communicators lazily so issue a real collective here as well.
        name = f"dynamopd-{self.cfg['weight_sync_run_id']}-{self.rank}"
        size = self.cfg["groups"]
        timeout = datetime.timedelta(seconds=self.cfg.get("action_timeout_s", 1800))
        endpoint = self.cfg.get("weight_sync_endpoint")
        if endpoint:
            host, port = endpoint
            tcp = dist.TCPStore(host, port, None, False, timeout)
            self.store = dist.PrefixStore(name, tcp)
        else:
            self.store = dist.FileStore(str(Path(self.cfg["state_dir"])/f".{name}"), size)
        self.store.set_timeout(timeout)
        options = dist.ProcessGroupNCCL.Options()
        options._timeout = timeout
        self.lane = dist.ProcessGroupNCCL(self.store, self.group, size, options)
        probe = torch.ones(1, device=self.device)
        self.lane.allreduce([probe]).wait()
        if probe.item() != size:
            raise RuntimeError("incomplete NCCL publication group")

    def prepare_ipc(self):
        # Match verl's IPC allocator mode. Restore explicit training allocator
        # settings after SGLang acknowledges the last bucket.
        settings = os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or os.environ.get("PYTORCH_ALLOC_CONF", "")
        ipc_settings = re.sub(r"\bexpandable_segments\s*:\s*True\b", "expandable_segments:False", settings)
        if ipc_settings != settings:
            torch.cuda.memory._set_allocator_settings(ipc_settings)
            self.allocator_settings = settings

    def begin_teacher(self, teacher):
        from safetensors import safe_open
        if self.active_version is not None or self.snapshot is not None:
            raise RuntimeError("unfinished weight publication")
        if teacher not in self.frozen:
            tensors = {}
            folder = self.cfg["staged"][self.cfg["teachers"][teacher]]
            for file in sorted(Path(folder).glob("*.safetensors")):
                with safe_open(file, framework="pt", device="cpu") as reader:
                    tensors.update((key, reader.get_tensor(key)) for key in reader.keys())
            missing = {m.name for bucket in self.buckets for m in bucket} - tensors.keys()
            if missing:
                # HF safetensors can store just one member of a tied group. Use the
                # teacher's architecture and config, even when the saved alias is reversed.
                from transformers import AutoConfig, AutoModelForCausalLM
                config = AutoConfig.from_pretrained(folder, local_files_only=True)
                with torch.device("meta"):
                    teacher_model = AutoModelForCausalLM.from_config(config)
                aliases = {}
                for name, param in teacher_model.named_parameters(remove_duplicate=False):
                    aliases.setdefault(id(param), []).append(name)
                for names in aliases.values():
                    saved = next((name for name in names if name in tensors), None)
                    if saved is not None:
                        for name in missing.intersection(names):
                            tensors[name] = tensors[saved]
            for bucket in self.buckets:
                for m in bucket:
                    if m.name not in tensors or tensors[m.name].shape != m.shape:
                        raise ValueError(f"frozen teacher has incompatible tensor {m.name}")
            self.frozen[teacher] = tensors
        self.prepare_ipc()
        self.active_version, self.cursor, self.source = teacher, 0, None
        return dict(buckets=len(self.buckets), schema=self.digest)

    def begin(self, source, version, local_model=None):
        if self.active_version is not None or (self.group == source and self.snapshot_version != version):
            raise RuntimeError("tensor publication has a stale source or an outstanding update")
        self.prepare_ipc()
        # A fixed-size collective verifies the agreement even if versions differ.
        # All replicas agree iff the maximum and minimum match for every byte.
        digest = hashlib.sha256(f"{version}/{source}/{self.digest}".encode()).digest()
        values = torch.tensor(list(digest), dtype=torch.int32, device=self.device)
        bounds = torch.stack((values, -values))
        options = dist.AllreduceOptions()
        options.reduceOp = dist.ReduceOp.MAX
        self.lane.allreduce([bounds], options).wait()
        if not torch.equal(bounds[0], -bounds[1]):
            raise RuntimeError("replicas disagree on weight schema, source or version")
        self.source, self.active_version, self.cursor = source, version, 0
        self.local_state = local_model.state_dict() if local_model is not None else None
        self.cross_node_payload_bytes = 0
        return dict(buckets=len(self.buckets), schema=self.digest,
                    weight_bytes=sum(b[-1].end_idx for b in self.buckets))

    @torch.no_grad()
    def bucket(self, index, version, *, wide_receiver=False):
        from sglang.srt.utils import MultiprocessingSerializer

        if version != self.active_version or index != self.cursor or index >= len(self.buckets):
            raise RuntimeError("out-of-order weight bucket")
        # The owner calls this method only after SGLang acknowledges the previous
        # bucket. Keep the storage alive across the HTTP call that consumes its IPC handle.
        self.buffer = None
        entries = self.buckets[index]
        flat = torch.empty(entries[-1].end_idx, dtype=torch.uint8, device=self.device)
        # Every replicated trainer already acknowledged the exact updated FP32
        # shards. Reconstruct its rollout bucket within the node-local FSDP mesh.
        # Buffers and frozen parameters still use the authoritative owner image.
        local = self.local_state is not None and all(m.name in self.local_parameter_names for m in entries)
        if self.source is None:
            for item in entries:
                target = flat[item.start_idx:item.end_idx].view(item.dtype).view(item.shape)
                target.copy_(self.frozen[version][item.name], non_blocking=True)
        elif local or self.group == self.source:
            for item in entries:
                if self.group == self.source:
                    shard = self.snapshot.pop(item.name)
                else:
                    shard = self.local_state[item.name].to(device=self.device, dtype=item.dtype)
                full = shard.full_tensor() if isinstance(shard, DTensor) else shard
                flat[item.start_idx:item.end_idx].view(item.dtype).view(item.shape).copy_(full)
                del full, shard
        if self.source is not None and not local:
            options = dist.BroadcastOptions()
            options.rootRank = self.source
            options.rootTensor = 0
            self.lane.broadcast([flat], options).wait()
            self.cross_node_payload_bytes += flat.numel()
        torch.cuda.synchronize(self.device)
        self.buffer = flat
        self.cursor += 1
        payload = {"flattened_tensor": flat, "metadata": entries}
        result = {"handle": MultiprocessingSerializer.serialize(payload, output_str=True)}
        if wide_receiver:
            # Re-serialize the same storage for each recipient. Each import owns
            # one producer-side CUDA IPC reference counter and releases it once.
            result["wide_handle"] = MultiprocessingSerializer.serialize(payload, output_str=True)
            if result["handle"] == result["wide_handle"]:
                raise RuntimeError("two SGLang consumers received the same CUDA IPC reference")
        return result

    def finish(self, version, *, collect_ipc=False):
        if version != self.active_version or self.cursor != len(self.buckets) or self.snapshot:
            raise RuntimeError("incomplete tensor publication")
        self.buffer = self.snapshot = self.snapshot_version = self.active_version = None
        self.local_state = None
        stats = {"rollout_lane_payload_bytes": self.cross_node_payload_bytes} if isinstance(version, int) else {}
        if collect_ipc:
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
            stats["publisher_allocated_bytes"] = torch.cuda.memory_allocated(self.device)
            stats["publisher_reserved_bytes"] = torch.cuda.memory_reserved(self.device)
        if self.allocator_settings is not None:
            torch.cuda.memory._set_allocator_settings(self.allocator_settings)
            self.allocator_settings = None
        return stats

    def close(self):
        if self.lane is not None:
            self.lane.shutdown()
