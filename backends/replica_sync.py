"""Sum corresponding FSDP shards across dynamic trainers; refresh their FP32 parameters."""
import torch
import torch.distributed as dist


def shard_buckets(parameters, capacity):
    """Describe matching flat slices on the same FSDP rank of every replica."""
    pieces, used = [], 0
    for index, param in enumerate(parameters):
        local = param.to_local()
        if local.dtype != torch.float32:
            raise ValueError("replica exchange requires FP32 FSDP master shards")
        start = 0
        while start < local.numel():
            count = min(local.numel() - start, capacity - used)
            pieces.append((index, start, used, count))
            start += count
            used += count
            if used == capacity:
                yield pieces, used
                pieces, used = [], 0
    if pieces:
        yield pieces, used


@torch.no_grad()
def exchange_shards(shard, owner, *, gradients, contributes):
    """One bounded NCCL buffer per rank. Inference-only ranks can keep the model on CPU."""
    pub = shard.publisher
    is_owner = pub.group == owner
    params = [p for p in shard.model.parameters() if p.requires_grad]
    if gradients and is_owner and not contributes:
        # A zero-contribution optimizer has no grad tensors, including on
        # ranks whose local slice of a particular parameter is empty.
        for param in params:
            param.grad = torch.empty_like(param)
    capacity = min(sum(p.to_local().numel() for p in params),
                   shard.cfg.get("replica_bucket_mib", 64) * 2**20 // 4)
    scratch = torch.empty(capacity, device=shard.device, dtype=torch.float32)
    if gradients:
        options = dist.ReduceOptions()
        options.reduceOp = dist.ReduceOp.SUM
    else:
        options = dist.BroadcastOptions()
    options.rootRank = owner
    options.rootTensor = 0
    size = 0
    for pieces, used in shard_buckets(params, capacity):
        buf = scratch[:used]
        if contributes:
            for index, start, offset, count in pieces:
                value = params[index].grad if gradients else params[index]
                buf.narrow(0, offset, count).copy_(value.to_local().view(-1).narrow(0, start, count))
        elif gradients:
            buf.zero_()
        if gradients:
            pub.lane.reduce([buf], options).wait()
        else:
            pub.lane.broadcast([buf], options).wait()
        if (gradients and is_owner) or (not gradients and not is_owner):
            for index, start, offset, count in pieces:
                dest = params[index].grad if gradients else params[index]
                dest.to_local().view(-1).narrow(0, start, count).copy_(buf.narrow(0, offset, count))
        size += buf.numel() * buf.element_size()
    return size


@torch.no_grad()
def reduce_gradients(shard, owner, counts, version):
    pub = shard.publisher
    expected = counts[pub.group]
    local = torch.tensor(shard.tokens, device=shard.device, dtype=torch.int64)
    group_total = local.clone()
    dist.all_reduce(group_total)
    needs_trainer = bool(expected) or pub.group == owner
    valid = group_total.item() == expected and (not needs_trainer or (shard.awake and shard.version == version))
    # Exchange the actual rank counts and reject a bad local receipt on all
    # endpoints before any rank enters a parameter-sized collective.
    valid = valid and all((p.grad is not None) == bool(expected)
                          for p in shard.model.parameters() if p.requires_grad)
    status = torch.stack((local, torch.full_like(local, int(not valid))))
    pub.lane.allreduce([status]).wait()
    dist.all_reduce(status[1:])
    if status[1].item():
        raise RuntimeError("training replicas disagree on version, gradients or effective token counts")
    size = exchange_shards(shard, owner, gradients=True, contributes=bool(expected))
    if pub.group == owner:
        # commit() subsequently sums across ranks of the owner's FSDP mesh.
        shard.tokens = int(status[0].item())
    else:
        shard.opt.zero_grad(set_to_none=True)
        shard.tokens = 0
    return dict(owner=owner, group_tokens=expected, global_tokens=sum(counts), gradient_shard_bytes=size)


@torch.no_grad()
def refresh_parameters(shard, owner, version):
    source = shard.publisher.group == owner
    invalid = bool(shard.tokens or (source and (not shard.awake or shard.version != version))
                   or (not source and shard.version >= version))
    status = torch.tensor(int(invalid), device=shard.device)
    shard.publisher.lane.allreduce([status]).wait()
    dist.all_reduce(status)
    if status.item():
        raise RuntimeError("unexpected FSDP version during replica publication")
    size = exchange_shards(shard, owner, gradients=False, contributes=source)
    shard.version = version
    return dict(master_version=version, master_shard_bytes=size)
