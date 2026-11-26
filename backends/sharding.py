"""FSDP2 communication groups and Adam state placement for the shard worker."""
from collections import Counter

import torch
from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard


def shard_causal_lm(model, mesh):
    """Shard decoder blocks bottom-up, keeping shared parameters in one group."""
    # Transformers declares each architecture's intact block types in this
    # metadata. Reading it also supports models whose blocks are not named layers.
    block_types = set(getattr(model, "_no_split_modules", None) or ())
    embeddings = {model.get_input_embeddings(), model.get_output_embeddings()}
    uses = Counter(id(p) for _, p in model.named_parameters(remove_duplicate=False))
    targets = []
    for module in model.modules():
        if module is model or (module not in embeddings and type(module).__name__ not in block_types):
            continue
        local = Counter(id(p) for _, p in module.named_parameters(remove_duplicate=False))
        # A separate group would replace just one owner's reference to a tied
        # weight. Let the enclosing group shard all its aliases together.
        if local and all(count == uses[key] for key, count in local.items()):
            targets.append(module)

    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for module in [*reversed(targets), model]:
        fully_shard(module, mesh=mesh, mp_policy=policy, reshard_after_forward=True)
        # The worker divides once by the token count across the entire mesh.
        module.set_gradient_divide_factor(1.0)
        module.set_force_sum_reduction_for_comms(True)


def restore_adam_state(optimizer, device):
    """Return this worker's fused AdamW moments and step counters to its device."""
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device)
