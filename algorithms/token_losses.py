"""Sampled log probabilities and policy losses for response tokens."""
from functools import cache

import torch
import torch.nn.functional as F


@cache
def _flash_cross_entropy():
    from flash_attn.losses.cross_entropy import CrossEntropyLoss

    return CrossEntropyLoss(reduction="none", inplace_backward=False)


def sampled_logprob(logits, token_ids, *, fused=False):
    """Log probability of each sampled ID given its row of vocabulary logits."""
    if fused and logits.is_cuda:
        return -_flash_cross_entropy()(logits, token_ids)
    if logits.dtype in (torch.float16, torch.bfloat16):
        logits = logits.float()
    return -F.cross_entropy(logits, token_ids, reduction="none")


def clipped_k1(student_logp, old_logp, teacher_logp):
    """Unreduced negative dual-clipped PPO gain for sampled K1 distillation."""
    advantage = (teacher_logp - student_logp).detach()
    log_ratio = student_logp - old_logp.detach()
    ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
    bounded_ratio = torch.clamp(ratio, min=0.8, max=1.2)
    gain = torch.minimum(advantage * ratio, advantage * bounded_ratio)
    return -torch.where(advantage < 0, torch.maximum(gain, 3.0 * advantage), gain)
