"""Sampled K1 and the vanilla clipped policy loss used by the matched verl job."""
import torch


def policy_loss(current, old, teacher):
    advantage = (teacher - current).detach()
    ratio = (current - old.detach()).clamp(-20., 20.).exp()
    raw = -advantage * ratio
    clipped = torch.maximum(raw, -advantage * ratio.clamp(.8, 1.2))
    dual_clipped = torch.minimum(-advantage * 3., clipped)
    return torch.where(advantage < 0, dual_clipped, clipped)
