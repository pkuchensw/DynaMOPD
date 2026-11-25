"""Objective contract shared by the driver, reference worker and FSDP2 ranks."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..components import resolve

if TYPE_CHECKING:
    from torch import Tensor


@dataclass(frozen=True)
class TokenBatch:
    """Aligned response positions for one trajectory. Only student_logp carries a gradient."""

    student_logp: Tensor
    behavior_logp: Tensor
    teacher_logp: Tensor
    sample: dict
    student_at_teacher_ids: Tensor | None = None
    teacher_topk_logp: Tensor | None = None
    old_logp: Tensor | None = None


class DistillationAlgorithm(ABC):
    # Feature negotiation happens before allocating GPUs.
    supervision = "sampled_token_logprob"
    teacher_top_k = 0
    recompute_old = False

    def __init__(self, options: dict):
        self.options = options

    def route(self, example: dict, config: dict) -> str:
        if "route_field" in config:
            key = example[config["route_field"]]
        else:
            key = example.get("teacher", example.get("teacher_routing_key", example.get("data_source")))
        return config.get("routes", {}).get(key, key)

    @abstractmethod
    def token_loss(self, batch: TokenBatch) -> Tensor:
        """Return one unnormalized differentiable loss per sampled response position."""

    def loss_sum(self, batch: TokenBatch) -> Tensor:
        terms = self.token_loss(batch)
        if terms.shape != batch.student_logp.shape:
            raise ValueError("algorithm must return one loss per response token")
        return terms.sum()


OBJECTIVES = {
    "on_policy_k1": "DynaMOPD.algorithms.sampled:OnPolicyK1",
    "sampled_reverse_kl": "DynaMOPD.algorithms.sampled:SampledReverseKL",
    "sampled_forward_kl": "DynaMOPD.algorithms.sampled:SampledForwardKL",
    "topk_cross_entropy": "DynaMOPD.algorithms.topk:TopKCrossEntropy",
}


def build_algorithm(config: dict) -> DistillationAlgorithm:
    name = config.get("algorithm", "on_policy_k1")
    cls = resolve(name, OBJECTIVES)
    if not issubclass(cls, DistillationAlgorithm):
        raise TypeError(f"{name} must implement DistillationAlgorithm")
    return cls(dict(config.get("algorithm_options", {})))
