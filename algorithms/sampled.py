"""OPD estimators over response tokens sampled from the frozen rollout policy."""
import math

from .base import DistillationAlgorithm, TokenBatch


class SampledReverseKL(DistillationAlgorithm):
    def token_loss(self, batch: TokenBatch):
        advantage = (batch.behavior_logp - batch.teacher_logp).detach()
        return batch.student_logp * advantage


class OnPolicyK1(DistillationAlgorithm):
    """verl K1 policy gradient for one optimizer step at the frozen round policy."""

    recompute_old = True

    def token_loss(self, batch: TokenBatch):
        from .verl_k1 import policy_loss

        if batch.old_logp is None:
            raise ValueError("K1 requires log probabilities recomputed by the frozen actor")
        return policy_loss(batch.student_logp, batch.old_logp, batch.teacher_logp)


class SampledForwardKL(DistillationAlgorithm):
    """Teacher cross-entropy gradient at on-policy prefixes, using importance sampling."""

    def __init__(self, options):
        super().__init__(options)
        self.max_weight = options.get("max_importance_weight")
        if self.max_weight is not None and (not math.isfinite(self.max_weight) or self.max_weight <= 0):
            raise ValueError("max_importance_weight must be finite and positive")

    def token_loss(self, batch: TokenBatch):
        log_weight = (batch.teacher_logp - batch.behavior_logp).detach()
        if self.max_weight is not None:
            log_weight = log_weight.clamp(max=math.log(self.max_weight))
        return -batch.student_logp * log_weight.exp()
