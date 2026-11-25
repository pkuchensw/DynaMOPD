"""Cross-entropy against the teacher distribution renormalized over its top K tokens."""
from .base import DistillationAlgorithm, TokenBatch


class TopKCrossEntropy(DistillationAlgorithm):
    supervision = "teacher_topk"

    def __init__(self, options):
        super().__init__(options)
        self.teacher_top_k = options.get("k", 8)
        if type(self.teacher_top_k) is not int or self.teacher_top_k < 1:
            raise ValueError("topk_cross_entropy requires a positive integer k")

    def token_loss(self, batch: TokenBatch):
        weights = batch.teacher_topk_logp.detach().softmax(dim=-1)
        return -(weights * batch.student_at_teacher_ids).sum(dim=-1)
