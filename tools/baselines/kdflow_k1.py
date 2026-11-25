"""KDFlow algorithm plugin for the native verl single-update K1 experiment."""
import torch

from kdflow.algorithms import register_algorithm
from kdflow.algorithms.vanilla_kd import VanillaKD
from profile_trace import traced
from DynaMOPD.algorithms.verl_k1 import policy_loss
from DynaMOPD.utils.torch_functional import logprobs_from_logits_flash_attn


@register_algorithm("verl_k1")
class VerlK1(VanillaKD):
    def __init__(self, strategy, student_model, teacher_lm_head, **kwargs):
        self.strategy = strategy
        self.args = strategy.args
        self.student = student_model
        self.teacher_lm_head = teacher_lm_head
        if self.args.kd.kd_ratio != 1 or self.args.kd.kd_loss_fn != "verl_k1":
            raise ValueError("the matched job requires pure sampled K1")

    @traced("training_microbatch")
    def training_step(self, micro_batch):
        ids = micro_batch["stu_input_ids"]
        mask = micro_batch["stu_loss_mask"].bool()
        if (not torch.equal(ids, micro_batch["tea_input_ids"])
                or not torch.equal(mask, micro_batch["tea_loss_mask"].bool())):
            raise ValueError("teacher and actor must see identical sampled token positions")
        labels = ids.roll(-1, 1)[mask]
        head = self.student.model.lm_head
        chunk = self.args.train.chunked_loss_size or max(1, len(labels))

        def forward():
            output = self.student(ids, attention_mask=micro_batch["stu_attn_mask"], allgather_logits=True,
                                  ring_attn_group=self.strategy.ring_attn_group,
                                  **(micro_batch.get("stu_multi_modal_inputs") or {}))
            return output["hidden_states"][-1][mask]

        with torch.no_grad():
            previous = forward()
            old = [logprobs_from_logits_flash_attn(head(previous[i:i+chunk], skip=False),
                                                   labels[i:i+chunk], inplace_backward=False)
                   for i in range(0, len(labels), chunk)]
            del previous
        current = forward()
        loss = current.new_zeros((), dtype=torch.float32)
        for i, previous_lp in zip(range(0, len(labels), chunk), old, strict=True):
            with torch.no_grad():
                teacher_logits = self.compute_multi_teacher_logits(
                    micro_batch["teacher_hiddens"], mask, micro_batch["teacher_routing_key"], i, i+chunk)
                target = logprobs_from_logits_flash_attn(teacher_logits, labels[i:i+chunk], inplace_backward=False)
            student_lp = logprobs_from_logits_flash_attn(head(current[i:i+chunk], skip=False),
                                                         labels[i:i+chunk], inplace_backward=False)
            loss = loss + policy_loss(student_lp, previous_lp, target).sum()
        loss = loss / micro_batch["avg_micro_batch_token_num"]
        return {"train/loss": loss, "train/kd_loss": loss.detach()}
