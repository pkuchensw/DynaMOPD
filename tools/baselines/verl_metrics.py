"""Instrument the matched native verl actor update without modifying its checkout."""
import hashlib
from pathlib import Path


def install(repo, work):
    # Overlay only the trainer module. The rest of verl (including its Hydra
    # configs) remains the exact checkout selected for this benchmark.
    relative = Path("verl/trainer/ppo/ray_trainer.py")
    original = repo / relative
    body = original.read_text(encoding="utf-8")
    anchor = '        actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})'
    if body.count(anchor) != 1 or 'self.actor_rollout_wg.update_actor(batch_td)' not in body:
        raise ValueError("unsupported verl actor update for response-mask measurement")
    body = body.replace(anchor,
        '        loss_mask = batch_td["loss_mask"]\n'
        '        mask_values = loss_mask.values() if loss_mask.is_nested else loss_mask\n'
        '        actor_output["actor/effective_training_tokens"] = [int(mask_values.sum().item())]\n' + anchor)
    # Preserve the same per-row key in the generation request and the actor's
    # reordered DataProto so the rollout dump can certify (prompt, teacher) pairs.
    for original, replacement in (
        ('set({"data_source", "reward_model", "extra_info", "uid"})',
         'set({"data_source", "reward_model", "extra_info", "uid", self.config.distillation.teacher_key})'),
        ('            if "request_id" in batch.non_tensor_batch:',
         '            reward_extra_infos_to_dump["teacher_routing_key"] = list(batch.non_tensor_batch[self.config.distillation.teacher_key])\n'
         '            if "request_id" in batch.non_tensor_batch:'),
    ):
        if body.count(original) != 1:
            raise ValueError("unsupported verl rollout dump for teacher-route verification")
        body = body.replace(original, replacement)
    source, target = repo, work
    for part in relative.parts:
        target.mkdir(parents=True, exist_ok=True)
        if target != work:
            for sibling in source.iterdir():
                if sibling.name not in (part, "__pycache__"):
                    (target / sibling.name).symlink_to(sibling.resolve(), target_is_directory=sibling.is_dir())
        source, target = source / part, target / part
    target.write_text(body, encoding="utf-8")
    return hashlib.sha256(target.read_bytes()).hexdigest()
