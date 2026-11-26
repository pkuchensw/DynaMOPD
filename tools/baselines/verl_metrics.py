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
