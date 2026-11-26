"""Resource capacities used by the distillation benchmark."""
from DynaMOPD.cluster.topology import node_shape


def benchmark_nodes(cfg):
    shape = node_shape(cfg.get("cluster", {}))
    if shape and any(groups * cfg["group_size"] != len(cfg["benchmark_devices"]) for _, groups in shape):
        raise ValueError("every benchmark node must supply the declared per-node GPU layout")
    return len(shape) or 1


def layout(cfg, system):
    devices = cfg["benchmark_devices"]
    nodes = benchmark_nodes(cfg)
    total = len(devices) * nodes
    tp, shard = cfg["inference_tp_size"], cfg["group_size"]
    if not devices or len(set(devices)) != len(devices) or min(devices) < 0:
        raise ValueError("benchmark_devices must list distinct physical GPUs")
    if min(tp, shard) < 1 or shard % tp or total != cfg["groups"] * shard:
        raise ValueError("inference TP must divide the FSDP group and groups must cover the GPUs")
    if cfg["max_train_groups"] != cfg["groups"]:
        raise ValueError("the matched benchmark needs all FSDP groups available for training")
    if cfg["microbatch"] % shard or cfg["train_budget"] % shard:
        raise ValueError("group microbatch and token budget must divide evenly across its FSDP ranks")
    actor = total
    teacher = total
    if system == "verl":
        teacher = cfg["verl_teacher_gpus"]
        actor = total - teacher
        if actor < 1 or teacher < len(cfg["teachers"]) * tp:
            raise ValueError("verl requires an actor pool and at least one TP replica per routed teacher")
        if nodes > 1 and teacher % len(devices):
            raise ValueError("multi-node verl benchmark assigns whole nodes to each role pool")
    if actor % shard or teacher % tp:
        raise ValueError("each role's GPU pool must contain complete parallel groups")
    return dict(devices=devices, rollout_tp=tp, rollout_replicas=actor // tp,
                teacher_tp=tp, teacher_replicas=teacher // tp, fsdp_shard_size=shard,
                trainer_replicas=actor // shard, separate_teacher_pool=system == "verl",
                microbatch_per_gpu=cfg["microbatch"] // shard, token_budget_per_gpu=cfg["train_budget"] // shard)


def check_layouts(cfg, systems):
    layouts = {name: layout(cfg, name) for name in systems}
    reference = next(iter(layouts.values()))
    differing = {key: {name: spec[key] for name, spec in layouts.items()}
                 for key in reference if any(spec[key] != reference[key] for spec in layouts.values())}
    if differing:
        raise ValueError(f"GPU placements differ across the requested benchmark systems: {differing}. "
                         "The native verl launcher reserves a separate teacher pool. "
                         "Run that topology as a standalone reference.")
    return reference


def check_command(cfg, system, command):
    """Check the final upstream arguments, including options inherited from the native launchers."""
    spec = layout(cfg, system)
    if system == "dynamopd":
        import json
        from pathlib import Path
        snapshot = Path(command[command.index("--config")+1])
        if json.loads(snapshot.read_text()) != cfg:
            raise ValueError("DynaMOPD launch configuration differs from the benchmark")
        return
    if system == "kdflow":
        flags = iter(command[3:])
        actual = {name.removeprefix("--"): value for name, value in zip(flags, flags, strict=True)}
        expected = dict(num_nodes=benchmark_nodes(cfg), num_gpus_per_node=len(spec["devices"]), backend="fsdp2",
                        rollout_tp_size=spec["rollout_tp"], rollout_num_engines=spec["rollout_replicas"],
                        teacher_tp_size=spec["teacher_tp"], teacher_dp_size=spec["teacher_replicas"],
                        teacher_pp_size=1, fsdp_size=spec["fsdp_shard_size"], micro_train_batch_size=spec["microbatch_per_gpu"],
                        max_token_len_per_gpu=spec["token_budget_per_gpu"], train_batch_size=cfg["batch_size"],
                        rollout_batch_size=cfg["batch_size"], generate_max_len=cfg["max_new_tokens"])
    else:
        actual = dict(arg.lstrip("+").split("=", 1) for arg in command[3:] if "=" in arg)
        actor = spec["rollout_tp"] * spec["rollout_replicas"]
        teacher = spec["teacher_tp"] * spec["teacher_replicas"]
        width = len(spec["devices"])
        expected = {"trainer.nnodes": max(1, actor // width), "distillation.nnodes": max(1, teacher // width),
                    "trainer.n_gpus_per_node": min(width, actor), "distillation.n_gpus_per_node": min(width, teacher),
                    "actor_rollout_ref.rollout.tensor_model_parallel_size": spec["rollout_tp"],
                    "actor_rollout_ref.actor.fsdp_config.fsdp_size": spec["fsdp_shard_size"],
                    "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": spec["microbatch_per_gpu"],
                    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": spec["token_budget_per_gpu"],
                    "data.train_batch_size": cfg["batch_size"], "data.max_response_length": cfg["max_new_tokens"]}
        for key in cfg["teachers"]:
            expected[f"distillation.teacher_models.{key}.inference.tensor_model_parallel_size"] = spec["teacher_tp"]
        if sum(int(actual[f"distillation.teacher_models.{k}.num_replicas"]) for k in cfg["teachers"]) != spec["teacher_replicas"]:
            raise ValueError("verl teacher replica counts do not cover its GPU pool")
    mismatch = {k: (v, actual.get(k)) for k,v in expected.items() if str(v) != actual.get(k)}
    if mismatch:
        raise ValueError(f"{system} launch arguments differ from the shared configuration: {mismatch}")
