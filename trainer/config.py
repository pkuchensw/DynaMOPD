"""Validate the job configuration and resolve its trainer, policy and executor contracts."""
from copy import deepcopy
import math
import os
from pathlib import Path

from ..algorithms import build_algorithm
from ..cluster.topology import allocations, node_shape
from ..backends.registry import executor_class
from ..scheduler.policy import build_policy


def resolve_model_paths(cfg):
    """Resolve local checkpoint directories on the driver before dispatching work."""
    root = os.environ.get("DYNAMOPD_MODEL_ROOT") or cfg.get("model_root")
    if root and cfg.get("student"):
        directory = Path(root).expanduser().resolve()

        def checkpoint(name):
            return str((directory / Path(name).expanduser()).resolve())

        cfg["model_root"] = str(directory)
        cfg["student"] = checkpoint(cfg["student"])
        cfg["teachers"] = {route: checkpoint(name) for route, name in cfg["teachers"].items()}


def configure_sglang(cfg):
    for flag in ("teacher_rank_images", "parallel_wide_publication", "overlap_wide_startup"):
        if flag in cfg and type(cfg[flag]) is not bool:
            raise ValueError(f"{flag} must be a boolean")
    if type(cfg.get("group_size")) is not int or cfg["group_size"] < 1:
        raise ValueError("group_size must be a positive integer")
    width = cfg.get("inference_tp_size", cfg["group_size"])
    if type(width) is not int or width < 1 or cfg["group_size"] % width:
        raise ValueError("inference_tp_size must divide the local FSDP group size")
    if cfg.get("rollout_sampling_mode", "batch_invariant") not in ("batch_invariant", "request_seed"):
        raise ValueError("unsupported rollout_sampling_mode")
    legacy = any(cfg.get(k) for k in ("cache_teacher_weights", "prime_teacher_weights", "warm_teachers"))
    cfg.setdefault("inference_layout", "replicas" if legacy else "single")
    if cfg.get("frozen_teacher_tensor_sync") and (cfg.get("inference_layout") != "single"
                                                  or cfg.get("weight_sync", "tensor") != "tensor"):
        raise ValueError("frozen_teacher_tensor_sync requires a shared inference engine and tensor publication")
    if cfg.get("qwen2_dynamic_rope") and cfg["inference_layout"] != "single":
        raise ValueError("qwen2_dynamic_rope requires a single inference engine per group")
    if cfg["inference_layout"] not in ("single", "replicas"):
        raise ValueError("inference_layout must be single or replicas")
    if cfg.get("prime_teacher_weights") or cfg.get("cache_teacher_weights"):
        raise ValueError("idle teachers must release GPU memory; disable prime_teacher_weights and cache_teacher_weights")
    count = cfg.get("teachers_per_group", len(cfg["teachers"]))
    if type(count) is not int or count < 1:
        raise ValueError("teachers_per_group must be a positive integer")
    if cfg.get("trainer_attention", "sdpa") not in ("sdpa", "flash_attention_2"):
        raise ValueError("trainer_attention must be sdpa or flash_attention_2")
    if cfg.get("trainer_residency", "cpu") not in ("auto", "cpu"):
        raise ValueError("trainer_residency must be auto or cpu")
    if cfg.setdefault("weight_sync", "tensor") not in ("tensor", "disk"):
        raise ValueError("weight_sync must be tensor or disk")
    bucket_mib = cfg.setdefault("weight_bucket_mib", 2048)
    if type(bucket_mib) is not int or bucket_mib < 1:
        raise ValueError("weight_bucket_mib must be a positive integer")
    contents = cfg.setdefault("checkpoint_contents", ["hf_model", "training"])
    if not isinstance(contents, list) or sorted(contents) not in (["hf_model"], ["hf_model", "training"]):
        raise ValueError("checkpoint_contents must include hf_model and optionally training")
    if cfg.get("hf_checkpoint_dtype", "bfloat16") not in ("bfloat16", "float32"):
        raise ValueError("hf_checkpoint_dtype must be bfloat16 or float32")



def build_job(config, *, simulate, replay, resume_version):
    """Resolve defaults on a private config before persistent state or workers are created."""
    cfg = deepcopy(config)
    resolve_model_paths(cfg)
    sizes = ("rounds", "groups", "batch_size", "quantum", "score_chunk", "microbatch", "decode_batch", "score_batch")
    if any(type(cfg.get(k)) is not int or cfg[k] < 1 for k in sizes):
        raise ValueError("round and batch limits must be positive")
    if (cfg["batch_size"] < cfg["groups"]
            or not math.isfinite(cfg["temperature"]) or cfg["temperature"] <= 0):
        raise ValueError("each group requires a request and positive temperature")
    if not cfg["teachers"] or set(cfg["teachers"]) & {"cold", "train", "rollout", "standby"}:
        raise ValueError("configure nonempty teachers with unique non-reserved role names")
    cfg.setdefault("engine", "sglang_fsdp2")
    cfg.setdefault("algorithm", "on_policy_k1")
    cfg.setdefault("train_on_length", cfg["algorithm"] == "on_policy_k1")
    if type(cfg["train_on_length"]) is not bool:
        raise ValueError("train_on_length must be a boolean")
    cfg.setdefault("policy", "merge")
    if cfg.get("length_forecast", "conditional_median") not in {"conditional_quantiles", "conditional_median"}:
        raise ValueError("unknown length_forecast")
    alpha = cfg.setdefault("max_length_ema_alpha", 0.2)
    if alpha is not None and (type(alpha) not in (int, float) or not 0 < alpha <= 1):
        raise ValueError("max_length_ema_alpha must be in (0, 1] or null to disable")
    if type(cfg.get("phase_barriers", False)) is not bool:
        raise ValueError("phase_barriers must be a boolean")
    expansion = cfg.get("rollout_tp_expansion")
    if cfg["policy"] == "elastic_tp" and not (expansion or (simulate and cfg.get("sim_tp"))):
        raise ValueError("elastic_tp requires a TP cost profile for the selected executor")
    cluster = cfg.setdefault("cluster", {"transport": "local"})
    shape = node_shape(cluster)
    if shape:
        allocations(cfg)
        if not simulate and cluster.get("transport", "local") == "local":
            raise ValueError("explicit node allocation requires Ray transport")
    if expansion and not simulate:
        if cfg["engine"] != "sglang_fsdp2" or cfg.get("weight_sync", "tensor") != "tensor":
            raise ValueError("TP expansion requires SGLang with tensor weight publication")
        if cluster.get("transport") == "ray" and not shape:
            raise ValueError("Ray TP expansion needs cluster.nodes or node_count and groups_per_node")
    if cfg["engine"] == "sglang_fsdp2":
        configure_sglang(cfg)
    elif "checkpoint_contents" in cfg and "training" not in cfg["checkpoint_contents"]:
        raise ValueError("HF-only checkpoints currently require the SGLang/FSDP2 backend")
    executor = executor_class(cfg, simulate)
    replicated = (cfg["engine"] == "sglang_fsdp2" and cfg.get("weight_sync") == "tensor"
                  and (simulate or getattr(executor, "supports_trainer_replicas", False)))
    # Small jobs have too few independent microbatches to amortize another replica.
    default_groups = min(cfg["groups"], max(1, cfg["batch_size"] // (2 * max(1, cfg["microbatch"]))))
    limit = cfg.setdefault("max_train_groups", default_groups if replicated else 1)
    if type(limit) is not int or not 1 <= limit <= cfg["groups"] or (limit > 1 and not replicated):
        raise ValueError("max_train_groups must fit the allocation and the trainer replication backend")
    if type(cfg.get("replica_bucket_mib", 64)) is not int or cfg.get("replica_bucket_mib", 64) < 1:
        raise ValueError("replica_bucket_mib must be a positive integer")
    algorithm = build_algorithm(cfg)
    policy = build_policy(cfg["policy"], cfg.get("policy_options", {}))
    if algorithm.supervision not in executor.supervision:
        raise ValueError(f"{cfg['algorithm']} requests supervision {algorithm.supervision!r} unavailable on {executor.__name__}")
    cfg["teacher_top_k"] = algorithm.teacher_top_k
    if replay and not simulate:
        raise ValueError("replay requires simulation")
    if type(resume_version) is not int or resume_version < 0 or (resume_version and not executor.supports_resume):
        raise ValueError("executor does not support the requested resume version")
    interval = cfg.get("checkpoint_interval", 1)
    if type(interval) is not int or interval < 0 or (interval != 1 and not executor.supports_periodic_checkpoint):
        raise ValueError("unsupported checkpoint interval")
    return cfg, executor, policy, algorithm
