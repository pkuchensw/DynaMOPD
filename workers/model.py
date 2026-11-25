"""Stage immutable model files and identify compatible inference architectures."""
import errno
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4


def stage_model(path, folder):
    source = Path(path).resolve()
    files = sorted(f for f in source.iterdir() if f.is_file() and f.suffix in
                   (".safetensors", ".json", ".jinja", ".txt", ".model"))
    if not any(f.suffix==".safetensors" for f in files):
        raise ValueError(f"no safetensors checkpoint under {source}")
    digest = hashlib.sha256(str(source).encode())
    for f in files:
        stat = f.stat()
        digest.update(f"{f.name}/{stat.st_size}/{stat.st_mtime_ns}".encode())
    dest = folder/(source.name+"-"+digest.hexdigest()[:16])
    if not dest.exists():
        # A sibling staging task or process may publish this same image first.
        # Only a fully copied directory is visible at the content-keyed path.
        with tempfile.TemporaryDirectory(dir=folder) as scratch:
            candidate = Path(scratch) / "model"
            candidate.mkdir()
            for f in files:
                shutil.copyfile(f, candidate / f.name)
            try:
                candidate.rename(dest)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not dest.is_dir():
                    raise
    return str(dest)


def stage_models(cfg, pool):
    # Shared by all worker processes in this invocation, including Ray actors.
    cfg["weight_sync_run_id"] = uuid4().hex
    folder = Path(cfg["state_dir"])/"initial_models"
    folder.mkdir(exist_ok=True)
    paths = list(dict.fromkeys([cfg["student"], *cfg["teachers"].values()]))
    sources = {path: Path(path).resolve() for path in paths}
    unique = list(dict.fromkeys(sources.values()))
    staged = dict(zip(unique, pool.map(lambda path: stage_model(path, folder), unique), strict=True))
    return {path: staged[source] for path, source in sources.items()}


def inference_signature(path, context_limit, dynamic_rope=False):
    """Only alias weights when the active inference model computes the same function."""
    config = json.loads((Path(path) / "config.json").read_text())
    for key in ("_name_or_path", "transformers_version"):
        config.pop(key, None)
    # Qwen2 with fixed RoPE and an explicitly capped SGLang context. All
    # requests pass token IDs, and rollout sets its own stop IDs. These fields
    # do not change the prefill logits within the configured range.
    if (config.get("model_type") == "qwen2" and not config.get("rope_scaling")
            and not config.get("use_sliding_window") and not config.get("use_mrope")
            and config.get("max_position_embeddings", 0) >= context_limit):
        if dynamic_rope and config.get("architectures") == ["Qwen2ForCausalLM"] and not config.get("dual_chunk_attention_config"):
            config.pop("rope_theta", None)
        for key in ("max_position_embeddings", "sliding_window", "use_mrope", "eos_token_id"):
            config.pop(key, None)
    return json.dumps(config, sort_keys=True)
