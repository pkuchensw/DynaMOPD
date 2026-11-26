"""Stage immutable model files and identify compatible inference architectures."""
import errno
import hashlib
import json
from pathlib import Path
import tempfile
from uuid import uuid4


def file_digest(path, output=None):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024 * 1024):
            digest.update(chunk)
            if output is not None:
                output.write(chunk)
    return digest.hexdigest()


def stage_model(path, folder):
    source = Path(path).resolve()
    files = sorted(f for f in source.iterdir() if f.is_file() and f.suffix in
                   (".safetensors", ".json", ".jinja", ".txt", ".model"))
    if not any(f.suffix==".safetensors" for f in files):
        raise ValueError(f"no safetensors checkpoint under {source}")
    manifest = {f.name: file_digest(f) for f in files}
    digest = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    dest = folder/(source.name+"-"+digest)
    if not dest.exists():
        # A sibling staging task or process may publish this same image first.
        # Only a fully copied directory is visible at the content-keyed path.
        with tempfile.TemporaryDirectory(dir=folder) as scratch:
            candidate = Path(scratch) / "model"
            candidate.mkdir()
            for f in files:
                with (candidate / f.name).open("xb") as output:
                    if file_digest(f, output) != manifest[f.name]:
                        raise RuntimeError(f"checkpoint changed while staging {f}")
            try:
                candidate.rename(dest)
            except OSError as exc:
                if exc.errno not in (errno.EEXIST, errno.ENOTEMPTY) or not dest.is_dir():
                    raise
    return str(dest)


def stage_models(cfg, pool, folder=None):
    # Shared by all worker processes in this invocation, including Ray actors.
    cfg["weight_sync_run_id"] = uuid4().hex
    folder = Path(folder) if folder is not None else Path(cfg["state_dir"])/"initial_models"
    folder.mkdir(parents=True, exist_ok=True)
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
