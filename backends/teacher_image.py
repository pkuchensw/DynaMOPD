"""Immutable, post-load SGLang tensor-parallel images for repeated frozen-teacher leases."""
import fcntl
import glob
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import tempfile


class TeacherImage:
    def __init__(self, state_dir, model, width, device_type, *, source_id=None):
        # Ray can stage the same immutable source under a different temporary parent per actor.
        # The staged basename includes the source fingerprint. Keep the original native cache key.
        source = Path(state_dir) / "initial_models" / source_id if source_id else Path(model)
        identity = dict(source=str(source.resolve()), tp=width, dtype="bfloat16",
                        device=device_type, format=1,
                        packages={k: version(k) for k in ("sglang", "torch", "transformers")})
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.path = Path(state_dir) / "teacher_rank_images" / digest
        self.width = width

    def reference(self):
        try:
            return json.loads((self.path / "reference.json").read_text())
        except FileNotFoundError:
            return None

    def loader_args(self):
        # The model path still names the original HF config/tokenizer. SGLang
        # expands rank/part placeholders and then globs this absolute path.
        directory = glob.escape(str(self.path.resolve())).replace("{", "{{").replace("}", "}}")
        pattern = directory + "/model-rank-{rank}-part-{part}.safetensors"
        return ["--load-format", "sharded_state", "--model-loader-extra-config",
                json.dumps({"pattern": pattern})]

    def build(self, save, probe):
        """One representative TP lane writes. Concurrent lanes continue using HF."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.with_suffix(".lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return False
            if self.reference() is not None:
                return False
            with tempfile.TemporaryDirectory(dir=self.path.parent) as tmp:
                candidate = Path(tmp) / "image"
                candidate.mkdir()
                reference = probe()
                save(str(candidate))
                for rank in range(self.width):
                    if not (candidate / f"model-rank-{rank}-part-0.safetensors").is_file():
                        raise RuntimeError(f"missing frozen teacher shard for rank {rank}")
                (candidate / "reference.json").write_text(json.dumps(reference, allow_nan=False))
                candidate.rename(self.path)
            return True
