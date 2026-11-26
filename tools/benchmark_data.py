"""Evidence linking the canonical prompt/teacher pairs to each native training input."""
import hashlib
import json
from pathlib import Path

from ..trainer.data import load_data


def prompt_key(prompt):
    return json.dumps(prompt, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def check_rows(reference, actual, teachers):
    if len(reference) != len(actual):
        raise ValueError("training inputs have different row counts")
    for i, (left, right) in enumerate(zip(reference, actual)):
        route = left.get("teacher_routing_key")
        if route not in teachers or right.get("teacher_routing_key") != route:
            raise ValueError(f"training row {i} has an unknown or mismatched teacher route")
        if left["prompt"] != right["prompt"]:
            raise ValueError(f"training row {i} has a different prompt")


def training_input(folder, system, command, reference, cfg, *, at_launch=False):
    if system == "verl":
        from .benchmark_layout import hydra_values
        argument = hydra_values(command)["data.train_files"]
        relative = Path("train.parquet")
    else:
        relative = Path("kdflow/sampler-input.jsonl") if system == "kdflow" else Path("train.jsonl")
        flag = "--data" if system == "dynamopd" else "--train_dataset_path"
        if command.count(flag) != 1:
            raise ValueError(f"expected a single {flag} argument")
        argument = command[command.index(flag) + 1]
    if Path(argument).parts[-len(relative.parts):] != relative.parts:
        raise ValueError(f"{system} command points to a different training input")
    path = folder / relative
    if at_launch and Path(argument).resolve() != path.resolve():
        raise ValueError(f"{system} command does not reference the prepared training file")
    data = load_data(path)
    if system == "kdflow":
        import torch
        order = torch.randperm(len(data), generator=torch.Generator().manual_seed(cfg["seed"])).tolist()
        data = [data[i] for i in order]
    check_rows(reference, data, cfg["teachers"])
    if system == "dynamopd":
        from ..algorithms import build_algorithm
        algorithm = build_algorithm(cfg)
        if any(algorithm.route(row, cfg) != row["teacher_routing_key"] for row in data):
            raise ValueError("DynaMOPD route mapping differs from the benchmark teacher")
    return {"path": relative.as_posix(), "argument": argument, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
