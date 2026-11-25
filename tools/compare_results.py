"""Collect measured throughput and validation scores, retaining the measurement boundary."""
import argparse
import hashlib
from collections import Counter
import json
import math
from pathlib import Path
import re

from DynaMOPD.tools.audit import audit
from DynaMOPD.tools.benchmark_layout import check_command, check_layouts


def metric(line, name):
    match = re.search(r"(?<![\w/])"+re.escape(name)+r":\s*([-+\d.eE]+)",line)
    if match is None:
        raise ValueError(f"missing {name} in native training metrics")
    return float(match.group(1))


def read_jsonl(path):
    # U+2028 can occur inside a model's output string. Only physical newlines
    # delimit JSONL records.
    with path.open(encoding="utf-8") as records:
        return [json.loads(line) for line in records]


def verify_questions(run, manifest, dataset, tokenizer, strict=False):
    batch = manifest["batch"]
    matched = []
    for version in range(manifest["rounds"]):
        expected = dataset[version*batch:(version+1)*batch]
        if manifest["system"]=="dynamopd":
            samples = read_jsonl(run/f"samples-v{version}.jsonl")
            texts = tokenizer.batch_decode([s["prompt"] for s in samples],skip_special_tokens=False)
            if [s["teacher"] for s in samples]!=[r["teacher_routing_key"] for r in expected]:
                raise ValueError("DynaMOPD teacher route mismatch")
        else:
            path = run/"outputs"/f"{version+1}.jsonl"
            if not path.exists() and manifest["system"]=="kdflow":
                path = Path(manifest["state"])/"checkpoint/rollout_data"/path.name
            key = "prompt" if manifest["system"]=="kdflow" else "input"
            texts = [sample[key] for sample in read_jsonl(path)]
        if strict:
            ids = [tokenizer.apply_chat_template(r["prompt"], tokenize=True, add_generation_prompt=True,
                                                 enable_thinking=False, return_dict=False) for r in expected]
            if manifest["system"] == "dynamopd" and [s["prompt"] for s in samples] != ids:
                raise ValueError("DynaMOPD prompt token IDs differ from the shared template")
            rendered = tokenizer.batch_decode(ids, skip_special_tokens=manifest["system"] == "verl")
            if len(rendered) != batch or Counter(texts) != Counter(rendered):
                raise ValueError(f"{manifest['system']} step {version+1} used different full prompts")
            # Distinct official prompts can differ only in whitespace. The
            # exact multiset above preserves that distinction and multiplicity.
            matched.append(len(rendered))
            continue
        actual = [" ".join(text.split()) for text in texts]
        questions = [" ".join(row["prompt"][-1]["content"].split()) for row in expected]
        indices = [[i for i,text in enumerate(actual) if question in text] for question in questions]
        if (len(actual)!=batch or any(len(matches)!=1 for matches in indices)
                or sorted(matches[0] for matches in indices)!=list(range(batch))):
            raise ValueError(f"{manifest['system']} step {version+1} used a different question set")
        matched.append(len(questions))
    return matched


def summarize(folder, *, training_only=False, systems=None):
    from transformers import AutoTokenizer
    result = {}
    cfg = json.loads((folder/"config.json").read_text())
    tokenizer=AutoTokenizer.from_pretrained(cfg["student"],local_files_only=True)
    dataset=read_jsonl(folder/"train.jsonl")
    systems = systems or cfg.get("benchmark_systems", ("dynamopd","kdflow","verl"))
    strict = cfg.get("algorithm") == "on_policy_k1" and cfg.get("train_on_length", False)
    shared_layout = None
    if len(systems) > 1:
        if "verl_teacher_gpus" not in cfg or "inference_tp_size" not in cfg:
            raise ValueError("these historical runs have no shared parallel layout; request one system for its absolute metrics")
        shared_layout = check_layouts(cfg, systems)
        config_digest = hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()
        dataset_digest = hashlib.sha256((folder/"train.jsonl").read_bytes()).hexdigest()
    for system in systems:
        run = folder/(cfg.get("dynamopd_run","dynamopd") if system=="dynamopd" else system)
        manifest = json.loads((run/"run.json").read_text())
        if manifest.get("exit_code")!=0:
            raise RuntimeError(f"{system} did not finish")
        if shared_layout is not None:
            if (manifest.get("system") != system or manifest.get("parallel_layout") != shared_layout
                    or manifest.get("devices") != shared_layout["devices"]
                    or manifest.get("config_sha256") != config_digest or manifest.get("dataset_sha256") != dataset_digest):
                raise ValueError(f"{system} was launched with a different configuration, GPU allocation or dataset")
            check_command(cfg, system, manifest["command"])
        batch = manifest["batch"]
        if system=="dynamopd":
            rows = audit(run)["rounds"]
            times = [x["round_s"] for x in rows]
            totals = [x["generated_tokens"] for x in rows]
        else:
            lines = (run/"run.log").read_text(errors="replace").splitlines()
            if system=="kdflow":
                lines = [line for line in lines if "timing/step_time:" in line and "rollout/response_length/mean:" in line]
                times = [metric(line,"timing/step_time") for line in lines]
                totals = [round(metric(line,"rollout/response_length/mean")*batch) for line in lines]
            else:
                lines = [line for line in lines if "timing_s/step:" in line and "response_length/mean:" in line]
                times = [metric(line,"timing_s/step") for line in lines]
                totals = [round(metric(line,"response_length/mean")*batch) for line in lines]
        if len(times)!=manifest["rounds"]:
            raise ValueError(f"missing {system} steps: {times}")
        wall = manifest["end"]-manifest["start"]
        entry = dict(wall_s=wall, wall_generated_tokens_s=sum(totals)/wall,
                     wall_prompts_s=batch*manifest["rounds"]/wall,round_s=times,
                     generated_tokens=totals, all_rounds_generated_tokens_s=sum(totals)/sum(times),
                     warm_generated_tokens_s=sum(totals[1:])/sum(times[1:]) if len(times)>1 else None)
        entry["verified_questions_per_step"]=verify_questions(run,manifest,dataset,tokenizer,strict)
        if system=="dynamopd":
            entry["effective_training_tokens"] = [x["training_tokens"] for x in rows]
        if strict:
            if system == "kdflow":
                if manifest.get("algorithm") != "verl_k1":
                    raise ValueError("KDFlow comparison requires the sampled K1 adapter")
                logs = (run/"run.log").read_text(errors="replace")
                counts = [int(n) for n in re.findall(r"matched_training_tokens=(\d+)", logs)]
                entry["effective_training_tokens"] = counts
            elif system == "verl":
                cmd = set(manifest["command"])
                required = {"distillation.distillation_loss.loss_mode=k1",
                            "distillation.distillation_loss.use_policy_gradient=true",
                            "distillation.distillation_loss.use_task_rewards=false",
                            f"actor_rollout_ref.actor.ppo_mini_batch_size={batch}"}
                if not required <= cmd:
                    raise ValueError("native verl job uses a different distillation recipe")
                entry["effective_training_tokens"] = totals
            if entry["effective_training_tokens"] != totals:
                raise ValueError(f"{system} did not train every generated response token")
            entry["full_rendered_prompts_verified"] = True
            if len(times) > 2:
                entry["warm_without_final_save_tokens_s"] = sum(totals[1:-1]) / sum(times[1:-1])
        result[system]=entry
    if training_only:
        return result
    result["heldout"] = {}
    heldout = read_jsonl(folder/"heldout.jsonl")
    targets = [(i, str(row["reward_model"]["ground_truth"])) for i,row in enumerate(heldout)]
    routes = [(i,row["teacher_routing_key"]) for i,row in enumerate(heldout)]
    dist = json.loads((folder/"heldout-kl.json").read_text())
    for name in ("initial", *systems):
        samples = json.loads((folder/f"heldout-{name}.json").read_text())
        if [(s["i"],s["gold"]) for s in samples] != targets:
            raise ValueError(f"{name} evaluated different held-out questions")
        per_question = dist[name]["per_question"]
        if [(s["i"],s["teacher"]) for s in per_question] != routes:
            raise ValueError(f"{name} evaluated different held-out teacher routes")
        counts = [len(s["kl_nats"]) for s in per_question]
        if counts != [len(s["kl_nats"]) for s in dist["initial"]["per_question"]]:
            raise ValueError(f"{name} evaluated different numbers of positions")
        values = [v for s in per_question for v in s["kl_nats"]]
        if (not values or not all(math.isfinite(v) for v in values)
                or len(values)!=dist[name]["positions"]
                or not math.isclose(sum(values)/len(values),dist[name]["mean_kl_nats"])):
            raise ValueError(f"{name} has incomplete held-out KL measurements")
        result["heldout"][name] = dict(correct=sum(x["correct"] for x in samples),n=len(samples),
                                       fixed_prefix_kl_nats=dist[name]["mean_kl_nats"],
                                       positions=dist[name]["positions"])
    if strict and shared_layout is not None:
        baseline_name = "kdflow" if "kdflow" in systems else "verl"
        baseline = result[baseline_name]
        result["parallel_layout"] = shared_layout
        for name in systems:
            for label, metric_name in (("wall", "wall_generated_tokens_s"), ("warm", "warm_generated_tokens_s"),
                                       ("warm_without_final_save", "warm_without_final_save_tokens_s")):
                if baseline.get(metric_name):
                    result[name][f"{label}_speedup_vs_{baseline_name}"] = result[name][metric_name] / baseline[metric_name]
        result["recipe"] = dict(objective="sampled K1 with vanilla PPO", mask="all sampled response tokens",
                                old_policy="recomputed frozen actor", aggregation="global token mean",
                                updates_per_round=1, lr=cfg["lr"], grad_clip=cfg["clip"],
                                same_questions=True, independent_on_policy_sampling=True)
    return result


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("folder",type=Path)
    p.add_argument("--training-only", action="store_true", help="verify training metrics before held-out evaluation")
    p.add_argument("--systems", nargs="+", choices=("dynamopd", "kdflow", "verl"))
    a=p.parse_args()
    result=json.dumps(summarize(a.folder, training_only=a.training_only, systems=a.systems),indent=2)
    target = "training-metrics.json" if a.training_only else "comparison.json"
    (a.folder/target).write_text(result+"\n")
    print(result)
