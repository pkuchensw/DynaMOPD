"""Greedy integer-answer exact match on a training job's initial and final models."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from DynaMOPD.cluster.topology import visible_devices
from DynaMOPD.algorithms import build_algorithm
from DynaMOPD.trainer.data import load_data


def job_config(folder):
    """Read the resolved training config, including resumed runs' final version."""
    trace = folder / "events.jsonl"
    if trace.is_file():
        with trace.open(encoding="utf-8") as stream:
            for line in stream:
                event = json.loads(line)
                if event.get("event") == "config":
                    return event["config"]
        raise ValueError(f"no training configuration in {trace}")
    # Existing job collections keep the config and held-out set in their root.
    cfg = json.loads((folder / "config.json").read_text(encoding="utf-8"))
    run = folder / cfg.get("dynamopd_run", "dynamopd")
    if (run / "events.jsonl").is_file():
        return dict(cfg, **job_config(run))
    return cfg


def model_paths(folder, state, cfg):
    version = f"v{cfg.get('final_version', cfg['rounds'])}"
    label = cfg.get("dynamopd_run", "dynamopd")
    if state is not None:
        roots = [state, state / label]
    else:
        roots = [folder / "weights", folder / "states" / label]
        if cfg.get("state_dir"):
            roots.insert(0, Path(cfg["state_dir"]))
    for root in roots:
        if (root / version).is_dir():
            return {"initial": Path(cfg["student"]), "dynamopd": root / version}
    raise FileNotFoundError(f"missing final checkpoint {version} under {roots}; set --state-dir to the training state directory")


def run_workers(module, args, jobs):
    devices = visible_devices()
    if not devices:
        raise ValueError("evaluation needs a visible CUDA GPU")

    def lane(index):
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=devices[index], TOKENIZERS_PARALLELISM="false",
                   HF_HUB_OFFLINE="1", OMP_NUM_THREADS="2")
        for job in jobs[index::len(devices)]:
            subprocess.run([sys.executable, "-m", module, *args, *job], env=env, check=True)

    lanes = min(len(devices), len(jobs))
    with ThreadPoolExecutor(max_workers=lanes) as executor:
        list(executor.map(lane, range(lanes)))


def final_integer(text):
    # Require the whole answer to be an integer. A leading digit from a
    # fraction or decimal cannot earn credit for its integer part.
    markers = list(re.finditer(r"\bAnswer\s*:|\\boxed\{", text, re.I))
    if not markers:
        return None
    last = markers[-1]
    tail = text[last.end():]
    if last.group().startswith("\\"):
        tail, sep, _ = tail.partition("}")
        if not sep:
            return None
    else:
        tail = tail.strip().split("\n",1)[0].strip(" \t*$")
        tail = tail.removesuffix(".").rstrip("$").strip()
    match = re.fullmatch(r"([-+]?(?:\d{1,3}(?:,\d{3})+|\d+))(?:\.0+)?",tail.strip())
    return str(int(match[1].replace(",",""))) if match else None


def evaluate(name, path, folder, cfg, heldout):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    tok = AutoTokenizer.from_pretrained(cfg["student"], padding_side="left", local_files_only=True)
    tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                device_map={"":"cuda:0"}, local_files_only=True).eval()
    samples = load_data(heldout)
    algorithm = build_algorithm(cfg)
    result, reference = [], []
    generation = GenerationConfig.from_pretrained(cfg["student"], local_files_only=True)
    eos = generation.eos_token_id
    stop_ids = set(eos if isinstance(eos,list) else [eos])
    for begin in range(0,len(samples),16):
        chunk = samples[begin:begin+16]
        texts = []
        for row in chunk:
            messages = row.get("prompt", row.get("conversations"))
            if isinstance(messages, str):
                messages = [{"role": "user", "content": messages}]
            texts.append(tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False))
        inputs = tok(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(model.device)
        with torch.inference_mode():
            generated = model.generate(**inputs, generation_config=generation,
                                       max_new_tokens=cfg["max_new_tokens"], do_sample=False,
                                       pad_token_id=tok.pad_token_id)
        answers = tok.batch_decode(generated[:,inputs.input_ids.shape[1]:],skip_special_tokens=True)
        for i,(x,a) in enumerate(zip(chunk,answers,strict=True)):
            gold, pred = str(x["reward_model"]["ground_truth"]), final_integer(a)
            result.append(dict(i=begin+i,gold=gold,prediction=pred,correct=pred==gold,response=a))
            if name=="initial":
                prompt = inputs.input_ids[i][inputs.attention_mask[i].bool()].tolist()
                response = generated[i,inputs.input_ids.shape[1]:].tolist()
                count = next((j+1 for j,v in enumerate(response) if v in stop_ids),len(response))
                route = algorithm.route(x, cfg)
                if route not in cfg["teachers"]:
                    raise ValueError(f"unknown teacher {route} in held-out row {begin+i}")
                reference.append(dict(i=begin+i, teacher=route, prompt=prompt, response=response[:count]))
    if reference:
        (folder/"heldout-prefixes.jsonl").write_text("".join(json.dumps(x)+"\n" for x in reference), encoding="utf-8")
    (folder/f"heldout-{name}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2), encoding="utf-8")
    return sum(x["correct"] for x in result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="training job directory for inputs and evaluation results")
    p.add_argument("--state-dir", type=Path, help="override the job's checkpoint state directory")
    p.add_argument("--heldout", type=Path, help="held-out integer-answer dataset (default: RUN/heldout.jsonl)")
    p.add_argument("--worker", choices=("initial","dynamopd"), help=argparse.SUPPRESS)
    args=p.parse_args()
    cfg = job_config(args.out)
    models = model_paths(args.out, args.state_dir, cfg)
    heldout = args.heldout or args.out / "heldout.jsonl"
    if not heldout.is_file():
        p.error(f"missing held-out dataset {heldout}; use --heldout to provide it")
    if args.worker:
        print(args.worker, evaluate(args.worker, models[args.worker], args.out, cfg, heldout), flush=True)
        return
    shared = ["--out", str(args.out), "--heldout", str(heldout)]
    if args.state_dir is not None:
        shared.extend(["--state-dir", str(args.state_dir)])
    run_workers("DynaMOPD.tools.quality", shared, [["--worker", name] for name in models])


if __name__=="__main__":
    main()
