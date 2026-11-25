"""Greedy held-out DAPO exact match for frozen initial and final checkpoints."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import re
import subprocess
import sys

from DynaMOPD.cluster.topology import visible_devices


def model_paths(state, cfg):
    paths = dict(initial=cfg["student"], dynamopd=state/cfg.get("dynamopd_run","dynamopd")/f"v{cfg['rounds']}",
                kdflow=state/"kdflow/checkpoint",
                verl=state/"verl/checkpoint"/f"global_step_{cfg['rounds']}/actor/huggingface")
    return {name:paths[name] for name in ("initial", *cfg.get("benchmark_systems", ("dynamopd","kdflow","verl")))}


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


def evaluate(name, path, folder, cfg):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    tok = AutoTokenizer.from_pretrained(cfg["student"], padding_side="left")
    tok.pad_token_id = tok.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa",
                                                device_map={"":"cuda:0"}, local_files_only=True).eval()
    samples = [json.loads(line) for line in (folder/"heldout.jsonl").read_text().splitlines()]
    result, reference = [], []
    generation = GenerationConfig.from_pretrained(cfg["student"], local_files_only=True)
    eos = generation.eos_token_id
    stop_ids = set(eos if isinstance(eos,list) else [eos])
    for begin in range(0,len(samples),16):
        chunk = samples[begin:begin+16]
        texts = [tok.apply_chat_template(x["prompt"],add_generation_prompt=True,tokenize=False) for x in chunk]
        inputs = tok(texts,return_tensors="pt",padding=True,add_special_tokens=False).to("cuda:0")
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
                reference.append(dict(i=begin+i,teacher=x["teacher_routing_key"],prompt=prompt,response=response[:count]))
    if reference:
        (folder/"heldout-prefixes.jsonl").write_text("".join(json.dumps(x)+"\n" for x in reference))
    (folder/f"heldout-{name}.json").write_text(json.dumps(result,ensure_ascii=False,indent=2))
    return sum(x["correct"] for x in result)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--state-dir",type=Path,required=True)
    p.add_argument("--worker",choices=("initial","dynamopd","kdflow","verl"))
    args=p.parse_args()
    cfg=json.loads((args.out/"config.json").read_text())
    models=model_paths(args.state_dir,cfg)
    if args.worker:
        print(args.worker,evaluate(args.worker,models[args.worker],args.out,cfg),flush=True)
        return
    devices = visible_devices()
    if len(devices)<len(models):
        raise ValueError("evaluation launcher uses one GPU per model")
    def run(pair):
        device,name=pair
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=device,TOKENIZERS_PARALLELISM="false",
                 HF_HUB_OFFLINE="1",OMP_NUM_THREADS="2")
        subprocess.run([sys.executable,"-m","DynaMOPD.tools.quality","--out",str(args.out),
                        "--state-dir",str(args.state_dir),"--worker",name],env=env,check=True)
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        list(executor.map(run,zip(devices,models)))


if __name__=="__main__":
    main()
