"""Full-distribution student-to-teacher KL on fixed prefixes of held-out DAPO answers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from DynaMOPD.cluster.topology import visible_devices
from DynaMOPD.tools.quality import model_paths


def extract(path, rows, output):
    import torch
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(path,dtype=torch.bfloat16,device_map={"":"cuda:0"},
                                                attn_implementation="sdpa",local_files_only=True).eval()
    distributions = {}
    with torch.inference_mode():
        for r in rows:
            n = len(r["response"])
            offsets = sorted({0,n//3,2*n//3,n-1})
            context = r["prompt"] + r["response"][:-1]
            positions = torch.tensor([len(r["prompt"])-1+p for p in offsets],device="cuda")
            ids = torch.tensor([context],device="cuda")
            logits = model(input_ids=ids,logits_to_keep=positions,use_cache=False).logits[0]
            distributions[r["i"]] = logits.float().log_softmax(-1).cpu()
    torch.save(distributions,output)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--state-dir",type=Path,required=True)
    p.add_argument("--student", action="append", nargs=2, metavar=("NAME","CHECKPOINT"),
                   help="evaluate this named checkpoint on the same fixed prefixes; repeat for paired runs")
    p.add_argument("--extract")
    p.add_argument("--scratch",type=Path)
    args=p.parse_args()
    cfg=json.loads((args.out/"config.json").read_text())
    rows=[json.loads(s) for s in (args.out/"heldout-prefixes.jsonl").read_text().splitlines()]
    overrides = args.student or []
    if overrides:
        students = {"initial": cfg["student"]}
        for name, checkpoint in overrides:
            if not name.isidentifier() or name in students:
                p.error("--student requires a unique identifier for each checkpoint")
            students[name] = checkpoint
    else:
        students = model_paths(args.state_dir,cfg)
    # Reuse distributions when a frozen teacher is identical to the initial student.
    paths = {k:Path(v).resolve() for k,v in students.items()}
    teacher_keys = {}
    for route,teacher in cfg["teachers"].items():
        path=Path(teacher).resolve()
        key=next((k for k,v in paths.items() if v==path),f"teacher-{route}")
        paths[key]=path
        teacher_keys[route]=key
    if args.extract:
        key=args.extract
        selected = rows if key in students else [r for r in rows if teacher_keys[r["teacher"]]==key]
        extract(paths[key],selected,args.scratch/f"{key}.pt")
        return
    devices = visible_devices()
    if not devices:
        raise ValueError("prefix evaluation needs a visible GPU")
    def lane(job):
        index,group = job
        for key in group:
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=devices[index],TOKENIZERS_PARALLELISM="false",
                     HF_HUB_OFFLINE="1",OMP_NUM_THREADS="2")
            subprocess.run([sys.executable,"-m","DynaMOPD.tools.prefix_quality","--out",str(args.out),
                "--state-dir",str(args.state_dir),"--extract",key,"--scratch",scratch,
                *(arg for record in overrides for arg in ("--student",*record))],env=env,check=True)
    with tempfile.TemporaryDirectory(dir=args.state_dir) as scratch:
        keys=list(paths)
        lanes=[keys[i::len(devices)] for i in range(min(len(keys),len(devices)))]
        with ThreadPoolExecutor(max_workers=len(lanes)) as pool:
            list(pool.map(lane,enumerate(lanes)))
        import torch
        torch.set_num_threads(4)
        tables={k:torch.load(Path(scratch)/f"{k}.pt",weights_only=True) for k in keys}
        result={}
        for key in students:
            per_question=[]
            for r in rows:
                logp=tables[key][r["i"]].double()
                logq=tables[teacher_keys[r["teacher"]]][r["i"]].double()
                if logp.shape!=logq.shape:
                    raise ValueError("held-out distributions use incompatible vocabularies")
                values=(logp.exp()*(logp-logq)).sum(-1).tolist()
                per_question.append(dict(i=r["i"],teacher=r["teacher"],kl_nats=values))
            terms=[v for row in per_question for v in row["kl_nats"]]
            result[key]=dict(mean_kl_nats=sum(terms)/len(terms),positions=len(terms),per_question=per_question)
    (args.out/"heldout-kl.json").write_text(json.dumps(result,indent=2))
    print({k:v["mean_kl_nats"] for k,v in result.items()})


if __name__=="__main__":
    main()
