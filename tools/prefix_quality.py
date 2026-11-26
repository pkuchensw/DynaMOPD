"""Full-distribution student-to-teacher KL on fixed prefixes of held-out answers."""
import argparse
import json
from pathlib import Path
import tempfile

from DynaMOPD.tools.quality import job_config, model_paths, run_workers


def validate_vocabularies(reference, paths, rows):
    from transformers import AutoTokenizer
    vocab = AutoTokenizer.from_pretrained(reference, local_files_only=True).get_vocab()
    valid = set(vocab.values())
    for index, row in enumerate(rows):
        if not set(row["prompt"] + row["response"]) <= valid:
            raise ValueError(f"held-out prefix {index} contains IDs outside the source tokenizer")
    for path in set(paths) - {Path(reference).resolve()}:
        other = AutoTokenizer.from_pretrained(path, local_files_only=True).get_vocab()
        if other != vocab:
            raise ValueError(f"fixed-prefix KL requires identical token-to-ID mappings: {reference} and {path}")


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
            positions = torch.tensor([len(r["prompt"])-1+p for p in offsets],device=model.device)
            ids = torch.tensor([context],device=model.device)
            logits = model(input_ids=ids,logits_to_keep=positions,use_cache=False).logits[0]
            distributions[r["i"]] = logits.float().log_softmax(-1).cpu()
    torch.save(distributions,output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, required=True, help="training job directory containing heldout-prefixes.jsonl")
    p.add_argument("--state-dir", type=Path, help="override the job's checkpoint state directory")
    p.add_argument("--student", action="append", nargs=2, metavar=("NAME","CHECKPOINT"),
                   help="evaluate this named checkpoint on the same fixed prefixes; repeat for paired runs")
    p.add_argument("--extract", help=argparse.SUPPRESS)
    p.add_argument("--scratch", type=Path, help=argparse.SUPPRESS)
    args=p.parse_args()
    cfg = job_config(args.out)
    prefixes = args.out / "heldout-prefixes.jsonl"
    if not prefixes.is_file():
        p.error(f"missing {prefixes}; run DynaMOPD.tools.quality on the held-out dataset first")
    rows = [json.loads(s) for s in prefixes.read_text(encoding="utf-8").splitlines() if s.strip()]
    if not rows:
        p.error("fixed-prefix evaluation needs at least one held-out answer")
    overrides = args.student or []
    if overrides:
        students = {"initial": cfg["student"]}
        for name, checkpoint in overrides:
            if not name.isidentifier() or name in students:
                p.error("--student requires a unique identifier for each checkpoint")
            students[name] = checkpoint
    else:
        students = model_paths(args.out, args.state_dir, cfg)
    # Reuse distributions when a frozen teacher is identical to the initial student.
    paths = {k:Path(v).resolve() for k,v in students.items()}
    teacher_keys = {}
    for route in dict.fromkeys(row["teacher"] for row in rows):
        if route not in cfg["teachers"]:
            p.error(f"unknown teacher {route} in held-out prefixes")
        path=Path(cfg["teachers"][route]).resolve()
        key=next((k for k,v in paths.items() if v==path),f"teacher-{len(teacher_keys)}")
        paths[key]=path
        teacher_keys[route]=key
    if args.extract:
        key=args.extract
        selected = rows if key in students else [r for r in rows if teacher_keys[r["teacher"]]==key]
        extract(paths[key],selected,args.scratch/f"{key}.pt")
        return
    validate_vocabularies(cfg["student"], paths.values(), rows)
    shared = ["--out", str(args.out)]
    if args.state_dir is not None:
        shared.extend(["--state-dir", str(args.state_dir)])
    for name, checkpoint in overrides:
        shared.extend(["--student", name, checkpoint])
    with tempfile.TemporaryDirectory(dir=args.out) as scratch:
        keys = list(paths)
        run_workers("DynaMOPD.tools.prefix_quality", shared, [["--extract", key, "--scratch", scratch] for key in keys])
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
