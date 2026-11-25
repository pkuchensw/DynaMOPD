"""Derive SimRL cost inputs from completed worker operations of a real round."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import median


def fit(folder, version):
    events = [json.loads(line) for line in (folder/"events.jsonl").read_text().splitlines()]
    header = events[0]
    if header["backend"] != "sglang_fsdp2":
        raise ValueError("calibration requires native SGLang/FSDP2 worker receipts")
    cfg = {k:v for k,v in header["config"].items()
           if k not in ("state_dir","log_dir","export_dir","final_version","excluded_ids","eos_ids","staged")}
    # Runs recorded before tensor publication always used the HF export path.
    cfg.setdefault("weight_sync", "disk")
    samples = {r["rid"]:r for r in map(json.loads,(folder/f"samples-v{version}.jsonl").read_text().splitlines())}
    pending = {}
    observations = defaultdict(list)
    switches = defaultdict(list)
    activation = defaultdict(list)
    for i,e in enumerate(events):
        if e.get("version") != version:
            continue
        if e["event"] == "dispatch":
            pending[e["gpu"]] = e
        if e["event"] != "complete":
            continue
        g,op,dt = e["gpu"],e["op"],e["end"]-e["start"]
        ids = pending.pop(g)["ids"]
        if op=="commit":
            observations["optimizer"].append(e.get("optimizer_and_probe_s",dt))
            if e.get("durable",True):
                observations["commit"].append(dt)
            if "dcp_write_s" in e and e.get("hf_checkpoint",True):
                observations["export"].append(dt-e["dcp_write_s"])
        elif op=="move" and e["accepted"]:
            install = e["install_end"]-e["install_start"]
            observations["move_base"].append(dt-install)
            if e["rebuilt_tokens"]:
                observations["prefill_token"].append(install/e["rebuilt_tokens"])
        elif op=="switch":
            wait = e.get("standby_wait_s", 0.)
            if "teacher_rank_release_s" in e:
                observations[("release", e["old"])].append(e["teacher_rank_release_s"])
            ranks = e.get("teacher_rank_timings", [])
            if ranks and all("prepared_import_s" in r for r in ranks):
                measured = e.get("teacher_prepare_s")
                observations["prepare"].append(measured if measured is not None else max(r["prepared_import_s"] for r in ranks))
            if "teacher_rank_load_s" in e:
                activation[e["new"]].append(max(0., e["teacher_rank_load_s"] - wait))
            if e.get("teacher_start_s"):
                observations["teacher_start"].append(e["teacher_start_s"])
            if e.get("teacher_weights_reused"):
                observations["teacher_wake"].append(e["load_s"])
            # Initial rollout engines were already hot at round_start.
            key = e["old"]
            if key==e["new"]=="rollout" and e["engine_weight_version"]==f"v{version}":
                key = "cold"
            switches[(key,e["new"])].append((e["unload_s"], max(0., e["load_s"] - wait)))
            if e["old"]!=e["new"]:
                observations["switch"].append(dt)
        elif op=="train":
            width = sum(len(samples[r]["prompt"])+len(samples[r]["response"]) for r in ids)
            observations["train_token"].append(dt/width)
        elif op in ("decode","score"):
            receipts = events[i+1:i+1+len(ids)]
            expected = "prefix" if op=="decode" else "supervision"
            if any(x["event"]!=expected for x in receipts) or {x["rid"] for x in receipts}!=set(ids):
                raise ValueError("missing per-request receipts")
            if op=="decode":
                observations["decode_step"].append(dt/max(x["end"]-x["start"] for x in receipts))
            else:
                width = sum(len(samples[x["rid"]]["prompt"])+x["end"]-x["reused_kv_tokens"] for x in receipts)
                observations["score"].append((max(1,width),dt))
    if not observations["optimizer"]:
        raise ValueError("calibration requires a completed round")
    if activation:
        cfg["sim_teacher_activation_s"] = {role: median(seconds) for role, seconds in activation.items()}
    if observations["prepare"]:
        cfg["sim_teacher_prepare_s"] = median(observations["prepare"])
    releases = {role: median(observations["release", role]) for role in cfg["teachers"] if observations["release", role]}
    if releases:
        cfg["sim_teacher_release_s"] = releases
    cfg["sim_optimizer_s"] = median(observations["optimizer"])
    for label in ("teacher_start", "teacher_wake"):
        if observations[label]:
            cfg["sim_"+label+"_s"] = median(observations[label])
    if observations["commit"]:
        cfg["sim_publish_s"] = median(observations["commit"])
    if observations["export"]:
        cfg["sim_export_s"] = median(observations["export"])
    cfg["sim_switch_s"] = {a:{b:[median(row[i] for row in pairs) for i in range(2)]
                                for (old,b),pairs in switches.items() if old==a}
                           for a,_ in switches}
    for key in ("move_base","prefill_token","switch","train_token","decode_step"):
        if observations[key]:
            cfg["costs"][key] = max(1e-9,median(observations[key]))
    # Estimate fixed HTTP/launch cost on short batches and marginal cost on longer ones.
    score = sorted(observations["score"])
    if score:
        n = max(1,len(score)//4)
        lo,hi = score[:n],score[-n:]
        short,long = median(x for x,_ in lo),median(x for x,_ in hi)
        launch = median(t for _,t in lo)
        slope = max(1e-9,(median(t for _,t in hi)-launch)/max(1,long-short))
        cfg["costs"].update(score_base=max(.000001,launch-slope*short),score_token=slope)
    return cfg


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("trace",type=Path)
    p.add_argument("--version",type=int,default=0)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    with args.output.open("x") as f:
        json.dump(fit(args.trace,args.version),f,indent=2)
        f.write("\n")
