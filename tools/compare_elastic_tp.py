"""Compare two completed training runs that differ only in the elastic TP policy/profile."""
import argparse
import json
from pathlib import Path

from DynaMOPD.tools.audit import audit


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def events(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)


def collect(folder):
    job = folder
    if not (job/"events.jsonl").is_file():
        wrapper = read(folder/"config.json")
        job = folder/wrapper.get("dynamopd_run", "dynamopd")
    manifest = read(job/"run.json") if (job/"run.json").is_file() else {}
    if manifest and (manifest.get("exit_code") != 0 or manifest.get("system") != "dynamopd"):
        raise ValueError(f"no successful DynaMOPD run in {job}")
    trace = list(events(job/"events.jsonl"))
    if not trace or trace[-1]["event"] != "run_finished":
        raise ValueError(f"incomplete training trace in {job}")
    launch = next(e for e in trace if e["event"] == "config")
    if launch["backend"] != "sglang_fsdp2":
        raise ValueError("TP timings require two SGLang/FSDP2 training runs")
    config = launch["config"]
    placements = next(e["placements"] for e in trace if e["event"] == "executor_ready")
    base_tp = config.get("inference_tp_size", config.get("group_size"))
    rounds = []
    for result in audit(job)["rounds"]:
        v, duration = result["version"], result["round_s"]
        step = [e for e in trace if e.get("version")==v]
        requests = next(e for e in step if e["event"]=="round_start")["requests"]
        inputs = {r["rid"]:r for r in requests}
        if len(inputs) != len(requests) or len(inputs) != config["batch_size"]:
            raise ValueError(f"request count differs from the configured batch in {job} version {v}")
        pending = {}
        chunks = []
        expansions = []
        positions = dict.fromkeys(inputs,0)
        for e in step:
            op = e.get("op")
            if e["event"]=="dispatch":
                pending[e["gpu"]] = e
            if e["event"]=="complete":
                d = pending.pop(e["gpu"])
                assert d["op"] == op
                if op=="tp_expand":
                    expansions.append(dict(rid=d["ids"][0], position=positions[d["ids"][0]], **e))
                if op=="decode":
                    width = e.get("inference_tp_size", e.get("tp_width"))
                    if "peer" in d or "peer" in e:
                        is_wide = d.get("peer", e.get("peer")) is not None
                    elif width is not None and base_tp is not None:
                        is_wide = width > base_tp
                    else:
                        raise ValueError("decode trace has neither a TP lease nor an identifiable base width")
                    if is_wide and len(d["ids"]) != 1:
                        raise ValueError("wide decode receipt must belong to a single tail request")
                    chunk = dict(ids=d["ids"], wide=is_wide, width=width,
                                 s=e["end"]-e["start"], end=e["end"], tokens=0)
                    if len(d["ids"])==1:
                        rid, = d["ids"]
                        chunk.update(rid=rid,position=positions[rid])
                    chunks.append(chunk)
            if e["event"]=="prefix":
                assert e["rid"] in chunks[-1]["ids"] and positions[e["rid"]] == e["start"]
                chunks[-1]["tokens"] += e["end"]-e["start"]
                positions[e["rid"]] = e["end"]
        wide = [c for c in chunks if c["wide"]]
        for receipt in expansions:
            if receipt.get("accepted"):
                first = next((c for c in wide if c["rid"]==receipt["rid"] and c["position"]==receipt["position"]
                              and c["end"]>=receipt["end"]),None)
                if first is not None:
                    receipt.update(first_chunk_tokens=first["tokens"],handoff_to_first_chunk_s=first["end"]-receipt["start"])
        commit = next(e for e in step if e["event"]=="complete" and e.get("op")=="commit")
        checkpoint_s = sum(commit[name] for name in ("dcp_write_s","gather_weights_s","hf_export_s"))
        publications = [e for e in step if e["event"]=="complete" and e.get("weight_sync")=="tensor"]
        source = next((e for e in publications if e["gpu"]==e["publish_source"]), {})
        rounds.append(dict(version=v,round_s=duration,tokens=result["training_tokens"],
                           checkpoint_s=checkpoint_s,training_s=duration-checkpoint_s,
                           expansion=expansions,wide_chunks=len(wide),wide_tokens=sum(c["tokens"] for c in wide),
                           wide_decode_s=sum(c["s"] for c in wide),
                           release_s=sum(e["end"]-e["start"] for e in step if e.get("op")=="tp_shrink" and e["event"]=="complete"),
                           publisher_state_device=source.get("trainer_state_device"),
                           publisher_free_bytes_after=source.get("free_bytes"),
                           publisher_allocated_bytes=source.get("publisher_allocated_bytes"),
                           wide_publication_longest_group_s=max((e.get("wide_publish_s",0) for e in step if e["event"]=="complete"),default=0),
                           requests=inputs,lengths=positions,
                           single_chunks=[{k:v for k,v in c.items() if k!="ids"} for c in chunks if "rid" in c]))
    if not rounds or len(rounds) != config["rounds"]:
        raise ValueError(f"recorded training rounds differ from the configured run in {job}")
    wall = manifest["end"] - manifest["start"] if manifest else None
    run = dict(steps=rounds,wall_s=wall,questions=[],bootstrap=next((e["meshes"] for e in trace if e["event"]=="wide_mesh_ready"),[]))
    return job, config.copy(), placements, manifest, run


def samples(job, step):
    v = step["version"]
    requests, lengths = step.pop("requests"), step.pop("lengths")
    identities, responses = {}, {}
    for row in events(job/f"samples-v{v}.jsonl"):
        rid = row["rid"]
        req = requests.get(rid)
        if not req or rid in identities or row["version"] != v:
            raise ValueError(f"unexpected sample {rid} in {job} version {v}")
        limit = row.get("max_new_tokens", req["max_new_tokens"])
        if ((req["seed"], req["teacher"], req["prompt_tokens"], req["max_new_tokens"], lengths[rid])
                != (row["seed"], row["teacher"], len(row["prompt"]), limit, len(row["response"]))):
            raise ValueError(f"sample {rid} disagrees with its trace in {job}")
        identities[rid] = (row["prompt"], row["teacher"], row["seed"], limit)
        responses[rid] = row["response"]
    if identities.keys() != requests.keys():
        raise ValueError(f"missing samples in {job} version {v}")
    return identities, responses


def compare(off,on):
    jobs, cfgs, placements, manifests, runs = zip(*(collect(folder) for folder in (off,on)))
    base, wide = runs
    versions = [[step["version"] for step in run["steps"]] for run in runs]
    start = versions[0][0]
    if versions[0] != versions[1] or versions[0] != list(range(start, start + len(versions[0]))):
        raise ValueError("round versions differ or are not consecutive")
    if start:
        paths = [Path(cfg["state_dir"])/f"v{start}" for cfg in cfgs]
        if not all((p/"training"/".metadata").is_file() for p in paths) or not paths[0].samefile(paths[1]):
            raise ValueError(f"cannot verify a common starting checkpoint for resumed version {start}")
    if placements[0] != placements[1]:
        raise ValueError("physical GPU placements differ")
    meshes = []
    for expected, cfg in zip((False, True), cfgs, strict=True):
        for key in ("state_dir","log_dir","export_dir","run_name","dynamopd_run"):
            cfg.pop(key, None)
        mesh = cfg.pop("rollout_tp_expansion", None)
        meshes.append(mesh)
        policy = cfg.get("policy", "cost_search")
        if policy in ("merge", "elastic_tp"):
            enabled = bool(mesh) and policy == "elastic_tp"
            cfg["policy"] = "merge"
        elif policy == "cost_search":
            options = dict(cfg.get("policy_options", {}))
            allow_wide = options.pop("allow_tp_expansion", True)
            enabled = bool(mesh) and allow_wide
            cfg.update(policy=policy, policy_options=options)
        else:
            raise ValueError(f"{policy} cannot isolate TP expansion from other policy changes")
        if enabled != expected:
            raise ValueError("TP comparison requires an expansion-disabled run followed by an expansion-enabled run")
    if cfgs[0]!=cfgs[1] or (all(meshes) and meshes[0]!=meshes[1]):
        raise ValueError("non-TP training options or configured TP meshes differ")
    if all(manifests):
        for field in ("source_sha256","dataset_sha256","parallel_layout","batch","rounds","cap","devices","sglang_options"):
            if manifests[0].get(field) != manifests[1].get(field):
                raise ValueError(f"run manifest mismatch for {field}")
    pairs = []
    for small,large in zip(base["steps"],wide["steps"],strict=True):
        lookup = {(c["rid"],c["position"]):c for c in small["single_chunks"] if not c["wide"]}
        matched = [(c,lookup.get((c["rid"],c["position"]))) for c in large["single_chunks"] if c["wide"]]
        matched = [(a,b) for a,b in matched if b and b["tokens"]==a["tokens"]]
        v = large["version"]
        (off_ids, left), (on_ids, right) = [samples(job, step) for job, step in zip(jobs, (small, large), strict=True)]
        if off_ids != on_ids:
            raise ValueError(f"request IDs, prompts, teacher routes, seeds or response limits differ at version {v}")
        for run in runs:
            run["questions"].append(len(left))
        exact = []
        identical_continuation = []
        for a,b in matched:
            rid,n = a["rid"],a["position"]
            if left[rid][:n]==right[rid][:n]:
                exact.append((a,b))
                end = n+a["tokens"]
                if left[rid][n:end]==right[rid][n:end]:
                    identical_continuation.append((a,b))
        def total(p):
            return dict(chunks=len(p),tokens=sum(a["tokens"] for a,b in p),narrow_s=sum(b["s"] for a,b in p),
                        wide_s=sum(a["s"] for a,b in p))
        identical_responses = sum(left[rid]==right[rid] for rid in left)
        pairs.append(dict(version=v,identical_responses=identical_responses,requests=len(left),
                          at_same_request_and_position=total(matched),same_input_tokens=total(exact),
                          same_input_and_output_tokens=total(identical_continuation)))
    for run in (base,wide):
        for step in run["steps"]:
            del step["single_chunks"]
    def rate(run,interval):
        rows = run["steps"][interval]
        return sum(r["tokens"] for r in rows)/sum(r["round_s"] for r in rows)
    middle = slice(1,-1)
    middle_times = [sum(s["round_s"] for s in run["steps"][middle]) for run in (base,wide)]
    has_middle = len(base["steps"]) > 2
    times = [sum(s["training_s"] for s in run["steps"]) for run in (base,wide)]
    throughputs = [sum(s["tokens"] for s in run["steps"])/t for run,t in zip((base,wide),times)]
    return dict(off=base,on=wide,paired_decode=pairs,
                training_time_speedup=times[0]/times[1],training_effective_token_speedup=throughputs[1]/throughputs[0],
                middle_time_speedup=middle_times[0]/middle_times[1] if has_middle else None,
                middle_effective_token_speedup=rate(wide,middle)/rate(base,middle) if has_middle else None,
                all_step_time_speedup=sum(s["round_s"] for s in base["steps"])/sum(s["round_s"] for s in wide["steps"]),
                all_step_effective_token_speedup=rate(wide,slice(None))/rate(base,slice(None)))


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("off",type=Path)
    parser.add_argument("on",type=Path)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    args.out.write_text(json.dumps(compare(args.off,args.on),indent=2,allow_nan=False)+"\n")
