"""Compare two completed training runs that differ only in the elastic TP policy/profile."""
import argparse
import json
from pathlib import Path

from DynaMOPD.tools.compare_results import summarize


def read(path):
    return json.loads(path.read_text())


def events(path):
    with path.open() as f:
        return [json.loads(line) for line in f]


def collect(folder):
    job = folder/"dynamopd"
    stats = summarize(folder,training_only=True,systems=["dynamopd"])["dynamopd"]
    trace = events(job/"events.jsonl")
    config = read(folder/"config.json")
    base_tp = config.get("inference_tp_size", config.get("group_size"))
    rounds = []
    for v,duration in enumerate(stats["round_s"]):
        step = [e for e in trace if e.get("version")==v]
        requests = next(e for e in step if e["event"]=="round_start")["requests"]
        inputs = {r["rid"]:r for r in requests}
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
        source = next(e for e in publications if e["gpu"]==e["publish_source"])
        rounds.append(dict(version=v,round_s=duration,tokens=stats["effective_training_tokens"][v],
                           checkpoint_s=checkpoint_s,training_s=duration-checkpoint_s,
                           expansion=expansions,wide_chunks=len(wide),wide_tokens=sum(c["tokens"] for c in wide),
                           wide_decode_s=sum(c["s"] for c in wide),
                           release_s=sum(e["end"]-e["start"] for e in step if e.get("op")=="tp_shrink" and e["event"]=="complete"),
                           publisher_state_device=source.get("trainer_state_device"),
                           publisher_free_bytes_after=source["free_bytes"],
                           publisher_allocated_bytes=source.get("publisher_allocated_bytes"),
                           wide_publication_longest_group_s=max((e.get("wide_publish_s",0) for e in step if e["event"]=="complete"),default=0),
                           single_chunks=[{k:v for k,v in c.items() if k!="ids"} for c in chunks if "rid" in c]))
    return dict(manifest=read(job/"run.json"),steps=rounds,wall_s=stats["wall_s"],
                questions=stats["verified_questions_per_step"],bootstrap=next((e["meshes"] for e in trace if e["event"]=="wide_mesh_ready"),[]))


def compare(off,on):
    cfgs = [read(folder/"config.json") for folder in (off,on)]
    for cfg in cfgs:
        cfg.pop("rollout_tp_expansion",None)
        cfg.pop("policy",None)
    if cfgs[0]!=cfgs[1]:
        raise ValueError("non-TP benchmark options differ")
    base,wide = collect(off),collect(on)
    for field in ("dataset_sha256","parallel_layout","batch","rounds","cap","devices","sglang_options"):
        if base["manifest"][field]!=wide["manifest"][field]:
            raise ValueError(f"benchmark manifest mismatch for {field}")
    pairs = []
    for small,large in zip(base["steps"],wide["steps"],strict=True):
        lookup = {(c["rid"],c["position"]):c for c in small["single_chunks"] if not c["wide"]}
        matched = [(c,lookup.get((c["rid"],c["position"]))) for c in large["single_chunks"] if c["wide"]]
        matched = [(a,b) for a,b in matched if b and b["tokens"]==a["tokens"]]
        if not matched:
            continue
        v = large["version"]
        left,right = [{r["rid"]:r for r in events(folder/"dynamopd"/f"samples-v{v}.jsonl")} for folder in (off,on)]
        exact = []
        identical_continuation = []
        for a,b in matched:
            rid,n = a["rid"],a["position"]
            assert left[rid]["prompt"]==right[rid]["prompt"] and left[rid]["seed"]==right[rid]["seed"]
            if left[rid]["response"][:n]==right[rid]["response"][:n]:
                exact.append((a,b))
                end = n+a["tokens"]
                if left[rid]["response"][n:end]==right[rid]["response"][n:end]:
                    identical_continuation.append((a,b))
        def total(p):
            return dict(chunks=len(p),tokens=sum(a["tokens"] for a,b in p),narrow_s=sum(b["s"] for a,b in p),
                        wide_s=sum(a["s"] for a,b in p))
        identical_responses = sum(left[rid]["response"]==right[rid]["response"] for rid in left)
        pairs.append(dict(version=v,identical_responses=identical_responses,requests=len(left),
                          at_same_request_and_position=total(matched),same_input_tokens=total(exact),
                          same_input_and_output_tokens=total(identical_continuation)))
    for run in (base,wide):
        for step in run["steps"]:
            del step["single_chunks"]
        del run["manifest"]
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
