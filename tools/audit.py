"""Audit committed counters, exclusive group intervals and measured concurrent work."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path


def overlap(spans, *operations, min_active=1):
    events = defaultdict(lambda:Counter())
    for x in spans:
        if x["op"] in operations:
            events[x["start"]][x["op"]] += 1
            events[x["end"]][x["op"]] -= 1
    active, prev, duration = Counter(), 0., 0.
    for t, delta in sorted(events.items()):
        if all(active[op] >= min_active for op in operations):
            duration += t-prev
        active.update(delta)
        prev = t
    return duration


def audit(folder):
    lines = [json.loads(x) for x in (folder/"events.jsonl").read_text().splitlines()]
    cfg = next(x for x in lines if x["event"]=="config")
    groups = cfg["config"]["groups"]
    output = []
    rollout_pids = {}
    teacher_contexts = {}
    optimizer_owner = None
    eligible = {"eos", "length"} if cfg["config"].get("train_on_length", False) else {"eos"}
    for init in (x for x in lines if x["event"]=="round_start"):
        version = init["version"]
        events = [x for x in lines if x.get("version")==version]
        spans = [x for x in events if x["event"]=="complete"]
        pending, busy, generated, delivered, reason = {}, {}, Counter(), Counter(), {}
        last_end = {}
        owners = ({r["rid"]:r["owner"] for r in init["requests"]} if "requests" in init
                  else {f"v{version}/r{i}":i % groups for i in range(init["samples"])})
        train_claims = set()
        tokens_by_group = Counter()
        parallel_limit = cfg["config"].get("max_train_groups", 1)
        tp_pair = None
        score_before_terminal = 0
        teachers = {role: dict(batches=0, score_group_s=0., delivered_tokens=0, cached_weight_wakes=0)
                    for role in cfg["config"]["teachers"]}
        batches, cache_hits = [], []
        for x in events:
            op, g = x.get("op"), x.get("gpu")
            if x["event"]=="prefix":
                rid = x["rid"]
                assert owners[rid]==g and rid not in reason and generated[rid]==x["start"] < x["end"]
                generated[rid] = x["end"]
                if x["finish"]:
                    reason[rid] = x["finish"]
            if x["event"]=="group_drained" and x["cause"]=="migration":
                for rid in x["ids"]:
                    assert owners[rid]==g and rid not in reason
                    owners[rid] = x["new_owner"]
            if x["event"]=="supervision":
                rid = x["rid"]
                assert delivered[rid]==x["start"] < x["end"] <= generated[rid]
                delivered[rid] = x["end"]
                score_before_terminal += int(rid not in reason)
                cache_hits.append(x["reused_kv_tokens"])
                teachers[x["teacher"]]["delivered_tokens"] += x["end"] - x["start"]
            if x["event"]=="dispatch":
                leases = [g]+([x["peer"]] if x["peer"] is not None else [])
                if tp_pair and set(leases).intersection(tp_pair):
                    assert tuple(leases) == tp_pair and op in ("decode", "tp_shrink")
                if op=="tp_expand":
                    assert tp_pair is None and len(leases)==2 and len(x["ids"])==1
                    assert owners[x["ids"][0]]==g and len(owners)-len(reason)==1
                if op=="tp_shrink" or (op=="decode" and len(leases)==2):
                    assert tuple(leases)==tp_pair
                assert all(i not in busy for i in leases)
                for i in leases:
                    busy[i] = op
                pending[g] = (op, leases)
                if op=="decode":
                    assert all(owners[rid]==g and rid not in reason for rid in x["ids"])
                if op=="train":
                    if parallel_limit == 1:
                        assert optimizer_owner in (None,g)
                    for rid in x["ids"]:
                        assert reason.get(rid) in eligible and delivered[rid]==generated[rid] and rid not in train_claims
                        train_claims.add(rid)
                        tokens_by_group[g] += generated[rid]
                    assert len(tokens_by_group) <= parallel_limit
                if op=="score":
                    batches.append(len(x["ids"]))
            if x["event"]=="complete":
                if op == "score":
                    teacher = teachers[x["role"]]
                    teacher["batches"] += 1
                    teacher["score_group_s"] += x["end"] - x["start"]
                if op == "switch" and x.get("teacher_weights_reused"):
                    teachers[x["new"]]["cached_weight_wakes"] += 1
                prior, leases = pending.pop(g)
                if op=="tp_expand":
                    tp_pair = tuple(leases) if x.get("accepted", True) else None
                    assert x["to_tp"] == len(leases)*cfg["config"]["group_size"] and x["to_tp"] > x["from_tp"]
                    if cfg["backend"]=="sglang_fsdp2" and tp_pair:
                        assert x["engine_weight_version"] == f"v{version}" and x["handoff_logp_error"] <= cfg["config"].get("tp_logprob_tolerance",.25)
                if op=="tp_shrink":
                    assert x["from_tp"] == len(leases)*cfg["config"]["group_size"] and x["from_tp"]>x["to_tp"] and x["peer_kv_used_blocks"]==0
                    tp_pair = None
                assert prior==op and x["end"]>=x["start"]
                for i in leases:
                    assert busy.pop(i)==op
                    assert last_end.get(i,0.) <= x["start"]
                    last_end[i] = x["end"]
                if cfg["backend"]=="sglang_fsdp2" and op=="move":
                    clocks=[x["start"],x["export_start"],x["export_end"],x["install_start"],x["install_end"]]
                    if x["accepted"]:
                        clocks.extend([x["release_start"],x["release_end"]])
                    clocks.append(x["end"])
                    assert clocks==sorted(clocks)
        assert not busy and not pending and tp_pair is None
        if cfg["config"].get("phase_barriers"):
            first_score = min(x["start"] for x in spans if x["op"] == "score")
            last_score = max(x["end"] for x in spans if x["op"] == "score")
            assert max(x["end"] for x in spans if x["op"] == "decode") <= first_score
            assert last_score <= min(x["start"] for x in spans if x["op"] == "train")
        commits = [x for x in spans if x["op"]=="commit"]
        assert len(commits)==1 and commits[0]["published"]==version+1
        durable = commits[0].get("durable",True)
        assert optimizer_owner in (None, commits[0]["gpu"])
        optimizer_owner = commits[0]["gpu"] if parallel_limit > 1 or not durable else None
        n = sum(generated[rid] for rid in train_claims)
        assert len(set(tokens_by_group) | {commits[0]["gpu"]}) <= parallel_limit
        reductions = [x for x in spans if x["op"] == "reduce_grads"]
        if set(tokens_by_group) != {commits[0]["gpu"]}:
            assert parallel_limit > 1
            assert {x["gpu"] for x in reductions} == set(range(groups)) and len(reductions) == groups
            last_backward = max(x["end"] for x in spans if x["op"] == "train")
            assert all(last_backward <= x["start"] <= x["end"] <= commits[0]["start"]
                       and x["owner"] == commits[0]["gpu"] and x["global_tokens"] == n
                       and x["group_tokens"] == tokens_by_group[x["gpu"]] for x in reductions)
        else:
            assert not reductions
        assert len(reason)==init["samples"] and n==commits[0]["tokens"] and n>0
        assert set(r for r,v in reason.items() if v in eligible)==train_claims
        assert {x["gpu"] for x in spans if x["op"]=="decode"}==set(range(groups))
        barrier = next(x for x in events if x["event"]=="rollout_barrier")
        if cfg["backend"]=="sglang_fsdp2":
            width = cfg["config"]["group_size"]
            replicas = width // cfg["config"].get("inference_tp_size", width)
            for x in (event for event in spans if event["op"] == "switch"):
                if "active_teacher_gpu_pids" in x:
                    pids = tuple(x["inference_pids"])
                    assert rollout_pids.setdefault(x["gpu"], pids) == pids and len(pids) == replicas
                    active = set(x["active_teacher_gpu_pids"])
                    assert len(active) == len(x["active_teacher_gpu_pids"]) and active.isdisjoint(pids)
                    assert not x["rollout_engine_restarted"] and not x["cached_teacher_weights"]
                    assert len(active) in (0, width)
                    if active:
                        assert x["new"] in cfg["config"]["teachers"]
                        assert x["engine_weight_version"] == x["new"]
                    assert x["live_inference_engines"] == replicas * (2 if active else 1)
                    if x.get("released_teacher_nvml_pids"):
                        assert x["release_nvml_checked"]
                        assert set(x["released_teacher_nvml_pids"]) == (teacher_contexts.pop(x["gpu"])
                                                                        | set(x.get("late_teacher_nvml_pids", [])))
                    if x.get("teacher_nvml_pids"):
                        assert x["gpu"] not in teacher_contexts
                        assert len(x["teacher_nvml_pids"]) == width and all(x["teacher_nvml_pids"])
                        teacher_contexts[x["gpu"]] = {pid for lane in x["teacher_nvml_pids"] for pid in lane}
                    if x.get("standby_cuda_contexts") is not None:
                        assert x["standby_cuda_contexts"] == [0] * width
                    if "released_teacher_pids" in x:
                        assert len(set(x["released_teacher_pids"])) == width
                        assert x["teacher_gpu_ranks_after_release"] == 0
                        assert x["teacher_rank_release_s"] >= 0
                elif cfg["config"].get("inference_layout") == "single":
                    assert x["live_inference_engines"] == replicas and x["inference_pid"] > 0
                    assert not x["cached_teacher_weights"]
            assert all(x["engine_weight_version"]==f"v{version}" for x in spans if x["op"]=="decode")
            assert commits[0]["fsdp_world"]==cfg["config"]["group_size"]
            trainer = next(x for x in spans if x["op"]=="switch" and x["new"]=="train")
            assert trainer["local_numel"] > 0
            if width == 1:
                assert trainer["global_numel"] == trainer["local_numel"]
            else:
                assert trainer["global_numel"] > trainer["local_numel"]
            assert sum(sum(x["rank_tokens"]) for x in spans if x["op"]=="train")==n
            published = [x for x in spans if x["op"]=="switch" and x["new"]=="rollout"
                         and x["engine_weight_version"]==f"v{version+1}"]
            assert len(published)==groups and {x["gpu"] for x in published}==set(range(groups))
            if cfg["config"].get("weight_sync") == "tensor":
                if parallel_limit > 1:
                    assert all(x["master_version"] == version + 1 and x["master_bytes"] > 0 for x in published)
                assert all(x["weight_sync"]=="tensor" and x["publish_source"]==commits[0]["gpu"] for x in published)
                manifests = {(x["weight_schema"], x["weight_buckets"], x["weight_bytes"]) for x in published}
                assert len(manifests)==1 and all(x["weight_buckets"]>0 and x["weight_bytes"]>0 for x in published)
                if not commits[0]["hf_checkpoint"]:
                    assert all(commits[0][k]==0 for k in ("dcp_write_s","gather_weights_s","hf_export_s"))
            assert barrier["loaded_version"]==version+1 and all(
                commits[0]["end"]<=x["start"]<=x["end"]<=barrier["t"] for x in published)
        end = next(x["t"] for x in events if x["event"]=="round_end")
        moves = [x for x in spans if x["op"]=="move" and x["accepted"]]
        resizes = [x for x in spans if x["op"] in ("tp_expand", "tp_shrink")]
        tensor_switches = [x for x in spans if x.get("weight_sync")=="tensor"]
        source = next((x for x in tensor_switches if x["gpu"]==commits[0]["gpu"]), {})
        output.append(dict(version=version, eos=sum(v=="eos" for v in reason.values()),
                           training_tokens_by_group=dict(tokens_by_group),
                           gradient_reduction_s=overlap(spans, "reduce_grads"),
                           parallel_backward_s=overlap(spans, "train", min_active=2),
                           last_decode_to_round_end_s=end-max(x["end"] for x in spans if x["op"] == "decode"),
                           master_sync_longest_group_s=max((x["replica_sync_s"] for x in spans
                                                            if "replica_sync_s" in x), default=None),
                           truncated=sum(v=="length" for v in reason.values()), trained=len(train_claims),
                           round_s=end-init["t"], generated_tokens=sum(generated.values()), training_tokens=n,
                           score_before_terminal=score_before_terminal, teacher_batches=len(batches),
                           teachers=teachers, largest_teacher_batch=max(batches), cache_hit_chunks=sum(bool(v) for v in cache_hits),
                           decode_score_overlap_s=overlap(spans,"decode","score"),
                           decode_train_overlap_s=overlap(spans,"decode","train"),
                           decode_score_train_overlap_s=overlap(spans,"decode","score","train"),
                           score_train_overlap_s=overlap(spans,"score","train"),
                           migration_count=len(moves), migration_s=sum(x["end"]-x["start"] for x in moves),
                           tp_expansions=sum(x["op"]=="tp_expand" and x.get("accepted", True) for x in resizes),
                           tp_resize_s=sum(x["end"]-x["start"] for x in resizes),
                           tp_kv_transfer_bytes=sum(x["kv_transfer_bytes"] for x in resizes),
                           wide_decode_s=sum(x["end"]-x["start"] for x in spans if x["op"]=="decode" and x.get("peer") is not None),
                           commit_s=commits[0]["end"]-commits[0]["start"],
                           optimizer_and_probe_s=commits[0].get("optimizer_and_probe_s"),
                           dcp_write_s=commits[0].get("dcp_write_s"),
                           checkpoint_gather_s=commits[0].get("gather_weights_s"),
                           hf_export_s=commits[0].get("hf_export_s"),
                           trainer_state_device=source.get("trainer_state_device"),
                           source_release_s=source.get("unload_s"),
                           source_tensor_publish_s=source.get("tensor_publish_s"),
                           tensor_publish_longest_group_s=max((x["tensor_publish_s"] for x in tensor_switches), default=None),
                           commit_to_barrier_s=barrier["t"]-commits[0]["start"],
                           post_commit_to_barrier_s=barrier["t"]-commits[0]["end"],
                           published=commits[0]["published"],durable=durable,
                           hf_checkpoint=commits[0].get("hf_checkpoint", durable)))
    assert len(output)==cfg["config"]["rounds"]
    assert output[-1]["hf_checkpoint"]
    if "training" in cfg["config"].get("checkpoint_contents", ["hf_model","training"]):
        assert output[-1]["durable"]
    assert not teacher_contexts, "a leased teacher context lacks a matching release"
    return dict(backend=cfg["backend"], policy=cfg["policy"], rounds=output)


if __name__=="__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder",type=Path)
    p.add_argument("--output",type=Path)
    args = p.parse_args()
    result = json.dumps(audit(args.folder), indent=2)+"\n"
    if args.output:
        args.output.write_text(result)
    print(result, end="")
