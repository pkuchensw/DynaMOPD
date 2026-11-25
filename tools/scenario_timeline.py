"""Build a self-contained timeline from the five-scenario benchmark's existing traces."""
import argparse
from collections import Counter, defaultdict
import json
import os
from statistics import median
from pathlib import Path
import re


CASES = {
    "control": ("DAPO 对照", "7B-Instruct 学生，三路 7B 教师，回答上限 8192。"),
    "teacher14b": ("14B 教师", "仅 general 路由改为 14B。此轮使用当时的引擎重启路径，早于异构教师换岗优化。"),
    "cap16k": ("16K 回答上限", "回答上限 16384，每个训练 rank 的 token 预算同步提高到 18432。"),
    "gsm8k": ("GSM8K", "换用 GSM8K main/train，保留教师路由和 8192 回答上限。"),
    "math-student": ("Math-7B 学生", "使用 Qwen2.5-Math-7B 学生及其原生模板和 EOS。"),
}
NAMES = {"decode": "学生生成", "score": "教师评分", "train": "学生训练", "reduce_grads": "梯度归约",
         "tp_expand":"TP 扩容", "tp_shrink":"TP 归还", "commit": "提交更新", "move": "长尾合并", "switch": "角色切换", "publish": "发布权重"}
ROLE = {"train": "trainer", "rollout": "student"}
def read(p):
    return json.loads(p.read_text())

def lines(p):
    with p.open() as f:
        yield from map(json.loads, f)

def stamp(t):
    return round(t, 6)

def span(a, b, kind, gpus, label, detail="", seq=None):
    return dict(a=stamp(a), b=stamp(b), k=kind, g=list(gpus), label=label, detail=detail, seq=seq)

def triple(spans):
    changes = defaultdict(Counter)
    for s in spans:
        if s["k"] in ("decode", "score", "train") and s["g"]:
            changes[s["a"]][s["k"]] += 1
            changes[s["b"]][s["k"]] -= 1
    count = Counter()
    intervals = []
    previous = 0
    for t, delta in sorted(changes.items()):
        if all(count[k] for k in ("decode", "score", "train")) and t > previous:
            if intervals and previous == intervals[-1][1]:
                intervals[-1][1] = t
            else:
                intervals.append([previous, t])
        count.update(delta)
        previous = t
    return intervals

def dyna_round(rows, devices, expected):
    zero = next(x["t"] for x in rows if x["event"] == "round_start")
    inputs = {r["rid"]: r for r in next(x["requests"] for x in rows if x["event"] == "round_start")}
    s = {"duration": expected, "spans": [], "actions": [], "early": 0}
    pending = {}
    for i, e in enumerate(rows):
        event = e["event"]
        g = e.get("gpu")
        if event == "dispatch":
            if g in pending:
                raise ValueError("two unacknowledged actions on the same group")
            pending[g] = e
        if event == "round_end" and abs(e["t"] - zero - expected) > .001:
            raise ValueError("round duration disagrees with reported result")
        if event in ("migration_planned", "group_drained", "rollout_barrier"):
            title = {"migration_planned":"预订长尾合并", "group_drained":"生成组排空",
                     "rollout_barrier":"新权重全组就绪"}[event]
            who = list(devices[g]) if g is not None else sum(devices.values(), [])
            note = (f"G{g} → G{e['peer']}，等待安全点" if event == "migration_planned" else
                    (("自然排空" if e["cause"] == "natural" else f"迁移后释放，目标 G{e['new_owner']}")
                     if event == "group_drained" else f"{e['groups']} 个组确认 student v{e['loaded_version']}"))
            point = span(e["t"]-zero, e["t"]-zero, "mark", who, title, note, e["seq"])
            point["decision"] = e.get("decision_id")
            s["actions"].append(point)
        if event != "complete":
            continue
        d = pending.pop(g)
        op, peer = e["op"], e["peer"]
        assert d["op"] == op and d["peer"] == peer
        a, b = e["start"]-zero, e["end"]-zero
        ids = d["ids"]
        kind, name, info = op, NAMES[op], []
        who = devices[g][:] + (devices[peer] if peer is not None else [])
        active = who[:]
        add = s["spans"].append
        if op in ("score", "decode"):
            receipt = rows[i+1:i+1+len(ids)]
            assert all(x["event"] == ("prefix" if op=="decode" else "supervision") for x in receipt)
            assert set(ids) == {x["rid"] for x in receipt}
            nt = sum(x["end"]-x["start"] for x in receipt)
            info.append(f"{len(ids)} 条轨迹 · {nt:,} 个新增" + ("生成 token" if op=="decode" else "监督 token"))
            if op == "decode":
                counts = e["replica_requests"]
                tp = e["inference_tp_size"]
                assert len(counts)*tp == len(who) and sum(counts) == len(ids)
                if peer is not None:
                    name += f" · TP{tp}"
                info.append(f"student {e['engine_weight_version']} · TP{tp} 副本请求数 {counts}")
                counts = [count for count in counts for _ in range(tp)]
            else:
                name += " · " + e["role"]
                counts = [0]*len(who)
                for rid in ids:
                    counts[inputs[rid]["seed"] // len(devices) % len(who)] += 1
                early = sum(x["before_eos"] for x in receipt)
                s["early"] += early
                info.append(f"{early} 个区间在轨迹终止前交付 · {sum(x['ready'] for x in receipt)} 条样本就绪")
            active = [dev for dev, count in zip(who, counts, strict=True) if count]
        elif op in ("tp_expand","tp_shrink"):
            name = f"TP{e['from_tp']} → TP{e['to_tp']}"
            if not e.get("accepted",True):
                name += "（容量拒绝）"
            kind = "switch"
            info.append(f"借用 G{peer}，原角色 {e['donor_role']}")
            if "wake_s" in e:
                info.append(f"恢复 {e['wake_s']:.3f} s，重算 {e['rebuilt_tokens']:,} 个前缀 token 用时 {e['prefill_s']:.3f} s")
            if "sleep_s" in e:
                info.append(f"TP{e['from_tp']} 休眠及释放耗时 {e['sleep_s']:.3f} s")
        elif op == "move":
            name = f"G{g} → G{peer} " + ("合并完成" if e["accepted"] else "合并被拒")
            kind = "move" if e["accepted"] else "reject"
            if e["accepted"]:
                info.append(f"{len(ids)} 条请求 · 重建 {e['rebuilt_tokens']:,} 个 KV token · 实传 KV {e['kv_transfer_bytes']:,} B")
            else:
                info.append(f"需要 {e.get('required_kv_tokens',0):,} 个 KV token · 可用容量 {e.get('capacity_tokens',0):,}")
            for stage, title in (("export","导出采样状态"),("install","目标安装/容量检查"),("release","源交还")):
                if stage+"_end" in e:
                    info.append(f"{title} {e[stage+'_end']-e[stage+'_start']:.4f} s")
        elif op == "switch":
            publish = d.get("load_version") is not None and e["new"] == "rollout"
            kind = "publish" if publish else "switch"
            name = ROLE.get(e["old"], e["old"])+" → "+ROLE.get(e["new"], e["new"])
            if publish:
                name += f" · v{d['load_version']}"
            info += [f"卸载/交接 {e['unload_s']:.3f} s", f"加载/就绪 {e['load_s']:.3f} s"]
            if "teacher_rank_load_s" in e:
                info.append(f"教师接卡至就绪 {e['teacher_rank_load_s']:.3f} s，含等待 CPU 准备 {e['standby_wait_s']:.3f} s")
                info.append(f"待命 CUDA context {e['standby_cuda_contexts']} · 各卡实际 KV 容量 {e['teacher_kv_capacity']} token")
                if e.get("reused_host_parameter_bytes"):
                    info.append(f"复用同版本主机参数分片 {e['reused_host_parameter_bytes']/1e9:.2f} GB")
            if "teacher_rank_release_s" in e:
                verified = "宿主机 PID 已经 NVML 核验退出" if e["release_nvml_checked"] else "缺少 NVML 核验"
                info.append(f"教师退出 {e['teacher_rank_release_s']:.3f} s · {verified}")
            if e.get("engine_restarted"):
                info.append("本次重启推理引擎")
                if "previous_inference_pid" in e and "inference_pid" in e:
                    info.append(f"引擎 PID {e['previous_inference_pid']} → {e['inference_pid']}")
            if "tensor_publish_s" in e:
                info.append(f"其中张量发布 {e['tensor_publish_s']:.3f} s")
            if e.get("wide_publish_s"):
                info.append(f"宽 TP 权重发布及等待回执 {e['wide_publish_s']:.3f} s")
            if "free_before" in e:
                info.append(f"组内最少空闲显存 {e['free_before']/2**30:.1f} → {e['free_after_release']/2**30:.1f} GiB（释放后）")
        elif op == "train":
            info += [f"{len(ids)} 条就绪样本 · {e['tokens']:,} 个有效 token", f"各 rank 有效 token {e['rank_tokens']}"]
        elif op == "reduce_grads":
            info.append(f"本组 {e['group_tokens']:,} 个有效 token · 汇至 G{e['owner']} · 全批 {e['global_tokens']:,}")
        elif op == "commit":
            info.append(f"全批 {e['tokens']:,} 个有效 token · 一次优化器更新 → v{e['published']}")
            if e.get("hf_checkpoint"):
                info.append(f"含 HF 导出 {e['hf_export_s']:.2f} s")
        note = "；".join(info)
        action = span(a,b,kind,who,name,note,e["seq"])
        action.update(decision=d["decision_id"], sent=stamp(d["t"]-zero), restart=e.get("engine_restarted",False))
        if "teacher_rank_load_s" in e:
            action.update(teacherLoad=e["teacher_rank_load_s"], unload=e["unload_s"], teacher=e["new"])
        if "teacher_rank_release_s" in e:
            action["teacherExit"] = e["teacher_rank_release_s"]
        s["actions"].append(action)
        if op == "switch":
            cut = min(b,a+e["unload_s"])
            add(span(a,cut,"unload",who,"卸载 / 交接 · "+name,note,e["seq"]))
            add(span(cut,b,kind,who,("恢复并发布 · " if publish else "加载 / 就绪 · ")+name,note,e["seq"]))
        else:
            add(span(a,b,kind,active,name,note,e["seq"]))
    assert not pending
    s["actions"].sort(key=lambda x:(x["a"],x["seq"]))
    s["triple"] = triple(s["spans"])
    return s

def read_workers(job):
    result = []
    for file in sorted((job/"events").glob("events_*.jsonl")):
        opened = {}
        for event in lines(file):
            key = event["id"]
            if event["event"] == "start":
                opened[key] = event
            else:
                first = opened.pop(key)
                if not event.get("ok") or event["wall_ns"] < first["wall_ns"]:
                    raise ValueError(f"invalid interval in {file}")
                result.append((first["wall_ns"],event["wall_ns"],first))
        if opened:
            raise ValueError(f"incomplete calls in {file}")
    return sorted(result,key=lambda item:item[0])

def kdflow_rollout(raw, zero, until, devices):
    """Use measured request lifetimes and replay client order through a round-robin pool."""
    batch_end, = [b for a,b,e in raw if a == zero and e["method"] == "RolloutManager.rollout"]
    batch = [(a,b,e) for a,b,e in raw if zero <= a <= b <= batch_end]
    generation, = [(a,b) for a,b,e in batch if e["phase"] == "rollout_generate"]
    assert batch_end <= until
    wake, sleep = set(), set()
    requests = []
    changes = Counter()
    for a,b,e in batch:
        if e["phase"] == "rollout_request":
            assert generation[0] <= a < b <= generation[1]
            requests.append((a,b))
            changes[a] += 1
            changes[b] -= 1
        elif e["phase"] in ("rollout_load", "rollout_offload"):
            visible = list(map(int, e["cuda_visible"].split(",")))
            gpu = visible[0] if len(visible)==1 else visible[e["base_gpu_id"]]
            if e["phase"] == "rollout_load" and b <= generation[0]:
                wake.add(gpu)
            if e["phase"] == "rollout_offload" and a >= generation[1]:
                sleep.add(gpu)
    assert wake == sleep == set(devices)
    counts = []
    active = 0
    for t, delta in sorted(changes.items()):
        if delta:
            active += delta
            assert active >= 0
            counts.append([(t-zero)/1e9,active])
    assert counts and counts[-1][1] == 0
    start, end = counts[0][0], counts[-1][0]
    tail = end-counts[-2][0] if counts[-2][1] == 1 else 0.
    # Only the final contiguous sub-capacity period counts as the tail.
    last_full = max((i for i,(_,n) in enumerate(counts) if n >= len(devices)),default=-1)
    tail_start = counts[last_full+1][0]
    # Neither HTTP arrival order nor the worker ID is present in these recordings.
    # A–H are logical slots in ONE reconstruction, independent of physical IDs.
    lanes = [[] for _ in devices]
    for i,(a,b) in enumerate(requests):
        a,b = (a-zero)/1e9,(b-zero)/1e9
        lane = lanes[i % len(lanes)]
        if lane and a <= lane[-1][1]:
            lane[-1][1] = max(lane[-1][1],b)
            lane[-1][2] += 1
        else:
            lane.append([a,b,1])
    spans = []
    for logical,intervals in enumerate(lanes):
        cursor = start
        for a,b,n in intervals:
            if cursor < a:
                spans.append(span(cursor,a,"replica_wait",[logical],"等待","轮询重建中，此副本没有在途生成请求。"))
            spans.append(span(a,b,"replica_decode",[logical],"推理",
                              f"轮询重建中，{n} 个生成请求的在途区间合并；包含排队和返回。"))
            cursor = b
        if cursor < end:
            spans.append(span(cursor,end,"replica_wait",[logical],"等待","轮询重建中，此副本已完成本步生成，正在等其他副本。"))
    return dict(inflight=counts, single_request_tail_s=tail, tail_start=tail_start,
                start=start,end=end, replicas=len(devices),replica_spans=spans)

def baseline_rounds(job, system, cfg, stats):
    raw = read_workers(job)
    anchor = "RolloutManager.rollout" if system == "kdflow" else "AgentLoopManager.generate_sequences"
    starts = [a for a,b,e in raw if e["method"] == anchor]
    assert len(starts) == len(stats["all_step_s"]) == 4
    models = {Path(p).name:route for route,p in cfg["teachers"].items()}
    result = []
    for i, (zero, duration) in enumerate(zip(starts,stats["all_step_s"],strict=True)):
        until = starts[i+1] if i+1<len(starts) else float("inf")
        bars, requests = [], defaultdict(list)
        counts = Counter()
        for begin,end,e in raw:
            if not zero <= begin < min(until,zero+duration*1e9):
                continue
            a,b = (begin-zero)/1e9,(end-zero)/1e9
            ph = e["phase"]
            devs = list(map(int,e["cuda_visible"].split(",")))
            gpu = devs[0] if len(devs)==1 else (devs[e["base_gpu_id"]] if "base_gpu_id" in e else None)
            route = ""
            model = e.get("teacher_name_or_path", e.get("model",""))
            if model:
                route = models.get(re.sub(r"-[0-9a-f]{16}$","",Path(model).name),"")
            if ph in ("rollout_request","inference_request"):
                k = "score" if e.get("rollout_mode") == "RolloutMode.COLOCATED" else "decode"
                lane = 8 if ph=="rollout_request" else gpu
                assert lane is not None and (k=="decode" or route)
                requests[lane,k,route if k=="score" else "",e["method"]].append((a,b))
                counts[k] += 1
                continue
            if ph in ("teacher_load","teacher_offload"):
                gpu = 8
            phases = {"teacher_forward_worker":("score","教师评分 · "+route),
                      "teacher_load":("switch","唤醒教师 · "+route), "teacher_offload":("unload","休眠教师 · "+route),
                      "student_train":("train","学生训练"),"student_logprob":("logprob","重算 old logprob"),
                      "weight_sync_worker":("publish","发布 student 权重"),"rollout_offload":("unload","释放 rollout"),
                      "inference_offload":("unload","释放推理引擎"),"rollout_load":("switch","唤醒 rollout"),
                      "student_offload":("unload","训练状态卸载"),"student_load":("switch","训练状态上卡"),
                      "student_state_move":(("unload" if e.get("device")=="cpu" else "switch"),"训练状态 → "+str(e.get("device","")))}
            if ph not in phases:
                continue
            assert gpu is not None
            k,label = phases[ph]
            bars.append(span(a,b,k,[gpu],label,e["method"]+" · "+("卡池调用" if gpu==8 else "worker 调用")))
        for (gpu,k,route,method),intervals in requests.items():
            # Exact union of concurrent requests. Preserve all unobserved gaps.
            merged=[]
            for a,b in intervals:
                if merged and a<=merged[-1][1]:
                    merged[-1][1]=max(b,merged[-1][1]); merged[-1][2]+=1
                else:
                    merged.append([a,b,1])
            for a,b,count in merged:
                bars.append(span(a,b,k,[gpu],NAMES[k]+(" · "+route if route else ""),
                                 f"{method} · {count} 个有交集或连续的请求 · 含排队/通信"))
        assert counts["decode"] == cfg["batch_size"]
        if system == "verl":
            assert counts["score"] == cfg["batch_size"]
        if not bars or any(v["b"]>duration+.02 for v in bars):
            raise ValueError(f"{job} step {i+1}: worker time exceeds step timer")
        step = dict(duration=duration)
        if system == "kdflow":
            step["router"] = kdflow_rollout(raw, zero, until, stats["layout"]["devices"])
        step["spans"] = sorted(bars,key=lambda s:s["a"])
        result.append(step)
    return result

def measured_case(root, override):
    report_path = root / override["report"]
    report = read(report_path)
    baseline = root / override["baselines"]
    trace = report_path.parent / report["systems"]["dynamopd"]["trace"]
    jobs = {"dynamopd": trace.parent, "kdflow": baseline/"kdflow", "verl": baseline/"standalone-verl/verl"}
    stats = {}
    for system, job in jobs.items():
        manifest = read(job/"run.json")
        row = report["systems"][system]
        if manifest["exit_code"] or row["verified_questions_per_step"] != [manifest["batch"]]*4:
            raise ValueError(f"incomplete training data for {job}")
        stats[system] = dict(steps=report["counted_steps"], batch=manifest["batch"], steady_s=row["middle_s"],
                             tokens_s=row["middle_tokens_per_s"], all_step_s=row["round_s"],
                             all_effective_tokens=row["effective_training_tokens"], layout=manifest["parallel_layout"])
    if "tp_measurement" in report:
        metrics = dict(kind="tail_tp", date=override["date"], **report["tp_measurement"])
        paths = [("四步训练与 TP 测量汇总",report_path),("DynaMOPD 训练事件",trace),
                 ("TP1 → TP4 热交接逐次读数",report_path.parent/"live.json"),
                 ("TP1 → TP4 冷切逐次读数",report_path.parent/"cold/engine.json"),
                 ("首次尝试的评分 OOM",report_path.parent/report["failed_attempt"]),
                 ("复测启动参数",report_path.parent/"train.sh"),
                 ("KDFlow 原始事件",jobs["kdflow"]/"events"),("verl 原始事件",jobs["verl"]/"events")]
        if "elastic" in override:
            experiment_path = root/override["elastic"]
            exp = read(experiment_path)
            for mode in ("off","on"):
                paths.append((f"TP 扩容 {mode} 训练事件", root/override[mode]/"events.jsonl"))
            paths.append(("同配置 TP 开关对照",experiment_path))
            quality_path = experiment_path.parent/"heldout-kl.json"
            if quality_path.is_file():
                paths.append(("两组最终学生在同一批留出前缀上的 KL",quality_path))
                metrics["elastic_quality"] = {key:vals["mean_kl_nats"] for key,vals in read(quality_path).items()}
            paths[1] = ("早期独立交接实验对应的训练事件",trace)
            jobs["dynamopd"] = root/override["on"]
            current = exp["on"]
            ds = current["steps"]
            manifest = read(jobs["dynamopd"]/"run.json")
            if manifest["exit_code"] or current["questions"] != [manifest["batch"]]*4:
                raise ValueError("elastic training did not complete four audited rounds")
            times = [x["round_s"] for x in ds]
            tokens = [x["tokens"] for x in ds]
            stats["dynamopd"].update(all_step_s=times,all_effective_tokens=tokens,steady_s=sum(times[1:3])/2,
                                     tokens_s=sum(tokens[1:3])/sum(times[1:3]),layout=manifest["parallel_layout"])
            metrics.update(elastic=exp,default_step=max(ds,key=lambda r:r["wide_tokens"])["version"]+1)
        return jobs, stats, metrics, source_links(root, paths)
    leases = report["leases"]["dynamopd"]
    switches = [e["switch_s"] for e in leases]
    releases = report["leases"]["dynamopd_release_s"]
    ranks = report["paired_trainer_evictions"]
    worst_by_mode = [max(row["evict_s"] for rank in ranks for row in rank if row["mode"] == mode)
                     for mode in ("copy_on_release", "optimizer_mirror", "optimizer_and_parameters")]
    immediate = report["immediate_reactivation"][-1]
    systems = report["systems"]
    metrics = dict(date=override["date"], count=len(switches), shortest=min(switches), median=median(switches),
                   longest=max(switches), release=[min(releases),max(releases)], evict=worst_by_mode,
                   immediate=immediate["teacher_rank_load_s"], wait=immediate["standby_wait_s"],
                   quality=report["fixed_prefix_kl_nats"], previous=systems["previous_dynamopd"]["middle_s"],
                   optimizerOnly=systems["dynamopd_optimizer_mirror"]["middle_s"],
                   wall={name:systems[name]["wall_s"] for name in jobs})
    paths = [("逐步审计与消融实验",report_path),("DynaMOPD 原始事件",trace),
             ("KDFlow 事件目录",jobs["kdflow"]/"events"),("verl 事件目录",jobs["verl"]/"events"),
             ("14B 换岗压力测试",trace.parents[1]/"validation/immediate/timings.json"),
             ("最终学生留出 KL",trace.parents[1]/"validation/heldout-kl.json"),
             ("基线留出 KL",report_path.parent/"validation/heldout-kl.json")]
    return jobs, stats, metrics, source_links(root, paths)


def source_links(root, paths):
    for _, path in paths:
        if not path.exists():
            raise FileNotFoundError(path)
    return [(name,os.path.relpath(path, root)+("/" if path.is_dir() else "")) for name,path in paths]


def build(root):
    results=read(root/"results.json")
    report={}
    sources = root/"timeline-sources.json"
    overrides = read(sources) if sources.exists() else {}
    for case,(title,description) in CASES.items():
        folder=root/case
        item=dict(name=title,desc=description,systems={},summary={})
        if case in overrides:
            jobs, stats, item["measurement"], item["sources"] = measured_case(root, overrides[case])
            if case == "teacher14b":
                item["desc"] = "general 路由使用 Qwen2.5-14B。CPU 提前准备教师进程，在发布 student 权重后复制同版本参数分片和 AdamW 状态。"
            cfg = read(jobs["dynamopd"]/"launch-config.json")
        else:
            jobs = dict(dynamopd=folder/"dynamopd", kdflow=folder/"kdflow", verl=folder/"standalone-verl/verl")
            stats = results[case]
            cfg = read(folder/"config.json")
        records=list(lines(jobs["dynamopd"]/"events.jsonl"))
        placements=next(x["placements"] for x in records if x["event"]=="executor_ready")
        devices={x["group"]:list(map(int,x["devices"])) for x in placements}
        assert sorted(sum(devices.values(),[]))==list(range(8))
        for sys in ("dynamopd","kdflow","verl"):
            st=stats[sys]
            assert st["steps"]==[2,3] and st["batch"]==512
            item["summary"][sys]=dict(mean=st["steady_s"],tps=st["tokens_s"])
            if sys=="dynamopd":
                runs=[dyna_round([e for e in records if e.get("version")==v],devices,n)
                      for v,n in enumerate(st["all_step_s"])]
            else:
                runs=baseline_rounds(jobs[sys],sys,cfg,st)
            for run,nt in zip(runs,st["all_effective_tokens"],strict=True):
                run["tokens"]=nt
            item["systems"][sys]=runs
        report[case]=item
    return report

if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix",type=Path)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    data=json.dumps(build(args.matrix),ensure_ascii=False,separators=(",",":"),allow_nan=False).replace("<",r"\u003c")
    template = args.matrix / "scenario_timeline.template.html"
    if not template.is_file():
        template = Path(__file__).with_name("scenario_timeline.template.html")
    view = template.read_text()
    (args.out or args.matrix/"gpu-timeline.html").write_text(view.replace("__REPORT_DATA__",data))

