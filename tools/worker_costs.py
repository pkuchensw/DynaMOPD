"""Fit worker time from real receipts, then score later rounds without fitting on them."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median, quantiles

from ..backends.sim_profile import (SwitchState, decode_duration, decode_windows, rpc_decode_features, engine_classes,
                                    fit_line, fitted_switch, train_seconds, train_width)


def read_trace(folder):
    with (folder / "events.jsonl").open() as stream:
        header = json.loads(next(stream))
        if header.get("backend") != "sglang_fsdp2":
            raise ValueError("expected a native SGLang/FSDP2 trace")
        events = [json.loads(line) for line in stream]
    cfg = dict(header["config"])
    classes = engine_classes(cfg)
    class_ids = {value: index for index, value in enumerate(dict.fromkeys(classes.values()))}
    cfg["sim_engine_classes"] = {role: class_ids[value] for role, value in classes.items()}
    cfg["sim_teacher_lifecycle"] = ("lease" if any("rollout_engine_restarted" in e for e in events)
                                    else "restart")
    return cfg, events


def observations(cfg, events):
    classes = engine_classes(cfg)
    states = [SwitchState(cfg, classes) for _ in range(cfg["groups"])]
    pending = {}
    last_teacher_exit = {}
    for index, e in enumerate(events):
        kind = e["event"]
        if kind == "round_start":
            lengths = {r["rid"]: r["prompt_tokens"] for r in e["requests"]}
            seeds = {r["rid"]: r.get("seed") for r in e["requests"]}
            limits = {r["rid"]: r.get("max_new_tokens", cfg["max_new_tokens"]) for r in e["requests"]}
            prompts = lengths.copy()
            scored_context = {}
            owner = e["optimizer_owner"]
        elif kind == "prefix":
            lengths[e["rid"]] = prompts[e["rid"]] + e["end"]
        elif kind == "supervision":
            scored_context[e["gpu"], e["rid"]] = prompts[e["rid"]]+e["end"]
        elif kind == "dispatch":
            pending[e["gpu"]] = e
            if owner is None and e.get("next_role") == "train":
                owner = e["gpu"]
        elif kind == "complete":
            g, op = e["gpu"], e["op"]
            cmd = pending.pop(g)
            dt = e["end"] - e["start"]
            point = dict(v=e["version"], op=op, dt=dt)
            if op == "switch":
                old, new = cmd["role"], e["new"]
                pub = cmd.get("load_version") is not None
                keys = states[g].keys(old, new, pub, g == owner)
                release_key = states[g].release_key(old, pub, g == owner)
                states[g].advance(new, e.get("trainer_state_device"))
                point.update(keys=keys, release_key=release_key, old=old, new=new, pub=pub, unload=e["unload_s"], load=e["load_s"],
                             sync=e.get("replica_sync_s", 0.), device=e.get("trainer_state_device"),
                             transfer=e.get("tensor_publish_s"))
                if "teacher_rank_release_s" in e:
                    last_teacher_exit[g] = e["start"] + e["teacher_rank_release_s"]
                if "teacher_rank_load_s" in e:
                    point["activation_s"] = max(0., e["teacher_rank_load_s"]-e.get("standby_wait_s", 0.))
                ranks = e.get("teacher_rank_timings", [])
                if ranks and all("prepared_import_s" in r for r in ranks):
                    measured = e.get("teacher_prepare_s")
                    point["prepare_s"] = max(r["prepared_import_s"] for r in ranks) if measured is None else measured
                    point["wait_s"] = e["standby_wait_s"]
                    point["load"] = max(0., point["load"] - point["wait_s"])
                    point["prepare_elapsed"] = max(0., e["start"] + e["unload_s"] - last_teacher_exit.get(g, float("-inf")))
                ids = e.get("evicted")
                if ids is not None and all((g, rid) in scored_context for rid in ids):
                    point["evicted_context"] = sum(scored_context[g, rid] for rid in ids)
            elif op in ("score", "decode") and all(seeds[rid] is not None for rid in cmd["ids"]):
                receipts = events[index+1:index+1+len(cmd["ids"])]
                expected = "supervision" if op == "score" else "prefix"
                if any(x["event"] != expected for x in receipts) or {x["rid"] for x in receipts} != set(cmd["ids"]):
                    raise ValueError(f"missing {expected} for measured {op}")
                group_size = cfg.get("group_size", 1)
                lanes = group_size // cfg.get("inference_tp_size", group_size)
                loads = [0] * lanes
                requests = [[] for _ in range(lanes)]
                estimated = [[] for _ in range(lanes)]
                quotas = [[] for _ in range(lanes)]
                by_id = {x["rid"]: x for x in receipts}
                for rid in cmd["ids"]:
                    x = by_id[rid]
                    rid = x["rid"]
                    lane = seeds[rid] // cfg["groups"] % lanes
                    if op == "score":
                        loads[lane] += max(1, prompts[rid]+x["end"]-1-x["reused_kv_tokens"])
                    else:
                        cached = e.get("cached_tokens", {}).get(rid, 0)
                        context, steps = prompts[rid]+x["start"], x["end"]-x["start"]
                        requests[lane].append((context, steps, cached))
                        estimated[lane].append((context, steps, context-1 if x["start"] else 0))
                        quotas[lane].append((steps, context, min(cfg["quantum"], limits[rid]-x["start"])))
                if op == "score":
                    point.update(role=cmd["role"], critical_tokens=max(loads))
                elif cmd.get("peer") is None and "cached_tokens" in e:
                    pages = cfg.get("sim_blocks", cfg.get("engine_kv_tokens", 131072)//16)
                    ticks = [[max(steps for steps, _ in wave) for wave in decode_windows(lane, pages, 16)]
                             for lane in quotas]
                    point["lanes"] = [rpc_decode_features(lane, t) for lane,t in zip(requests,ticks)]
                    point["estimated_lanes"] = [rpc_decode_features(lane, t) for lane,t in zip(estimated,ticks)]
            elif op == "train":
                point["lengths"] = [lengths[rid] for rid in cmd["ids"]]
            elif op == "commit":
                point["mode"] = "durable" if e.get("durable", True) else "export" if e.get("hf_checkpoint") else "step"
            elif op == "move":
                point.update(accepted=e["accepted"], rebuilt=e.get("rebuilt_tokens", 0),
                             install=e["install_end"] - e["install_start"])
            yield point


def fit_decode(rows):
    """Fit nonnegative kernel coefficients to the maximum measured replica lane."""
    import numpy as np
    from scipy.optimize import least_squares

    x = np.asarray([p["lanes"] for p in rows], dtype=float)
    y = np.asarray([p["dt"] for p in rows], dtype=float)
    # When every observed chunk runs to the quantum, launch and per-step time
    # are inseparable. Attribute their sum to steps so short tails scale down.
    dimensions = x.shape[2]
    columns = [i for i in range(dimensions) if x[:, :, i].any()]
    if not np.ptp(x[:, :, 1].max(axis=1)):
        columns.remove(0)
    x = x[:, :, columns]
    scale = np.maximum(1., x.max(axis=(0, 1)))
    x /= scale
    guess = np.zeros(x.shape[2])
    ticks = columns.index(1)
    guess[ticks] = np.median(y / x[:, :, ticks].max(axis=1))
    errors = lambda w: (x @ w).max(axis=1) - y
    noise = max(1.e-6, np.median(np.abs(errors(guess))))
    result = least_squares(errors, guess, bounds=(0., np.inf), loss="soft_l1", f_scale=noise)
    if not result.success:
        raise ValueError("decode latency fit did not converge")
    coefficients = np.zeros(dimensions)
    coefficients[columns] = result.x / scale
    if 5 not in columns:
        coefficients[5] = coefficients[1]
    return coefficients.tolist()


def fit_profile(cfg, points):
    samples = defaultdict(list)
    by_op = defaultdict(list)
    for p in points:
        if p["op"] == "switch":
            for key in set(p["keys"]):
                samples[key].append(p)
        by_op[p["op"]].append(p)
    cells = {}
    for key, group in samples.items():
        fields = {field: median(p[field] for p in group) for field in ("unload", "load", "sync")}
        devices = [p["device"] for p in group]
        # An absent device report preserves the previous trainer residency.
        local = [max(0., p["load"] - p["transfer"]) for p in group if p["transfer"] is not None]
        if local:
            fields["local_load"] = median(local)
        fields["device"] = max(dict.fromkeys(devices), key=devices.count)
        cells[key] = dict(fields, n=len(group))
    profile = {"switches": cells, "lifecycle": cfg["sim_teacher_lifecycle"]}
    launches = defaultdict(list)
    for p in by_op["switch"]:
        if "activation_s" in p:
            launches[p["new"]].append(p["activation_s"])
    profile["teacher_activation_s"] = {role: median(times) for role, times in launches.items()}
    decode = [p for p in by_op["decode"] if "lanes" in p]
    if decode:
        try:
            profile["decode_rpc"] = fit_decode(decode)
        except (ImportError, ValueError):
            # A small first batch may not identify all coefficients. Per-step
            # latency is still identifiable from acknowledged token counts.
            profile["decode_rpc"] = [0., median(p["dt"]/max(lane[1] for lane in p["lanes"])
                                              for p in decode), 0., 0., 0., 0.]
    teachers = defaultdict(list)
    for p in by_op["score"]:
        if "critical_tokens" in p:
            teachers[p["role"]].append((p["critical_tokens"], p["dt"]))
    profile["score"] = {role: fit_line(rows) for role, rows in teachers.items()}
    preparations = [p["prepare_s"] for p in by_op["switch"] if "prepare_s" in p]
    if preparations:
        profile["teacher_prepare_s"] = median(preparations)
    releases = defaultdict(list)
    for p in by_op["switch"]:
        if p["release_key"]:
            releases[p["release_key"]].append(p)
    profile["trainer_release"] = {}
    for key, group in releases.items():
        devices = [p["device"] for p in group]
        profile["trainer_release"][key] = dict(unload=median(p["unload"] for p in group),
                                               device=max(devices, key=devices.count))
    collectives = defaultdict(list)
    for p in by_op["switch"]:
        if p["transfer"] is not None:
            collectives[p["v"]].append(p["transfer"])
    complete = [min(times) for times in collectives.values() if len(times) == cfg["groups"]]
    if complete and cfg.get("max_train_groups", 1) > 1:
        profile["tensor_collective_s"] = median(complete)
    if by_op["train"]:
        profile["train"] = fit_line([(train_width(p["lengths"], cfg), p["dt"]) for p in by_op["train"]])
    residuals = defaultdict(list)
    for p in points:
        if "lanes" in p:
            key = "decode"
            predict = max(decode_duration(profile["decode_rpc"], lane) for lane in p["lanes"])
        elif "critical_tokens" in p:
            key = "score/" + p["role"]
            base, slope = profile["score"][p["role"]]
            predict = base + slope*p["critical_tokens"]
        elif p["op"] == "train":
            key = "train"
            predict = train_seconds(dict(cfg, sim_worker_profile=profile), p["lengths"])
        else:
            continue
        if predict > 0 and p["dt"] > 0:
            residuals[key].append(p["dt"]/predict)
    profile["service_ratios"] = {key: ([min(values), *quantiles(values, n=20, method="inclusive"), max(values)]
                                       if len(values)>1 else values) for key,values in residuals.items()}
    if by_op["reduce_grads"]:
        profile["reduce"] = median(p["dt"] for p in by_op["reduce_grads"])
    for mode in ("step", "export", "durable"):
        times = [p["dt"] for p in by_op["commit"] if p["mode"] == mode]
        if times:
            profile[mode] = median(times)
    moves = by_op["move"]
    if moves:
        profile["move_reject"] = median(p["dt"] for p in moves if not p["accepted"]) if any(
            not p["accepted"] for p in moves) else None
        accepts = [p for p in moves if p["accepted"]]
        if accepts:
            profile["move_base"] = median(p["dt"] - p["install"] for p in accepts)
            if any(p["rebuilt"] for p in accepts):
                profile["move_install"] = fit_line([(p["rebuilt"], p["install"]) for p in accepts if p["rebuilt"]])
    return profile


def evaluate(cfg, points, trained, held, validation=None, extra_fitting=()):
    if not trained or not held or (validation is None and max(trained) >= min(held)):
        raise ValueError("fit and validation must be disjoint chronological rounds")
    testing = points if validation is None else validation
    for versions, samples in ((trained, points), (held, testing)):
        for v in versions:
            if not any(p["op"] == "commit" and p["v"] == v for p in samples):
                raise ValueError(f"missing completed round {v}")
    fitting = [p for p in points if p["v"] in trained] + list(extra_fitting)
    profile = fit_profile(cfg, fitting)
    historical = defaultdict(list)
    for p in fitting:
        if p["op"] == "switch":
            historical[p["old"], p["new"]].append(p)
    pair = {edge: median(p["unload"] for p in group) + median(p["load"] + p.get("wait_s", 0.) for p in group)
            for edge, group in historical.items()}
    old_train = median(p["dt"] / sum(p["lengths"]) for p in fitting if p["op"] == "train")
    decode_steps = [p["dt"] / max(l[1] for l in p["lanes"]) for p in fitting if "lanes" in p]
    score_points = [(p["critical_tokens"], p["dt"]) for p in fitting if "critical_tokens" in p]
    pooled_score = fit_line(score_points) if score_points else None
    errors = defaultdict(list)
    barriers = defaultdict(list)
    for p in testing:
        if p["v"] not in held:
            continue
        if p["op"] == "switch":
            key, cell = fitted_switch(profile, p["keys"], p["release_key"])
            base = pair.get((p["old"], p["new"]), cfg["costs"]["switch"])
            legacy = base + (cfg.get("sim_replica_sync_s", cfg.get("sim_publish_s", 1.))
                             if p["pub"] and cfg.get("max_train_groups", 1) > 1 else 0.)
            pred = cell["unload"] + cell["load"] if cell else cfg["costs"]["switch"]
            if "prepare_elapsed" in p:
                pred += max(0., profile.get("teacher_prepare_s", 0.) - p["prepare_elapsed"])
            label = "publish" if p["pub"] else "switch"
            if not p["pub"] and p["old"] != p["new"] and {p["old"], p["new"]} <= cfg["teachers"].keys():
                errors["teacher_to_teacher"].append((abs(legacy-p["dt"]), abs(pred-p["dt"]), key is not None, p["dt"]))
            if p["pub"] and "tensor_collective_s" in profile:
                local = cell["unload"] + cell["local_load"] if cell and "local_load" in cell else pred
                barriers[p["v"]].append((legacy, local, key is not None, p["dt"]))
                continue
        elif p["op"] == "decode" and "lanes" in p and decode_steps:
            key, label = "decode", "decode"
            legacy = median(decode_steps) * max(l[1] for l in p["lanes"])
            pred = max(decode_duration(profile["decode_rpc"], lane) for lane in p["lanes"])
            estimated = max(decode_duration(profile["decode_rpc"], lane) for lane in p["estimated_lanes"])
            errors["decode/estimated_cache"].append((abs(legacy-p["dt"]), abs(estimated-p["dt"]), True, p["dt"]))
        elif p["op"] == "score" and "critical_tokens" in p:
            tokens = p["critical_tokens"]
            costs = cfg["costs"]
            base, slope = pooled_score or (costs["score_base"], costs["score_token"])
            legacy = base + tokens * slope
            coeff = profile["score"].get(p["role"])
            pred = coeff[0] + tokens * coeff[1] if coeff else legacy
            key = p["role"] if coeff else None
            label = "score/" + p["role"]
        elif p["op"] == "train":
            key = "train"
            legacy = sum(p["lengths"]) * old_train
            pred = train_seconds(dict(cfg, sim_worker_profile=profile), p["lengths"])
            label = "train"
        else:
            continue
        errors[label].append((abs(legacy-p["dt"]), abs(pred-p["dt"]), key is not None, p["dt"]))
    for rows in barriers.values():
        if len(rows) != cfg["groups"]:
            raise ValueError("incomplete publication barrier")
        wall = max(r[1] for r in rows) + profile["tensor_collective_s"]
        errors["publish"].extend((abs(old-dt), abs(wall-dt), covered, dt) for old, _, covered, dt in rows)
    return profile, {label: dict(n=len(rows), covered=sum(r[2] for r in rows),
                         previous_mae_s=mean(r[0] for r in rows), profile_mae_s=mean(r[1] for r in rows),
                         measured_mean_s=mean(r[3] for r in rows)) for label, rows in errors.items()}


def kv_coverage(points):
    switches = [p for p in points if p["op"] == "switch" and p["new"] == "train" and "evicted_context" in p]
    positive = [p for p in switches if p["evicted_context"]]
    return dict(training_switches=len(switches), training_switches_with_teacher_kv=len(positive),
                max_evicted_context_tokens=max((p["evicted_context"] for p in positive), default=0))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--fit", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--holdout", type=int, nargs="+", default=[2])
    parser.add_argument("--config-out", type=Path)
    parser.add_argument("--policy", default="cost_search", help="Policy for both executors in the exported configuration")
    parser.add_argument("--validation-trace", type=Path, help="Evaluate the fitted model on a separate native run")
    parser.add_argument("--extra-fit-trace", type=Path, action="append", default=[],
                        help="Include the same fit rounds from another workload with the same worker layout")
    args = parser.parse_args()
    cfg, events = read_trace(args.trace)
    points = list(observations(cfg, events))
    validation = None
    def matching_trace(path):
        other, trace = read_trace(path)
        for key in ("student", "teachers", "groups", "group_size", "inference_tp_size", "sim_teacher_lifecycle",
                    "trainer_attention", "weight_sync", "engine_kv_tokens", "preload_teacher_imports"):
            if other.get(key) != cfg.get(key):
                raise ValueError(f"{path} has a different {key}")
        return list(observations(other, trace))
    used = {args.trace.resolve()}
    if args.validation_trace and args.validation_trace.resolve() not in used:
        validation = matching_trace(args.validation_trace)
        used.add(args.validation_trace.resolve())
    extra = []
    for path in args.extra_fit_trace:
        if path.resolve() in used:
            raise ValueError("additional training traces must be distinct from evaluation and each other")
        used.add(path.resolve())
        sample = [p for p in matching_trace(path) if p["v"] in args.fit]
        if {p["v"] for p in sample if p["op"] == "commit"} != set(args.fit):
            raise ValueError(f"incomplete calibration trace: {path}")
        # Collective arrivals are aggregated by round, within one original execution.
        extra.extend(dict(p, v=(str(path), p["v"])) for p in sample)
    profile, report = evaluate(cfg, points, set(args.fit), set(args.holdout), validation, extra)
    report["kv_coverage"] = kv_coverage(p for p in (points if validation is None else validation)
                                        if p["v"] in set(args.holdout))
    if args.config_out:
        from .calibrate import fit
        calibrated = fit(args.trace, max(args.fit))
        calibrated.update(policy=args.policy, sim_policy=args.policy, sim_engine_classes=cfg["sim_engine_classes"],
                          sim_teacher_lifecycle=cfg["sim_teacher_lifecycle"], sim_worker_profile=profile)
        with args.config_out.open("x") as stream:
            json.dump(calibrated, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
