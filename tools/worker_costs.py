"""Fit worker time from real receipts, then score later rounds without fitting on them."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean, median

from ..backends.sim_profile import SwitchState, engine_classes, fit_line, fitted_switch, train_seconds, train_width


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
    for e in events:
        kind = e["event"]
        if kind == "round_start":
            lengths = {r["rid"]: r["prompt_tokens"] for r in e["requests"]}
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
                ranks = e.get("teacher_rank_timings", [])
                if ranks and all("prepared_import_s" in r for r in ranks):
                    point["prepare_s"] = max(r["prepared_import_s"] for r in ranks)
                    point["wait_s"] = e["standby_wait_s"]
                    point["load"] = max(0., point["load"] - point["wait_s"])
                    point["prepare_elapsed"] = max(0., e["start"] + e["unload_s"] - last_teacher_exit.get(g, float("-inf")))
                ids = e.get("evicted")
                if ids is not None and all((g, rid) in scored_context for rid in ids):
                    point["evicted_context"] = sum(scored_context[g, rid] for rid in ids)
            elif op == "train":
                point["lengths"] = [lengths[rid] for rid in cmd["ids"]]
            elif op == "commit":
                point["mode"] = "durable" if e.get("durable", True) else "export" if e.get("hf_checkpoint") else "step"
            elif op == "move":
                point.update(accepted=e["accepted"], rebuilt=e.get("rebuilt_tokens", 0),
                             install=e["install_end"] - e["install_start"])
            yield point


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


def evaluate(cfg, points, trained, held):
    if not trained or not held or max(trained) >= min(held):
        raise ValueError("fit and validation must be disjoint chronological rounds")
    for v in trained | held:
        if not any(p["op"] == "commit" and p["v"] == v for p in points):
            raise ValueError(f"missing completed round {v}")
    fitting = [p for p in points if p["v"] in trained]
    profile = fit_profile(cfg, fitting)
    historical = defaultdict(list)
    for p in fitting:
        if p["op"] == "switch":
            historical[p["old"], p["new"]].append(p)
    pair = {edge: median(p["unload"] for p in group) + median(p["load"] + p.get("wait_s", 0.) for p in group)
            for edge, group in historical.items()}
    old_train = median(p["dt"] / sum(p["lengths"]) for p in fitting if p["op"] == "train")
    errors = defaultdict(list)
    barriers = defaultdict(list)
    for p in points:
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
            if p["pub"] and "tensor_collective_s" in profile:
                local = cell["unload"] + cell["local_load"] if cell and "local_load" in cell else pred
                barriers[p["v"]].append((legacy, local, key is not None, p["dt"]))
                continue
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
    args = parser.parse_args()
    cfg, events = read_trace(args.trace)
    points = list(observations(cfg, events))
    profile, report = evaluate(cfg, points, set(args.fit), set(args.holdout))
    report["kv_coverage"] = kv_coverage(p for p in points if p["v"] in set(args.fit + args.holdout))
    if args.config_out:
        from .calibrate import fit
        calibrated = fit(args.trace, max(args.fit))
        calibrated.update(sim_policy="cost_search", sim_engine_classes=cfg["sim_engine_classes"],
                          sim_teacher_lifecycle=cfg["sim_teacher_lifecycle"], sim_worker_profile=profile)
        with args.config_out.open("x") as stream:
            json.dump(calibrated, stream, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
