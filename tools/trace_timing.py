"""Replay settled action dependencies with alternative compute service times."""
import argparse
import json
from collections import Counter
from pathlib import Path

from ..backends.sim_profile import decode_duration, train_seconds
from .worker_costs import observations, read_trace


def work_seconds(cfg, profile, point, cache):
    if cache == "measured_time":
        return point["dt"]
    if point["op"] == "decode" and "lanes" in point:
        lanes = point[cache]
        if "decode_rpc" in profile:
            return max(decode_duration(profile["decode_rpc"], lane) for lane in lanes)
        if "decode" in profile and all(lane[-1] == 0 for lane in lanes):
            return max(decode_duration(profile["decode"], lane[:5]) for lane in lanes)
    if point["op"] == "score" and point.get("role") in profile.get("score", {}):
        base, slope = profile["score"][point["role"]]
        return base+slope*point["critical_tokens"]
    if point["op"] == "train" and "train" in profile:
        return train_seconds(cfg, point["lengths"], point["wave_widths"])
    return None


def replay(folder, profile):
    cfg, events = read_trace(folder)
    cfg = dict(cfg, sim_worker_profile=profile)
    points = iter(observations(cfg, events))
    dispatch, actions = {}, {}
    for index, e in enumerate(events):
        if e["event"] == "dispatch":
            dispatch[e["gpu"]] = index
        elif e["event"] == "complete":
            point = next(points)
            issued = dispatch.pop(e["gpu"])
            if point["op"] != events[issued]["op"] or point["v"] != e["version"]:
                raise ValueError("worker timing point does not match its dispatch")
            actions[issued] = e, point
    if dispatch:
        raise ValueError("trace contains unsettled dispatches")

    def timeline(cache):
        pending, results, coverage = {}, {}, {}
        for index, e in enumerate(events):
            kind = e["event"]
            if kind == "round_start":
                previous = start = e["t"]
                cursor = 0.
                predicted, measured = Counter(), Counter()
            elif kind == "dispatch":
                # Charge observed controller computation and dispatch-to-worker latency.
                cursor += e["t"]-previous
                done, point = actions[index]
                dt = work_seconds(cfg, profile, point, cache)
                if point["op"] in ("decode", "score", "train"):
                    counts = measured if dt is None or cache == "measured_time" else predicted
                    counts[point["op"]] += 1
                pending[e["gpu"]] = cursor + done["start"]-e["t"] + (point["dt"] if dt is None else dt)
                previous = e["t"]
            elif kind == "complete":
                # The recorded order fixes the decisions and the receipt waits they depend on.
                # Other workers keep advancing during those waits.
                cursor = max(cursor,pending.pop(e["gpu"])) + e["t"]-max(previous,e["end"])
                previous = e["t"]
            elif kind == "round_end":
                if pending:
                    raise ValueError("round finished with unsettled workers")
                results[e["version"]] = cursor + e["t"]-previous
                coverage[e["version"]] = dict(predicted=dict(predicted), measured=dict(measured))
                if cache == "measured_time" and abs(results[e["version"]]-(e["t"]-start))>1.e-5:
                    raise ValueError("measured service times did not reconstruct the trace clock")
        return results, coverage

    actual, _ = timeline("measured_time")
    hits, _ = timeline("lanes")
    estimated, coverage = timeline("estimated_lanes")
    return [dict(version=v, actual_s=t, observed_cache_s=hits[v], estimated_cache_s=estimated[v],
                 compute_coverage=coverage[v])
            for v,t in actual.items()]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    profile = cfg.get("sim_execution_profile") or cfg.get("sim_worker_profile")
    if not profile:
        parser.error("configuration must contain a fitted worker profile")
    print(json.dumps(replay(args.trace, profile), indent=2))
