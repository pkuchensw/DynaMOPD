"""Worker timing primitives shared by trace fitting, prediction and virtual execution."""
from pathlib import Path
from statistics import median
from ..scheduler.batching import training_waves


def engine_classes(cfg):
    if cfg.get("sim_engine_classes"):
        return cfg["sim_engine_classes"]
    paths = dict(rollout=cfg["student"], **cfg["teachers"])
    if cfg.get("inference_layout") != "single":
        return paths
    from ..workers.model import inference_signature
    limit = cfg["max_prompt"] + cfg["max_new_tokens"] + 1
    return {role: inference_signature(path, limit, cfg.get("qwen2_dynamic_rope", False))
            if (Path(path) / "config.json").is_file() else path for role, path in paths.items()}


def decode_roofline(cfg):
    c = cfg["costs"]
    defaults = dict(F=c["decode_step"], W=c["decode_step"]*.7, G=c["prefill_token"], A_d=2e-7)
    return dict(defaults, **cfg.get("sim_decode_roofline", {}))


def replay_move_base(cfg, *, accepted=True):
    """Fixed seconds per virtual handoff and their provenance. KV rebuild is billed separately."""
    profile = cfg.get("sim_worker_profile", {})
    if not accepted and profile.get("move_reject") is not None:
        return profile["move_reject"], "profile.move_reject"
    if "move_base" in profile:
        return profile["move_base"], "profile.move_base"
    if "move_base" in cfg["costs"]:
        return cfg["costs"]["move_base"], "costs.move_base"
    # An accepted handoff exports and releases on the source. A rejected one
    # exports and gets a reply from the destination. Both need two control calls.
    for seconds, source in ((cfg.get("sim_rpc_overhead_s"), "rpc_overhead_proxy"),
                            (cfg["costs"].get("score_base"), "score_launch_proxy"),
                            (cfg.get("sim_short_score_s"), "short_score_batch_proxy"),
                            (cfg["costs"].get("score_token"), "single_score_token_proxy")):
        if seconds is not None and seconds > 0:
            return 2 * seconds, source
    raise ValueError("migration replay needs a measured handoff or a positive RPC or scoring timing")


def decode_features(requests):
    """One parallel SGLang lane, with (initial context, output steps, cached input) per request."""
    ticks = tokens = attention = prefill = 0
    for context, steps, cached in requests:
        ticks = max(ticks, steps)
        tokens += steps
        attention += steps * context + steps * (steps - 1) // 2
        prefill += max(0, context - cached)
    return (int(ticks > 0), ticks, tokens, attention, prefill)


def decode_duration(coefficients, features):
    return sum(c * x for c, x in zip(coefficients, features, strict=True))


def rpc_decode_features(requests, window_steps):
    features = decode_features(requests)
    return (*features, max(0, sum(window_steps) - features[1]))


def decode_windows(entries, pages, page_size):
    """Partition one lane at its KV capacity using visible prefix and requested output quota."""
    window, used = [], 0
    for item, context, quota in entries:
        needed = (context + quota - 1 + page_size - 1) // page_size
        if needed > pages:
            raise MemoryError("one rollout context and its decode quota exceed the lane's KV capacity")
        if used + needed > pages:
            yield window
            window, used = [], 0
        window.append((item, needed))
        used += needed
    if window:
        yield window


def train_width(lengths, cfg):
    """Reconstruct the old assignment for historical receipts without rank widths."""
    world = cfg.get("group_size", 1)
    if cfg.get("trainer_attention") == "flash_attention_2":
        loads = [0] * world
        for length in sorted(lengths, reverse=True):
            rank = min(range(world), key=loads.__getitem__)
            loads[rank] += max(1, length - 1)
        return max(loads)
    return max(max(part) * len(part) for i in range(min(world, len(lengths)))
               if (part := lengths[i::world]))


def switch_keys(old, new, phase, engine_action, resident, owner, teachers):
    """Back off within the same publication/engine mechanism, never across a restart or lease."""
    a, b = ("teacher" if role in teachers else role for role in (old, new))
    prefix = (phase, engine_action)
    specific = ((a, new, resident, owner), (old, new), (a, new), (new,)) if new in teachers else ()
    for suffix in ((old, new, resident, owner), *specific, (a, b, resident, owner), (a, b, owner),
                   (a, b), (b,)):
        yield "|".join(map(str, prefix + suffix))


class SwitchState:
    def __init__(self, cfg, classes):
        self.cfg = cfg
        self.classes = classes
        self.engine = classes["rollout"]
        self.resident = False
        self.wake_calibrated = bool(cfg.get("calibrate_engine_wake"))
        self.teacher_ready_at = 0.

    def release_key(self, old, publish, owner):
        if publish and old == "train":
            return f"train|{self.wake_calibrated}|{owner}"
        return None

    def keys(self, old, new, publish, owner):
        phase = "publish" if publish else ("initial" if old == "cold" else "service")
        lifecycle = self.cfg.get("sim_teacher_lifecycle", "lease")
        if lifecycle not in ("lease", "restart"):
            raise ValueError("sim_teacher_lifecycle must be lease or restart")
        if lifecycle == "restart":
            mechanism = "restart" if new in self.classes and self.classes[new] != self.engine else "warm"
        else:
            dedicated = {role for role in self.cfg["teachers"] if self.classes[role] != self.classes["rollout"]}
            mechanism = ("release+activate" if old in dedicated and new in dedicated else
                         "release" if old in dedicated else "activate" if new in dedicated else "warm")
        return list(switch_keys(old, new, phase, mechanism, self.resident, owner, self.cfg["teachers"]))

    def advance(self, new, trainer_device=None):
        if new == "train":
            self.resident = False
        elif trainer_device is not None:
            self.resident = trainer_device == "cuda"
        if new in self.classes and self.cfg.get("sim_teacher_lifecycle", "lease") == "restart":
            if self.engine != self.classes[new]:
                self.wake_calibrated = False
            self.engine = self.classes[new]


def fitted_switch(profile, keys, release_key=None):
    for key in keys:
        if key in profile.get("switches", {}):
            cell = profile["switches"][key]
            release = profile.get("trainer_release", {}).get(release_key)
            return key, dict(cell, **release) if release else cell
    return None, None


def teacher_lease_window(ready_at, start, unload, release, prepare, leaving, entering):
    """Ready time of the CPU replacement rank and wait exposed after unloading."""
    if leaving:
        ready_at = start + release + prepare
    return ready_at, max(0., ready_at - start - unload) if entering else 0.


def fit_line(points):
    """Robust affine fit on short/long batches with nonnegative phase times."""
    ordered = sorted(points)
    n = max(1, len(ordered) // 3)
    lo, hi = ordered[:n], ordered[-n:]
    x0, x1 = (median(x for x, _ in group) for group in (lo, hi))
    if x1 == x0:
        return [0., median(t / x for x, t in points)]
    slope = max(0., (median(t for _, t in hi) - median(t for _, t in lo)) / (x1 - x0))
    base = max(0., median(t - slope * x for x, t in points))
    return [base, median(max(0., t - base) / x for x, t in points)]


def train_seconds(cfg, lengths, wave_widths=None):
    coeff = cfg.get("sim_worker_profile", {}).get("train")
    if coeff:
        if wave_widths is None:
            wave_widths = [max(loads) for _, _, loads in training_waves(lengths, cfg, int, forecast=True)]
        return len(wave_widths) * coeff[0] + coeff[1] * sum(wave_widths)
    return cfg["costs"]["train_token"] * sum(lengths)

