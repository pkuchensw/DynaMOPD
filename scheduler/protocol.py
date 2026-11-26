"""Immutable commands, round-local trajectory journal and exclusive GPU leases."""
from dataclasses import dataclass, field
import math


@dataclass(frozen=True)
class Action:
    kind: str
    gpu: int
    version: int
    payload: dict = field(default_factory=dict)
    peer: int | None = None

    @property
    def devices(self):
        return (self.gpu,) if self.peer is None else (self.gpu, self.peer)


def validate_batch_receipt(action, value):
    """Check every interval against the dispatched snapshot before releasing the lease."""
    scoring = action.kind == "score"
    results = value["scores" if scoring else "updates"]
    items = action.payload["items"]
    if set(results) != {item["rid"] for item in items}:
        raise RuntimeError(f"incomplete {action.kind} receipt")
    for item in items:
        result = results[item["rid"]]
        prefix = len(item["tokens"])
        if scoring:
            if result["start"] != item["start"] or result["end"] != prefix:
                raise RuntimeError(f"teacher interval differs from dispatched prefix for {item['rid']}")
        else:
            steps = min(action.payload["steps"], item["limit"] - prefix)
            if result["start"] != prefix or not 0 < len(result["tokens"]) <= steps:
                raise RuntimeError(f"decode interval exceeds dispatched chunk for {item['rid']}")


@dataclass
class Row:
    rid: str
    prompt: list[int]
    teacher: str
    seed: int
    version: int
    owner: int
    limit: int
    tokens: list[int] = field(default_factory=list)
    behavior: list[float] = field(default_factory=list)
    labels: list[float] = field(default_factory=list)
    finish: str | None = None
    # Each location names a teacher cache ending immediately before token 'end'.
    kv: dict[int, int] = field(default_factory=dict)
    scoring: bool = False
    training: bool = False
    trained: bool = False
    pending_since: float | None = None
    top_k: int = 0
    teacher_topk: list = field(default_factory=list)
    train_on_length: bool = False
    deferred_prefill: bool = False

    @property
    def discarded(self):
        return self.finish == "length" and not self.train_on_length

    def prefix(self, start, tokens, logp, finish):
        end = start + len(tokens)
        if (self.finish or start != len(self.tokens) or not tokens or len(tokens) != len(logp)
                or end > self.limit or not all(map(math.isfinite, logp))
                or (finish == "length" and end != self.limit) or (finish is None and end == self.limit)
                or finish not in (None, "eos", "length")):
            raise ValueError(f"invalid prefix receipt for {self.rid}")
        self.tokens.extend(tokens)
        self.behavior.extend(logp)
        self.finish = finish

    def deliver(self, start, end, labels, gpu, retained, topk=None):
        if (not self.scoring or start != len(self.labels) or end > len(self.tokens)
                or end <= start or end - start != len(labels) or not all(map(math.isfinite, labels))):
            raise ValueError(f"invalid supervision receipt for {self.rid}")
        if self.top_k:
            if topk is None or len(topk) != len(labels) or len(self.teacher_topk) != start:
                raise ValueError(f"incomplete top-k supervision for {self.rid}")
            for candidates in topk:
                if (len(candidates) != self.top_k or len({i for i,_ in candidates}) != self.top_k
                        or any(type(i) is not int or i < 0 or not math.isfinite(lp) for i,lp in candidates)):
                    raise ValueError(f"invalid teacher candidates for {self.rid}")
            self.teacher_topk.extend(topk)
        elif topk is not None:
            raise ValueError(f"unexpected top-k supervision for {self.rid}")
        self.labels.extend(labels)
        self.scoring = False
        if retained:
            self.kv[gpu] = end
        else:
            self.kv.pop(gpu, None)

    @property
    def ready(self):
        return (self.finish is not None and not self.discarded and len(self.labels) == len(self.tokens)
                and (not self.top_k or len(self.teacher_topk) == len(self.tokens))
                and not (self.training or self.trained))

    def item(self, end=None):
        return dict(rid=self.rid, prompt=self.prompt, tokens=self.tokens[:end],
                    student_version=self.version,teacher=self.teacher,
                    seed=self.seed, limit=self.limit, finish=self.finish)


@dataclass
class Group:
    gpu: int
    role: str = "cold"
    version: int = -1
    busy: Action | None = None
    free_bytes: int = 0
    resident_since: float = 0.
    busy_since: float = 0.
    last_inference_role: str = "rollout"
    trainer_resident: bool = False
    wake_calibrated: bool | None = None
    prepared_teachers: frozenset | None = None
    cached_teacher_weights: frozenset = frozenset()
    teacher_start_s: float = 0.
    teacher_activation_s: dict = field(default_factory=dict)
    teacher_ready_at: float = 0.


class Costs:
    """Exponentially weighted completed-action observations, in seconds."""
    def __init__(self, cfg):
        self.measuring = cfg.get("first_round_calibration", False)
        self.values = {} if self.measuring else dict(cfg["costs"])
        self.score_samples = {}
        if not self.measuring:
            self.values.setdefault("teacher_wake", self.values["score_base"])
            self.values.setdefault("replica_reduce", self.values["switch"])

    def get(self, name):
        if self.measuring and name == "replica_reduce" and name not in self.values:
            return self.values["rollout_publication"]
        return self.values[name]

    def switch(self, old, new):
        return self.values.get((old, new), self.get("switch"))

    def observe_score(self, teacher, tokens, seconds):
        if tokens <= 0 or seconds <= 0 or not math.isfinite(seconds):
            return
        n, x, y, xx, xy = self.score_samples.get(teacher, (0, 0., 0., 0., 0.))
        n += 1
        dx, dy = tokens - x, seconds - y
        x, y = x + dx/n, y + dy/n
        self.score_samples[teacher] = n, x, y, xx + dx*(tokens-x), xy + dx*(seconds-y)

    def score_coefficients(self, teacher, prior):
        if teacher not in self.score_samples:
            return prior
        _, x, y, xx, xy = self.score_samples[teacher]
        slope = max(0., min(y/x, xy/xx)) if xx > 0. else y/x
        return max(0., y-slope*x), slope

    def observe(self, name, value):
        if math.isfinite(value) and (value > 0 or (name == "move_base" and value == 0)):
            self.values[name] = .7 * self.values.get(name, value) + .3 * value
