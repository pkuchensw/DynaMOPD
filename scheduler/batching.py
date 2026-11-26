"""Deterministic FSDP microbatches shared by admission, workers and SimRL."""


def training_waves(entries, cfg, length, limit=None, *, forecast=False):
    """Yield (ordered examples, per-rank indices, per-rank input tokens).

    The LM consumes prompt + response[:-1]. SDPA pads within each rank.
    An unresolved forecast can exceed the real budget; only actual inputs
    must fit a rank. Every nonempty wave is one synchronized forward/backward.
    """
    world = cfg.get("group_size", 1)
    group_count = cfg["microbatch"]
    rank_count = (group_count + world - 1) // world
    rank_budget = cfg["train_budget"] // world
    packed = cfg.get("trainer_attention") == "flash_attention_2"
    limit = min(limit or group_count, group_count)

    def assign(widths):
        parts = [[] for _ in range(world)]
        loads = [0] * world
        for i in sorted(range(len(widths)), key=widths.__getitem__, reverse=True):
            choices = []
            for rank, part in enumerate(parts):
                load = loads[rank] + widths[i] if packed or not part else widths[part[0]] * (len(part) + 1)
                if len(part) < rank_count and load <= rank_budget:
                    choices.append((load, len(part), rank))
            if not choices:
                return None
            load, _, rank = min(choices)
            parts[rank].append(i)
            loads[rank] = load
        return parts, loads

    offset = 0
    while offset < len(entries):
        block = entries[offset:offset + limit]
        widths = [max(1, length(row) - 1) for row in block]
        plan = assign(widths)
        if plan is None:
            if widths[0] > rank_budget:
                if not forecast:
                    raise ValueError(f"one training sample needs {widths[0]} input tokens on one rank, "
                                     f"exceeding the per-rank train budget of {rank_budget}")
                block = block[:1]
                plan = ([[0]] + [[] for _ in range(world - 1)], [widths[0]] + [0] * (world - 1))
            else:
                lo, hi = 1, len(block)
                while lo + 1 < hi:
                    mid = (lo + hi) // 2
                    if assign(widths[:mid]) is None:
                        hi = mid
                    else:
                        lo = mid
                block = block[:lo]
                plan = assign(widths[:lo])
        parts, loads = plan
        yield block, parts, loads
        offset += len(block)
