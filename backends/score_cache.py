"""Bound the logical teacher cache with the same physical TP-lane flushes in both executors."""


def trim_score_cache(cache, budget):
    """Return flushed lanes and forgotten prefixes; independent lanes keep their KV."""
    sizes = {}
    for lane, tokens in cache.values():
        sizes[lane] = sizes.get(lane, 0) + tokens
    excess = sum(sizes.values()) - budget
    flushed = []
    while excess > 0 and sizes:
        # SGLang exposes a flush per independent engine. Release the smallest
        # sufficient lane, or the largest one if more than one flush is needed.
        enough = [lane for lane in sizes if sizes[lane] >= excess]
        lane = min(enough, key=sizes.get) if enough else max(sizes, key=sizes.get)
        flushed.append(lane)
        excess -= sizes.pop(lane)
    evicted = [rid for rid, (lane, _) in cache.items() if lane in flushed] if flushed else []
    for rid in evicted:
        del cache[rid]
    return flushed, evicted
