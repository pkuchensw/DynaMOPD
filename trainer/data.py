"""Tokenization, teacher routing and round input/output formats."""
import json
import random

from ..scheduler.protocol import Row


def vocabulary(cfg):
    from transformers import AutoConfig, AutoTokenizer, GenerationConfig
    tokenizer = AutoTokenizer.from_pretrained(cfg["student"], local_files_only=True)
    student = tokenizer.get_vocab()
    common = set(student.values())
    widths = []
    identical_mapping = True
    for path in [cfg["student"], *cfg["teachers"].values()]:
        other = AutoTokenizer.from_pretrained(path, local_files_only=True).get_vocab()
        identical_mapping = identical_mapping and student == other
        common.intersection_update(i for token, i in student.items() if other.get(token) == i)
        widths.append(AutoConfig.from_pretrained(path).vocab_size)
    if not common or max(common) >= min(widths):
        raise ValueError("incompatible teacher vocabulary")
    cfg.pop("sampled_id_limits", None)
    # A sampled K1 target only needs a row in the student and its routed teacher.
    # Keep the common exclusion set for backends that use a shared logits mask.
    if cfg.get("algorithm") == "on_policy_k1" and identical_mapping:
        cfg["excluded_ids"] = list(range(min(widths), max(widths)))
        cfg["sampled_id_limits"] = {
            key: min(widths[0], width) for key, width in zip(cfg["teachers"], widths[1:], strict=True)
        }
    else:
        cfg["excluded_ids"] = sorted(set(range(max(widths))) - common)
    eos = GenerationConfig.from_pretrained(cfg["student"]).eos_token_id
    cfg["eos_ids"] = [eos] if isinstance(eos, int) else eos
    if not set(cfg["eos_ids"]) <= common:
        raise ValueError("EOS missing from shared vocabulary")
    return tokenizer, common


def get_rows(cfg, tokenizer, common, data, version, simulate, algorithm):
    rows = []
    for i in range(cfg["batch_size"]):
        x = data[(version*cfg["batch_size"]+i) % len(data)]
        # Simulation inputs carry the resolved teacher saved by the real job.
        route = x["teacher"] if simulate else algorithm.route(x, cfg)
        if route not in cfg["teachers"]:
            raise ValueError(f"unknown teacher {route}")
        if simulate:
            ids = [1] * x["prompt_len"]
        else:
            messages = x.get("prompt", x.get("conversations"))
            if isinstance(messages, str):
                messages = [{"role":"user", "content":messages}]
            rendered = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                      enable_thinking=False)
            # Qwen3 Base lacks the two added thinking tokens. Drop the empty preamble in
            # that case, retaining the ordinary assistant header and original message.
            empty = "<think>\n\n</think>\n\n"
            if rendered.endswith(empty) and not set(tokenizer.encode(empty, add_special_tokens=False)) <= common:
                rendered = rendered[:-len(empty)]
            ids = tokenizer.encode(rendered, add_special_tokens=False)
            if not set(ids) <= common:
                raise ValueError("prompt uses token IDs outside the shared teacher vocabulary")
        limit = x.get("max_new_tokens", cfg["max_new_tokens"])
        if not ids or len(ids) > cfg["max_prompt"] or not 1 <= limit <= cfg["max_new_tokens"]:
            raise ValueError("unsupported prompt or response length")
        rid = f"v{version}/r{i}"
        seed = cfg["seed"]+version*cfg["batch_size"]+i
        if simulate:
            rid = x.get("rid", rid)
            seed = x.get("seed", seed)
        rows.append(Row(rid, ids, route, seed, version,
                        i % cfg["groups"], limit, top_k=cfg.get("teacher_top_k", 0),
                        train_on_length=cfg.get("train_on_length", False)))
    return rows


def load_data(path):
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq
        records = pq.read_table(path).to_pylist()
    else:
        records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("empty training dataset")
    return records


def simulated_inputs(cfg):
    rng = random.Random(cfg["seed"])
    lengths = (8,12,104,208,160,248,40,136,9,18,16,24,12,8,120,184)
    routes = list(cfg["teachers"])
    return [dict(prompt_len=rng.randint(24,96), target=lengths[i % len(lengths)], teacher=routes[i % len(routes)])
            for i in range(cfg["batch_size"])]


def replay_inputs(path, cfg):
    records = load_data(path)
    if len(records) != cfg["batch_size"]:
        raise ValueError("replay batch size and configuration differ")
    inputs = []
    for x in records:
        length = len(x["response"])
        truncated = x["finish"] == "length"
        # Older traces recorded a sample's cap only implicitly when it truncated.
        limit = x.get("max_new_tokens", length if truncated else cfg["max_new_tokens"])
        if (x["finish"] not in ("eos", "length") or not 0 < length <= limit <= cfg["max_new_tokens"]
                or (truncated and length != limit)):
            raise ValueError(f"replay response is incompatible with its token limit in {path}")
        inputs.append(dict(prompt_len=len(x["prompt"]), teacher=x["teacher"],
                           target=length + int(truncated), max_new_tokens=limit,
                           **{key: x[key] for key in ("rid", "seed") if key in x}))
    return inputs


def save_samples(path, rows):
    with path.open("x") as f:
        for row in rows:
            extra = {"teacher_topk": row.teacher_topk} if row.top_k else {}
            f.write(json.dumps(dict(rid=row.rid, version=row.version, seed=row.seed,
                                    prompt=row.prompt, teacher=row.teacher, response=row.tokens,
                                    finish=row.finish, max_new_tokens=row.limit, behavior=row.behavior,
                                    teacher_logp=row.labels, trained=row.trained, **extra))+"\n")
