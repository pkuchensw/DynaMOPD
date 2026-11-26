"""Local SGLang TP servers. Per-request sampling state is the seed and absolute position."""
import os
import json
import signal
import socket
import struct
import subprocess
import sys
import time
from pathlib import Path
from uuid import uuid4

import requests

from .score_cache import trim_score_cache


def sglang_args(cfg, width, name, model, initial_version):
    limits = cfg.get("sglang_limits", {})
    max_requests = limits.get("max_running_requests", max(16, cfg["decode_batch"], cfg["score_batch"]))
    args = ["--model-path", model, "--tp-size", str(width), "--dtype", "bfloat16",
            "--weight-version", initial_version, "--attention-backend", "triton",
            "--disable-piecewise-cuda-graph", "--disable-custom-all-reduce", "--skip-server-warmup",
            "--context-length", str(cfg["max_prompt"]+cfg["max_new_tokens"]+1),
            "--max-total-tokens", str(cfg["engine_kv_tokens"]), "--mem-fraction-static",
            str(cfg.get("mem_fraction_static", .36)), "--max-running-requests", str(max_requests),
            "--log-level", "warning"]
    for key in ("max_prefill_tokens", "chunked_prefill_size"):
        if key in limits:
            args.extend(["--"+key.replace("_", "-"), str(limits[key])])
    if name == "rollout" and cfg.get("qwen2_dynamic_rope", False):
        from .qwen2_rope import LOADER
        args += ["--custom-weight-loader", LOADER]
    if name == "rollout" or cfg.get("rollout_sampling_mode", "batch_invariant") == "batch_invariant":
        args += ["--enable-deterministic-inference"]
    if name=="rollout" and cfg.get("cuda_graph", False):
        args += ["--cuda-graph-max-bs", str(limits.get("cuda_graph_max_bs", max(8,cfg["decode_batch"])))]
    else:
        args += ["--disable-cuda-graph"]
    return args


class Server:
    def __init__(self, cfg, devices, name, model, initial_version, *, cpu_weight_backup=False):
        self.name, self.version = name, initial_version
        self.session = requests.Session()
        self.session.trust_env = False
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        physical = ",".join(devices)
        self.log = Path(cfg["log_dir"])/f"sglang-{physical}-{name}.log"
        request_seeds = name == "rollout" and cfg.get("rollout_sampling_mode") == "request_seed"
        launcher = "DynaMOPD.backends.seeded_server" if request_seeds else "sglang.launch_server"
        self.tp_size = len(devices)
        self.rope_theta = self.read_rope(model)
        args = [sys.executable, "-m", launcher, "--host", "127.0.0.1", "--port", str(port),
                "--enable-memory-saver", *sglang_args(cfg, self.tp_size, name, model, initial_version)]
        if cpu_weight_backup:
            args.append("--enable-weights-cpu-backup")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=physical)
        with self.log.open("a") as log:
            self.proc = subprocess.Popen(args, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         start_new_session=True)
        try:
            until = time.monotonic()+cfg.get("startup_timeout", 1200)
            while time.monotonic() < until and self.proc.poll() is None:
                try:
                    info = self.session.get(self.url+"/model_info", timeout=2)
                    if info.ok and info.json().get("weight_version") == initial_version:
                        break
                except requests.RequestException:
                    pass
                time.sleep(.5)
            else:
                raise RuntimeError(f"SGLang failed to start. See {self.log}")
            info = self.session.get(self.url+"/server_info", timeout=15)
            info.raise_for_status()
            self.capacity = int(info.json()["max_total_num_tokens"])
            # Compile both the scoring and decode paths outside the measured round.
            if cfg.get("check_alignment", True):
                from .checks import check_server
                self.log.with_suffix(".check.json").write_text(json.dumps(check_server(self), indent=2))
            else:
                self.post("generate", dict(input_ids=[1,2,3], sampling_params=dict(max_new_tokens=2,
                          temperature=1., sampling_seed=1), return_logprob=True, logprob_start_len=0))
        except BaseException:
            self.close()
            raise

    def post(self, endpoint, body):
        resp = self.session.post(f"{self.url}/{endpoint}", json=body, timeout=(10,900))
        if not resp.ok:
            raise RuntimeError(f"SGLang {endpoint} {resp.status_code}: {resp.text[:1200]} ({self.log})")
        if endpoint=="generate":
            self.generate_body_bytes = len(resp.request.body)
        return resp.json() if resp.content and "json" in resp.headers.get("content-type","") else None

    def sleep(self, tags=None):
        if tags is None or "kv_cache" in tags:
            self.post("flush_cache?timeout=10", {})
        self.post("release_memory_occupation", {"tags": tags} if tags else {})

    def wake(self, tags=None):
        self.post("resume_memory_occupation", {"tags": tags} if tags else {})

    @staticmethod
    def read_rope(path):
        config = json.loads((Path(path) / "config.json").read_text())
        if (config.get("architectures") == ["Qwen2ForCausalLM"] and not config.get("rope_scaling")
                and not config.get("use_sliding_window") and not config.get("use_mrope")
                and not config.get("dual_chunk_attention_config")):
            return float(config.get("rope_theta", 10000))
        return None

    def restore_rope(self, path, force=False):
        import torch
        import base64
        import pickle
        from .qwen2_rope import LOADER

        theta = self.read_rope(path)
        if theta is None or (not force and theta == self.rope_theta):
            return False
        parameter = [("rope_theta", torch.tensor(theta, dtype=torch.float64))]
        # Transfer the tiny CPU scalar by value. HTTP workers have independent
        # multiprocessing authkeys and cannot redeem a driver-owned CPU storage FD.
        scalar = base64.b64encode(pickle.dumps(parameter, protocol=4)).decode("ascii")
        handles = [scalar] * self.tp_size
        result = self.post("update_weights_from_tensor", dict(serialized_named_tensors=handles,
                           load_format=LOADER, flush_cache=True))
        if not result["success"]:
            raise RuntimeError(f"SGLang refused rotary cache update: {result}")
        self.rope_theta = theta
        return True

    def publish(self, path, version):
        tag = f"v{version}" if isinstance(version, int) else version
        result = self.post("update_weights_from_disk", dict(model_path=str(path), load_format="safetensors",
                                                           flush_cache=True, weight_version=tag))
        if not result["success"]:
            raise RuntimeError(f"SGLang refused {tag}: {result}")
        self.confirm_version(tag)

    def publish_bucket(self, rank_handles, version, last):
        body = dict(serialized_named_tensors=rank_handles, load_format="flattened_bucket", flush_cache=last)
        if last:
            body["weight_version"] = f"v{version}" if isinstance(version, int) else version
        result = self.post("update_weights_from_tensor", body)
        if not result["success"]:
            raise RuntimeError(f"SGLang rejected weight bucket for v{version}: {result}")
        if last:
            self.confirm_version(body["weight_version"])

    def confirm_version(self, tag):
        info = self.session.get(self.url+"/model_info", timeout=10)
        info.raise_for_status()
        if info.json()["weight_version"] != tag:
            raise RuntimeError("SGLang did not acknowledge weight publication")
        self.version = tag

    def batch(self, items, params, **options):
        requests_ids = [f"{x['rid']}/{uuid4().hex}" for x in items]
        body = dict(input_ids=[x["prompt"]+x["tokens"] for x in items],
                    rid=requests_ids, sampling_params=params, **options)
        reply = self.post("generate", body)
        if not isinstance(reply, list) or len(reply) != len(items):
            raise RuntimeError("incomplete SGLang response")
        pairs = []
        for rid, item, result in zip(requests_ids, items, reply, strict=True):
            m = result["meta_info"]
            if (m["id"] != rid or m["weight_version"] != self.version or
                    m["finish_reason"]["type"] not in ("stop", "length")):
                raise RuntimeError(f"bad SGLang receipt for {rid}: {m['finish_reason']}")
            pairs.append((item, result, m))
        return pairs

    def close(self):
        if self.proc.poll() is None:
            os.killpg(self.proc.pid, signal.SIGTERM)
            try:
                self.proc.wait(timeout=12)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
        self.session.close()


class Inference:
    def __init__(self, cfg):
        self.cfg = cfg
        self.excluded_ids = frozenset(cfg.get("excluded_ids", ()))
        self.sampled_id_limits = cfg.get("sampled_id_limits")
        self.owned = {}
        self.cached = {}

    @staticmethod
    def sampling_state(x):
        return struct.pack(">QQ", x["seed"], len(x["prompt"])+len(x["tokens"]))

    def decode(self, engine, p):
        items = p["items"]
        for x in items:
            if x["tokens"] and self.owned.get(x["rid"]) != self.sampling_state(x):
                raise RuntimeError("decode without valid sampling ownership")
        params = [dict(max_new_tokens=min(p["steps"], x["limit"]-len(x["tokens"])), temperature=1.,
                       top_p=1., top_k=-1, sampling_seed=x["seed"], no_stop_trim=True,
                       stop_token_ids=self.cfg["eos_ids"]) for x in items]
        updates, hits = {}, {}
        for x, result, meta in engine.batch(items, params, return_logprob=True):
            ids = result["output_ids"]
            entries = meta["output_token_logprobs"]
            if not ids or [entry[1] for entry in entries] != ids:
                raise RuntimeError("misaligned SGLang output logprobs")
            if self.sampled_id_limits is None:
                unexpected = self.excluded_ids.intersection(ids)
            else:
                limit = self.sampled_id_limits[x["teacher"]]
                unexpected = {token for token in ids if not 0 <= token < limit}
            if unexpected:
                raise ValueError(f"{x['rid']} sampled IDs unsupported for teacher {x['teacher']}: {sorted(unexpected)}")
            start = len(x["tokens"])
            finish = None
            if ids[-1] in self.cfg["eos_ids"]:
                finish = "eos"
            elif meta["finish_reason"]["type"] == "stop":
                raise RuntimeError("unrecognized SGLang stopping token")
            elif start+len(ids) == x["limit"]:
                finish = "length"
            rid = x["rid"]
            updates[rid] = dict(tokens=ids, start=start, logp=[q[0] for q in entries], finish=finish)
            hits[rid] = meta["cached_tokens"]
            if finish:
                self.owned.pop(rid, None)
            else:
                self.owned[rid] = struct.pack(">QQ", x["seed"], len(x["prompt"])+start+len(ids))
        result = dict(updates=updates, cached_tokens=hits, engine_weight_version=engine.version)
        if hasattr(engine, "last_batch_sizes"):
            result.update(inference_tp_size=engine.width, replica_requests=engine.last_batch_sizes)
        return result

    def score(self, engine, p):
        items = p["items"]
        starts = [len(x["prompt"])+x["start"]-1 for x in items]
        k = self.cfg.get("teacher_top_k", 0)
        reply = engine.batch(items, dict(max_new_tokens=0, temperature=1.), return_logprob=True,
                             logprob_start_len=starts, top_logprobs_num=k)
        scores = {}
        for x, _, meta in reply:
            # SGLang returns a leading (None, predecessor) for EACH requested slice.
            entries = meta["input_token_logprobs"]
            expected = (x["prompt"]+x["tokens"])[len(x["prompt"])+x["start"]-1:]
            if [e[1] for e in entries] != expected or entries[0][0] is not None:
                raise RuntimeError(f"SGLang chunk boundary mismatch for {x['rid']}")
            rid, end = x["rid"], len(x["tokens"])
            scores[rid] = dict(start=x["start"], end=end, labels=[e[0] for e in entries[1:]],
                               reused=meta["cached_tokens"], retained=not x["final"])
            if k:
                candidates = meta["input_top_logprobs"]
                if len(candidates) != len(entries) or candidates[0] is not None:
                    raise RuntimeError(f"SGLang top-k boundary mismatch for {rid}")
                topk = [[(token, logp) for logp,token,_ in position] for position in candidates[1:]]
                if any(token in self.excluded_ids for position in topk for token,_ in position):
                    raise ValueError("teacher top-k contains IDs outside the shared token mapping")
                scores[rid]["topk"] = topk
            if x["final"]:
                self.cached.pop(rid, None)
            else:
                lane = engine.lane(x["seed"]) if hasattr(type(engine), "lane") else 0
                self.cached[rid] = (lane, len(x["prompt"]) + end)
        lanes, dropped = trim_score_cache(self.cached, self.cfg["teacher_kv_tokens"])
        for lane in lanes:
            if hasattr(type(engine), "flush_score_lane"):
                engine.flush_score_lane(lane)
            else:
                engine.post("flush_cache?timeout=10", {})
        result = dict(scores=scores, evicted=dropped, flushed_score_lanes=lanes)
        if hasattr(engine, "last_batch_sizes"):
            result.update(inference_tp_size=engine.width, replica_requests=engine.last_batch_sizes)
        return result


    def install(self, engine, p):
        rebuild = p.get("rebuild", "eager")
        if rebuild not in ("eager", "on_decode"):
            raise ValueError("unknown migration prefill mode")
        items = p["items"]
        for x in items:
            if x["rid"] in self.owned or p["rng"][x["rid"]] != self.sampling_state(x):
                raise RuntimeError("migration RNG/position mismatch")
        contexts = [len(x["prompt"])+len(x["tokens"]) for x in items]
        resident = sum(struct.unpack(">QQ", state)[1] for state in self.owned.values())
        needed = resident+sum(contexts)+(len(self.owned)+len(items))*self.cfg["quantum"]
        capacity = min(self.cfg["engine_kv_tokens"], engine.capacity)
        fits = needed <= capacity
        if hasattr(type(engine), "migration_fits"):
            loads = [struct.unpack(">QQ", state) for state in self.owned.values()]
            loads.extend((x["seed"], n) for x,n in zip(items, contexts, strict=True))
            fits = engine.migration_fits(loads, self.cfg["quantum"], self.cfg["engine_kv_tokens"])
            capacity = sum(min(s.capacity, self.cfg["engine_kv_tokens"]) for s in engine.servers)
        if not fits:
            return dict(accepted=False,required_kv_tokens=needed,capacity_tokens=capacity)
        if rebuild == "on_decode":
            # SGLang's next generate RPC includes the whole committed prefix.
            # Admit the exact sampling state before the source lease is released.
            self.owned.update(p["rng"])
            return dict(accepted=True, rebuilt_tokens=0, deferred_context_tokens=sum(contexts), kv_bytes=0,
                        kv_transfer_bytes=0, capacity_tokens=capacity, transfer="seed_position_then_decode")
        reply = engine.batch(items, dict(max_new_tokens=0), return_logprob=False)
        hits = [meta["cached_tokens"] for _,_,meta in reply]
        self.owned.update(p["rng"])
        return dict(accepted=True, rebuilt_tokens=sum(contexts)-sum(hits), hit_tokens=sum(hits),
                    kv_bytes=sum(contexts)*self.cfg["kv_bytes_per_token"], kv_transfer_bytes=0,
                    prefill_http_body_bytes=engine.generate_body_bytes,capacity_tokens=capacity,
                    transfer="seed_position_and_reprefill")
