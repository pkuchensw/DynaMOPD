"""Matched 8-H20 DAPO runs using the previously tuned, native KDFlow and verl commands."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time

from DynaMOPD.cluster.topology import visible_devices
from DynaMOPD.trainer.config import resolve_model_paths
from DynaMOPD.tools.benchmark_layout import benchmark_nodes, check_command, check_layouts

ROOT = Path(__file__).resolve().parents[1]


def benchmark_profile(path):
    root = path or os.environ.get("MOPD_PROFILE_ROOT")
    if not root:
        raise ValueError("set --profile-root or MOPD_PROFILE_ROOT to your baseline data and command directory")
    return Path(root).expanduser().resolve()


def prepare(out, batch, rounds, cap, devices, group_size=2, systems=("dynamopd","kdflow"), tp_size=1, *, profile_root=None):
    import pyarrow.parquet as pq
    data = benchmark_profile(profile_root)/"data"
    records = [json.loads(line) for line in (data/"dapo.jsonl").read_text().splitlines()]
    training = records[:batch*rounds]
    table = pq.read_table(data/"dapo.parquet").slice(0,len(training))
    if len(training) != batch*rounds:
        raise ValueError("the dataset has too few distinct rows for the requested steps")
    if [x["prompt"] for x in training] != table.column("prompt").to_pylist():
        raise ValueError("KDFlow and verl baseline prompt order differs")
    cfg = json.loads((ROOT/"configs/sglang-tp1-fsdp2-8xh20.json").read_text())
    groups = len(devices)//group_size
    cfg.update(algorithm="on_policy_k1", teachers=json.loads((data/"teachers.json").read_text()), batch_size=batch, rounds=rounds,
               groups=groups, group_size=group_size, benchmark_devices=devices, decode_batch=batch//groups*3//2,
               inference_tp_size=tp_size, max_train_groups=groups, verl_teacher_gpus=len(devices)//2,
               mem_fraction_static=.45, microbatch=8*group_size, train_budget=12288*group_size,
               kdflow_keep_optimizer=True, kdflow_synchronous_teacher=True,
               max_prompt=2048, max_new_tokens=cap, seed=42, lr=1e-6, tail_prior=640,
               checkpoint_interval=rounds, checkpoint_contents=["hf_model"], hf_checkpoint_dtype="float32",
               weight_sync="tensor", trainer_attention="flash_attention_2", train_on_length=True,
               benchmark_systems=list(systems))
    # KDFlow's hidden-state teacher needs a complete prefill for each sequence.
    cfg["sglang_limits"] = dict(max_running_requests=128, cuda_graph_max_bs=128,
                                max_prefill_tokens=max(16384, cfg["max_prompt"]+cap+1), chunked_prefill_size=-1)
    cfg["rollout_sampling_mode"] = "batch_invariant"
    cfg["cuda_graph"] = True
    resolve_model_paths(cfg)
    check_layouts(cfg, systems)
    out.mkdir(parents=True,exist_ok=False)
    (out/"train.jsonl").write_text("".join(json.dumps(x)+"\n" for x in training))
    pq.write_table(table,out/"train.parquet")
    evaluation = [json.loads(x) for x in (data/"pilot.jsonl").read_text().splitlines()][:32]
    train_prompts = {str(x["prompt"]) for x in training}
    assert len(evaluation)==32 and all(str(x["prompt"]) not in train_prompts for x in evaluation)
    (out/"heldout.jsonl").write_text("".join(json.dumps(x)+"\n" for x in evaluation))
    (out/"config.json").write_text(json.dumps(cfg,indent=2))


def replace_flags(args, changes):
    result = list(args)
    for k,v in changes.items():
        key = "--"+k
        if key in result:
            result[result.index(key)+1] = str(v)
        else:
            result.extend([key,str(v)])
    return result


def replace_hydra(args, changes):
    result = list(args)
    for k,v in changes.items():
        positions = [i for i,x in enumerate(result) if x.startswith(k+"=")]
        if positions:
            result[positions[0]] = k+"="+str(v)
        else:
            result.append(k+"="+str(v))
    return result


def run(system, out, state, backup_overlay=None, verl_keep_optimizer=False, *, profile_root=None, command_file=None):
    profile = benchmark_profile(profile_root) if system != "dynamopd" else None
    cfg = json.loads((out/"config.json").read_text())
    if cfg.get("algorithm") != "on_policy_k1" or cfg.get("train_on_length") is not True:
        raise ValueError("matched benchmark uses native verl K1 and its full response mask; run prepare for a fresh configuration")
    batch, rounds, cap = cfg["batch_size"], cfg["rounds"], cfg["max_new_tokens"]
    devices=cfg.get("benchmark_devices")
    if devices is None:
        devices = visible_devices()[:cfg["groups"] * cfg["group_size"]]
    n_gpus=len(devices)
    nodes = benchmark_nodes(cfg)
    address = cfg.get("cluster", {}).get("address")
    if nodes > 1 and (not address or address in ("local", "auto")):
        raise ValueError("multi-node benchmark requires an explicit Ray cluster address")
    if "inference_tp_size" not in cfg or "verl_teacher_gpus" not in cfg:
        raise ValueError("prepare a new benchmark directory with an explicit shared layout")
    if system not in cfg["benchmark_systems"]:
        raise ValueError(f"{system} is outside this comparison's declared systems")
    parallel = check_layouts(cfg, cfg["benchmark_systems"])
    tp = parallel["rollout_tp"]
    shard = parallel["fsdp_shard_size"]
    micro = parallel["microbatch_per_gpu"]
    budget = parallel["token_budget_per_gpu"]
    prompt = cfg["max_prompt"]
    memory = cfg["mem_fraction_static"]
    if cfg["rollout_sampling_mode"] != "batch_invariant":
        raise ValueError("the matched SGLang benchmark uses native deterministic kernels in both systems")
    limits = cfg["sglang_limits"]
    engine_options = dict(limits, context_length=prompt+cap+1, max_total_tokens=cfg["engine_kv_tokens"],
                          attention_backend="triton", dtype="bfloat16", enable_deterministic_inference=True, disable_piecewise_cuda_graph=True)
    label = cfg.get("dynamopd_run","dynamopd") if system=="dynamopd" else system
    target = out/label
    target.mkdir()
    (state/label).mkdir(parents=True)
    started = time.time()
    model_staging_s = None
    env = dict(os.environ)
    env.setdefault("PYTHON", sys.executable)
    env.update(OMP_NUM_THREADS="4",OPENBLAS_NUM_THREADS="1",HF_HUB_OFFLINE="1",HF_DATASETS_OFFLINE="1",
               CUDA_VISIBLE_DEVICES=",".join(map(str,devices)),WANDB_MODE="disabled",TOKENIZERS_PARALLELISM="false",
               HYDRA_FULL_ERROR="1",CUDA_DEVICE_MAX_CONNECTIONS="1",RAY_DEDUP_LOGS="0",
               RAY_enable_worker_prestart="0",RAY_prestart_worker_first_driver="0",
               RAY_worker_register_timeout_seconds="600",NCCL_DEBUG="WARN",PYTHONUNBUFFERED="1")
    for key in ("http_proxy","https_proxy","HTTP_PROXY","HTTPS_PROXY","ALL_PROXY","all_proxy","RAY_ADDRESS"):
        env.pop(key,None)
    if system=="dynamopd":
        repo = ROOT.parent
        snapshot = target/"launch-config.json"
        snapshot.write_text(json.dumps(cfg, indent=2))
        cmd = ["bash",str(ROOT/"launch.sh"),"--config",str(snapshot),"--data",str(out/"train.jsonl"),
               "--out",str(target),"--state-dir",str(state/label)]
    else:
        from DynaMOPD.workers.model import stage_model
        cache = state/system/"initial_models"
        cache.mkdir()
        sources=list(dict.fromkeys([cfg["student"],*cfg["teachers"].values()]))
        with ThreadPoolExecutor(max_workers=len(sources)) as io:
            staged=dict(zip(sources,io.map(lambda path:stage_model(path,cache),sources),strict=True))
        model_staging_s=time.time()-started
        teacher_paths={key:staged[path] for key,path in cfg["teachers"].items()}
        repo = profile/"repos"/("KDFlow" if system=="kdflow" else "verl")
        env.update(PYTHONPATH=f"{profile}:{repo}",MOPD_TRACE_DIR=str(target/"events"),
                   # Ray appends a session directory and an AF_UNIX socket name.
                   RAY_TMPDIR=tempfile.mkdtemp(prefix="dmray-"))
        if backup_overlay:
            overlay = backup_overlay.resolve()
            if not (overlay/"SOURCE_SHA256").is_file():
                raise ValueError("build the pageable backup overlay first")
            env["PYTHONPATH"] = f"{overlay}:{env['PYTHONPATH']}"
            profile_hook = profile/"sitecustomize.py"
            if profile_hook.is_file():
                env["DYNAMOPD_PROFILE_SITECUSTOMIZE"] = str(profile_hook)
        command_file = Path(command_file) if command_file else profile/"results"/f"{system}_dapo_1"/"run.json"
        base = json.loads(command_file.read_text())["command"]
        if system=="kdflow":
            from DynaMOPD.tools.baselines.kdflow import install
            import torch
            work = state/system/"native-k1"
            synchronous_teacher = cfg.get("kdflow_synchronous_teacher", False)
            keep_optimizer = cfg.get("kdflow_keep_optimizer", False)
            overlay_files = install(repo, work, synchronous_teacher=synchronous_teacher, keep_optimizer=keep_optimizer,
                                    engine_options=engine_options, devices=devices, external_ray=nodes > 1)
            if nodes > 1:
                env["RAY_ADDRESS"] = address
            env["PYTHONPATH"] = f"{work}:{ROOT.parent}:{env['PYTHONPATH']}"

            # KDFlow's sampler always uses randperm(seed + epoch). Invert its
            # epoch-zero permutation on disk so each optimizer sees the same prompt set.
            source = (out/"train.jsonl").read_text().splitlines()
            order = torch.randperm(len(source),generator=torch.Generator().manual_seed(cfg["seed"])).tolist()
            sampler_rows = [""]*len(source)
            for index,line in zip(order,source,strict=True):
                sampler_rows[index] = line
            kd_data = target/"sampler-input.jsonl"
            kd_data.write_text("\n".join(sampler_rows)+"\n")
            teacher_file = target/"teachers.json"
            teacher_file.write_text(json.dumps(teacher_paths,indent=2))
            cmd = replace_flags(base,dict(train_batch_size=batch,rollout_batch_size=batch,generate_max_len=cap,
                        student_name_or_path=staged[cfg["student"]],multi_teacher_config=teacher_file,
                        num_nodes=nodes,num_gpus_per_node=n_gpus,rollout_tp_size=tp,rollout_num_engines=parallel["rollout_replicas"],
                        teacher_dp_size=parallel["teacher_replicas"],teacher_tp_size=tp,teacher_pp_size=1,
                        fsdp_size=shard,micro_train_batch_size=micro,max_token_len_per_gpu=budget,
                        rollout_mem_fraction_static=memory,teacher_mem_fraction_static=memory,
                        seed=cfg["seed"],prompt_max_len=prompt,max_len=prompt+cap,temperature=cfg["temperature"],
                        top_p=1.,train_dataset_path=kd_data,
                        save_path=state/system/"checkpoint",learning_rate=cfg["lr"]))
            # KDFlow passes min_lr into its native cosine scheduler. Matching
            # floor and initial LR give a constant learning rate for every update.
            cmd=replace_flags(cmd,dict(min_lr=cfg["lr"],lr_warmup_ratio=0.,max_norm=cfg["clip"],
                                       weight_decay=cfg.get("weight_decay",0.),adam_betas="(0.9,0.999)"))
            cmd = replace_flags(cmd, dict(kd_loss_fn="verl_k1", kd_algorithm="verl_k1"))
        else:
            replicas=parallel["teacher_replicas"]
            actor_gpus = parallel["rollout_replicas"] * tp
            teacher_gpus = replicas * tp
            if nodes > 1:
                local_init = ("ray_kwargs.ray_init.num_cpus", "ray_kwargs.ray_init.object_store_memory",
                              "ray_kwargs.ray_init.include_dashboard", "ray_kwargs.ray_init.address")
                base = [arg for arg in base if arg.lstrip("+").split("=", 1)[0] not in local_init]
                base = replace_hydra(base, {"+ray_kwargs.ray_init.address": address})
            cmd = replace_hydra(base,{"data.train_batch_size":batch,"data.train_files":out/"train.parquet",
                        "actor_rollout_ref.model.path":staged[cfg["student"]],
                        "actor_rollout_ref.actor.optim.lr":cfg["lr"],
                        "actor_rollout_ref.actor.optim.weight_decay":cfg.get("weight_decay",0.),
                        "trainer.nnodes":max(1,actor_gpus//n_gpus),"distillation.nnodes":max(1,teacher_gpus//n_gpus),
                        "trainer.n_gpus_per_node":min(n_gpus,actor_gpus),"distillation.n_gpus_per_node":min(n_gpus,teacher_gpus),
                        "actor_rollout_ref.rollout.tensor_model_parallel_size":tp,
                        "actor_rollout_ref.actor.fsdp_config.fsdp_size":shard,
                        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu":micro,
                        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu":budget,
                        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu":micro,
                        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu":budget,
                        "actor_rollout_ref.rollout.gpu_memory_utilization":memory,
                        "actor_rollout_ref.actor.fsdp_config.seed":cfg["seed"],
                        "actor_rollout_ref.rollout.seed":cfg["seed"],"data.max_prompt_length":prompt,
                        "data.val_files":out/"train.parquet","data.max_response_length":cap,"data.shuffle":"false",
                        "actor_rollout_ref.actor.ppo_mini_batch_size":batch,"actor_rollout_ref.rollout.temperature":cfg["temperature"],
                        "actor_rollout_ref.rollout.top_p":1.,"actor_rollout_ref.rollout.max_model_len":prompt+cap+1,
                        "trainer.total_training_steps":rounds,"trainer.default_local_dir":state/system/"checkpoint",
                        "trainer.save_freq":rounds,"actor_rollout_ref.actor.checkpoint.save_contents":"[hf_model]",
                        "trainer.rollout_data_dir":target/"outputs"})
            for i, key in enumerate(cfg["teachers"]):
                prefix = f"+distillation.teacher_models.{key}.inference."
                cmd = replace_hydra(cmd,{prefix+"response_length":cap,prefix+"max_model_len":prompt+cap+1,
                            prefix+"tensor_model_parallel_size":tp,prefix+"gpu_memory_utilization":memory,
                            prefix+"prompt_length":prompt,
                            f"+distillation.teacher_models.{key}.num_replicas":replicas//len(cfg["teachers"])+int(i<replicas%len(cfg["teachers"])),
                            f"+distillation.teacher_models.{key}.model_path":teacher_paths[key]})
            for prefix in ["actor_rollout_ref.rollout."] + [
                f"+distillation.teacher_models.{route}.inference." for route in cfg["teachers"]
            ]:
                cmd = replace_hydra(cmd, {prefix+"max_num_seqs":limits["max_running_requests"]})
                engine = prefix if prefix.startswith("+") else "+"+prefix
                overrides = {engine+"engine_kwargs.sglang."+key: value for key,value in engine_options.items()
                             if key not in ("max_running_requests","context_length","dtype")}
                cmd = replace_hydra(cmd, overrides)
            if verl_keep_optimizer:
                cmd = replace_hydra(cmd, {"actor_rollout_ref.actor.fsdp_config.optimizer_offload":"false"})
    check_command(cfg, system, cmd)
    meta = dict(command=cmd, cwd=str(repo), start=started, system=system, batch=batch, rounds=rounds,
                cap=cap, devices=devices, state=str(state/label))
    meta["sglang_options"] = engine_options
    meta["parallel_layout"] = parallel
    if nodes > 1:
        meta.update(nodes=nodes, total_gpus=nodes*n_gpus, cluster=cfg["cluster"])
    meta["config_sha256"] = hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest()
    meta["dataset_sha256"] = hashlib.sha256((out/"train.jsonl").read_bytes()).hexdigest()
    if system == "dynamopd":
        source = hashlib.sha256()
        for path in sorted(ROOT.rglob("*.py")):
            relative = path.relative_to(ROOT)
            if relative.parts[0] != "runs":
                source.update(str(relative).encode() + b"\0" + path.read_bytes() + b"\0")
        meta["source_sha256"] = source.hexdigest()
    if system!="dynamopd":
        meta["ray_tmpdir"] = env["RAY_TMPDIR"]
        patch = subprocess.check_output(["git","diff","HEAD"],cwd=repo)
        meta.update(commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip(),
                    initial_model_staging_s=model_staging_s,
                    source_diff_sha256=hashlib.sha256(patch).hexdigest(),
                    profile_command=str(command_file))
        if backup_overlay:
            meta.update(weight_backup="pageable synchronous D2H/H2D",backup_overlay=str(overlay),
                        backup_source_sha256=(overlay/"SOURCE_SHA256").read_text().strip())
    if system == "kdflow":
        meta.update(cwd=str(work), algorithm="verl_k1", adapter_files=overlay_files,
                    synchronous_teacher=synchronous_teacher, keep_optimizer=keep_optimizer)
    manifest = target/"run.json"
    pending = manifest.with_suffix(".tmp")
    pending.write_text(json.dumps(meta, indent=2))
    pending.replace(manifest)
    with (target/"run.log").open("w") as log:
        ret = subprocess.run(cmd,cwd=meta["cwd"],env=env,stdout=log,stderr=subprocess.STDOUT).returncode
    meta.update(end=time.time(), exit_code=ret)
    if ret == 0 and system == "kdflow":
        try:
            shutil.copytree(state/system/"checkpoint"/"rollout_data", target/"outputs")
        except OSError as error:
            meta.update(exit_code=1, artifact_error=str(error))
    pending.write_text(json.dumps(meta, indent=2))
    pending.replace(manifest)
    if meta["exit_code"]:
        raise RuntimeError(f"{system} failed; see {manifest} and {target}/run.log")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=("prepare","dynamopd","kdflow","verl"))
    p.add_argument("--out",required=True,type=Path)
    p.add_argument("--state-dir",type=Path)
    p.add_argument("--profile-root", type=Path, help="baseline profile containing data and repos (or MOPD_PROFILE_ROOT)")
    p.add_argument("--command-file", type=Path, help="baseline JSON manifest with a command argument list")
    p.add_argument("--pageable-backup",type=Path,help="job-local build_pageable_backup output for native baselines")
    p.add_argument("--verl-keep-optimizer",action="store_true",
                   help="keep native verl Adam states on GPU across rollout and training")
    p.add_argument("--batch",type=int,default=512)
    p.add_argument("--rounds",type=int,default=6)
    p.add_argument("--cap",type=int,default=8192)
    p.add_argument("--devices", help="comma-separated physical indices; defaults to the visible GPUs at prepare")
    p.add_argument("--tp-size", type=int, default=1, help="shared SGLang TP width for rollout and all teachers")
    p.add_argument("--group-size", type=int, default=2,
                   help="shared FSDP2 shard width")
    p.add_argument("--systems", nargs="+", choices=("dynamopd","kdflow","verl"), default=["dynamopd","kdflow"],
                   help="systems included in held-out evaluation and comparison")
    a=p.parse_args()
    out = a.out.resolve()
    if a.stage=="prepare":
        ids = a.devices.split(",") if a.devices else visible_devices()
        try:
            devices = [int(i) for i in ids]
        except ValueError:
            p.error("--devices for the native baselines requires physical GPU indices")
        if min(a.group_size, a.tp_size) < 1 or not devices or len(set(devices))!=len(devices) or min(devices)<0:
            p.error("choose distinct GPU indices and positive TP/FSDP widths")
        if "kdflow" in a.systems and (a.batch<8*len(devices) or a.batch%(8*len(devices))):
            p.error("native KDFlow microbatch 8 requires a batch divisible by 8 x number of GPUs")
        if len(devices)%a.group_size:
            p.error("the benchmark requires complete GPU groups")
        prepare(out,a.batch,a.rounds,a.cap,devices,a.group_size,list(dict.fromkeys(a.systems)),a.tp_size, profile_root=a.profile_root)
    else:
        if a.verl_keep_optimizer and a.stage != "verl":
            p.error("--verl-keep-optimizer is only valid for the native verl run")
        run(a.stage,out,(a.state_dir or out/"states").resolve(),a.pageable_backup,a.verl_keep_optimizer, profile_root=a.profile_root, command_file=a.command_file)


if __name__=="__main__":
    main()
