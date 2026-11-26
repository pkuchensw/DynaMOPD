"""Prepare a private KDFlow package using its algorithm registry and original trainer."""
import hashlib
from pathlib import Path
import shutil


def install(repo, work, *, synchronous_teacher=False, keep_optimizer=False, engine_options=None, devices=None,
            external_ray=False):
    package = work / "kdflow"
    shutil.copytree(repo / "kdflow", package, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    adapter = Path(__file__).with_name("kdflow_k1.py")
    shutil.copyfile(adapter, package / "algorithms" / "verl_k1.py")
    arguments = package / "arguments/__init__.py"
    source = arguments.read_text()
    gate = 'if args.kd.kd_algorithm != "vanilla_kd":'
    if source.count(gate) != 1:
        raise ValueError("unexpected KDFlow multi-teacher registry gate")
    arguments.write_text(source.replace(gate, 'if args.kd.kd_algorithm not in ("vanilla_kd", "verl_k1"):'))
    # Token-compatible teachers can expose different numbers of padded output
    # rows. Negative infinity adds zero softmax mass for the sampled K1 loss.
    vanilla = package / "algorithms/vanilla_kd.py"
    body = vanilla.read_text()
    combine = "        return torch.cat([x for x in logits_list if x is not None], dim=0)"
    if body.count(combine) != 1:
        raise ValueError("unexpected KDFlow multi-teacher head assembly")
    vanilla.write_text(body.replace(combine, """        pieces = [x for x in logits_list if x is not None]
        width = max(x.shape[-1] for x in pieces)
        return torch.cat([torch.nn.functional.pad(x, (0, width - x.shape[-1]), value=-torch.inf)
                          if x.shape[-1] < width else x for x in pieces], dim=0)"""))
    processor = package / "trainer/data_processor/rollout_data_processor.py"
    body = processor.read_text()
    # The matched models share token IDs. Send the originally sampled student
    # prompt and response to every teacher, including length-limited samples.
    start = body.index("        teacher_processor = self._get_teacher_processor(teacher_routing_key)", body.index("    def _build_rollout_sample("))
    end = body.index("        prompt_length = stu_tokens", start)
    processor.write_text(body[:start] + '''        tea_tokens = {name.replace("stu_", "tea_", 1): value for name, value in stu_tokens.items()}
        tea_prompt = stu_prompt

''' + body[end:])
    trainer = package / "trainer/on_policy_kd_trainer.py"
    source = trainer.read_text()
    anchor = '            avg_micro_batch_token_num = batch_tokens / len(global_batch)'
    if source.count(anchor) != 1:
        raise ValueError("unknown KDFlow global normalization implementation")
    if keep_optimizer:
        # Use KDFlow's native sleep tags. Adam moments stay on their owning
        # FSDP ranks while the model and teacher heads still release memory.
        for original, replacement in (
            ('self.student.sleep(tags=["optim", "teacher_lm_head"])', 'self.student.sleep(tags=["teacher_lm_head"])'),
            ('self.student.sleep()', 'self.student.sleep(tags=["model", "teacher_lm_head"])'),
        ):
            if original not in source:
                raise ValueError("unknown KDFlow student sleep policy")
            source = source.replace(original, replacement)
    trainer.write_text(source.replace(anchor, '            print("matched_training_tokens=" + str(int(batch_tokens)), flush=True)\n' + anchor))
    files = [vanilla, package / "algorithms/verl_k1.py", processor, trainer, arguments]
    if external_ray:
        driver = package / "cli/train_kd_on_policy.py"
        body = driver.read_text()
        start_local = 'address="local", num_cpus=64, include_dashboard=False, object_store_memory=64*1024**3,'
        if body.count(start_local) != 1:
            raise ValueError("unexpected KDFlow Ray initialization")
        driver.write_text(body.replace(start_local, 'address=os.environ["RAY_ADDRESS"],'))
        files.append(driver)
    if devices is not None and list(devices) != list(range(len(devices))):
        # KDFlow's colocated workers inherit our visible mask. Ray reports host GPU
        # IDs, while SGLang and torch expect ordinals within that mask.
        placement = package / "ray/placement_group.py"
        actor = package / "ray/train/student_actor.py"
        remap = {str(device): index for index, device in enumerate(devices)}
        for path, old, new in (
            (placement, "return pg, reordered_bundle_indices, reordered_gpu_ids",
             f"return pg, reordered_bundle_indices, [{remap!r}[str(int(gpu))] for gpu in reordered_gpu_ids]"),
            (actor, "str(ray.get_gpu_ids()[0]) if ray_noset_visible_devices()",
             f"str({remap!r}[str(int(ray.get_gpu_ids()[0]))]) if ray_noset_visible_devices()"),
        ):
            source = path.read_text()
            if source.count(old) != 1:
                raise ValueError(f"unexpected KDFlow GPU placement in {path.name}")
            path.write_text(source.replace(old, new))
            files.append(path)
    if synchronous_teacher or engine_options:
        engine = package / "backend/sglang/sglang_engine.py"
        source = engine.read_text()
        anchor = "        engine = PatchedEngine(\n"
        if source.count(anchor) != 1:
            raise ValueError("unexpected KDFlow teacher engine constructor")
        call = anchor
        if engine_options:
            # Both paths have hard-coded SGLang defaults in this checkout.
            settings = package / "benchmark_sglang.py"
            settings.write_text("OPTIONS = " + repr(engine_options) + "\n\n"
                                "def teacher_engine(factory, **native_options):\n"
                                "    return factory(**(native_options | OPTIONS))\n")
            call = ("        from kdflow.benchmark_sglang import teacher_engine\n"
                    "        engine = teacher_engine(PatchedEngine,\n")
            rollout = package / "ray/rollout/rollout_actor.py"
            body = rollout.read_text()
            trace_server = package / "ray/rollout/benchmark_trace.py"
            shutil.copyfile(Path(__file__).with_name("kdflow_rollout_trace.py"), trace_server)
            server_import = "from sglang.srt.entrypoints.http_server import launch_server"
            if body.count(server_import) != 1:
                raise ValueError("unexpected KDFlow rollout server entry point")
            body = body.replace(server_import, "from kdflow.ray.rollout.benchmark_trace import launch_server")
            marker = "        server_arg_field_names = "
            if body.count(marker) != 1:
                raise ValueError("unexpected KDFlow rollout server arguments")
            rollout.write_text(body.replace(marker, "        from kdflow.benchmark_sglang import OPTIONS\n"
                                                    "        server_args_dict.update(OPTIONS)\n" + marker))
            # The same prompt and round get the same counter-based sampling seed.
            manager = package / "trainer/rollout_manager.py"
            body = manager.read_text()
            start = "                sampling_params,\n                image_data=images,"
            per_request = "                        sampling_params=sampling_params,\n                        session=session,"
            if body.count(start) != 1 or body.count(per_request) != 1:
                raise ValueError("unexpected KDFlow rollout seed path")
            body = body.replace(start, "                dict(sampling_params, sampling_seed=self.args.train.seed + "
                                "max(0, global_step - 1) * self.args.train.train_batch_size),\n"
                                "                image_data=images,")
            manager.write_text(body.replace(per_request, "                        sampling_params=dict(sampling_params, "
                                    "sampling_seed=sampling_params.get('sampling_seed', self.args.train.seed) + index),\n"
                                    "                        session=session,"))
            files.extend([settings, rollout, manager, trace_server])
        if synchronous_teacher:
            call += "            disable_overlap_schedule=True,\n"
        engine.write_text(source.replace(anchor, call))
        files.append(engine)
    return {str(path.relative_to(package)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files}
