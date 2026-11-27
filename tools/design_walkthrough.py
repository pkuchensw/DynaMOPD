"""Render the offline DynaMOPD system design guide (stdlib only)."""
import argparse
import ast
import html
import json
import math
import os
from collections import defaultdict
from pathlib import Path
from urllib.parse import quote


ROOT = Path(__file__).resolve().parents[1]
OPS = {"decode": ("Student generation", "decode"), "score": ("Teacher scoring", "score"),
       "train": ("Microbatch training", "train"), "move": ("Request migration", "handoff"),
       "switch": ("Role switch", "handoff"), "tp_expand": ("TP expansion", "handoff"),
       "tp_shrink": ("TP contraction", "handoff"), "reduce_grads": ("Gradient reduction", "commit"),
       "commit": ("Optimizer step", "commit")}
LABELS = dict(decode="Decode", score="Score", train="Train", handoff="Handoff", commit="Reduce / publish")


def esc(value):
    return html.escape(str(value), quote=True)


def table(headers, rows):
    head = "".join(f'<th scope="col">{esc(h)}</th>' for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in rows)
    return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>'


def flow(nodes):
    return '<ol class="flow">' + ''.join(
        f'<li><strong>{esc(title)}</strong><small>{esc(detail)}</small></li>'
        for title, detail in nodes) + '</ol>'


def section(anchor, nav, title, lead, body, closing, refs, notes=()):
    return dict(anchor=anchor, nav=nav, title=title, lead=lead, body=body, closing=closing, refs=refs, notes=notes)


def source(ref, out):
    path, name, marker = ref
    target = ROOT / path
    label = f"{path} · {name}"
    if not target.is_file():
        return f'<p class="filename">{esc(label)}</p>'
    code = target.read_text(encoding="utf-8")
    tree = ast.parse(code)
    scope = tree
    for part in name.split("."):
        scope = next((n for n in scope.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
                      and n.name == part), None)
        if scope is None:
            return f'<p class="filename">{esc(label)}</p>'
    lines = code.splitlines()
    lo, hi = scope.lineno - 1, scope.end_lineno
    start = next((n for n in range(lo, hi) if marker and marker in lines[n]), lo)
    end = min(hi, start + 7)
    code_rows = "\n".join(f"{n+1:>4}  {lines[n]}" for n in range(start, end))
    href = quote(os.path.relpath(target, out.parent), safe="/")
    return (f'<div class="src"><a href="{esc(href)}">{esc(label)}</a>'
            f'<pre><code>{esc(code_rows)}</code></pre></div>')


def example_timeline():
    groups = [
        [("decode", 0, 69), ("handoff", 69, 75), ("score", 75, 80)],
        [("decode", 0, 33), ("handoff", 33, 40), ("score", 40, 82)],
        [("decode", 0, 48), ("handoff", 48, 56), ("train", 56, 74), ("train", 82, 89)],
    ]
    chart = ['<figure><figcaption>GPU group assignments and the optimizer step <small>Illustration, timings are schematic</small></figcaption>']
    for i, row in enumerate(groups):
        bars = ''.join(f'<span class="bar {k}" style="left:{a}%;width:{b-a}%">{LABELS[k]}</span>' for k,a,b in row)
        chart.append(f'<div class="lane"><b>G{i}</b><div class="track">{bars}</div></div>')
    chart.append('<div class="lane"><b>Global</b><div class="track plain"><span class="bar commit" style="left:89%;width:11%">Commit</span></div></div></figure>')
    return ''.join(chart)


def prefix_example():
    return ('<figure><figcaption>A trajectory through student migration and teacher cache eviction'
            '<small>One request ID. Offsets count committed response tokens.</small></figcaption>'
            + table(["Event", "Student group", "Generated tokens", "Scores delivered", "Teacher KV cache", "Ready to train"], [
                ("Student commits t₀…t₃", "G0", 4, 0, "—", "No"),
                ("Teacher returns [0, 4)", "G0", 4, 4, "G1 through offset 4", "No"),
                ("G2 takes over and generates t₄", "G2", 5, 4, "G1 through offset 4", "No"),
                ("G1 evicts the KV cache", "G2", 5, 4, "—", "No"),
                ("G2 commits t₅ (EOS)", "G2", 6, 4, "—", "No"),
                ("Teacher scores [4, 6)", "G2", 6, 6, "—", "Yes")])
            + '<p class="caption">After G1 evicts its cache, the next scoring pass rereads the context. The first four labels remain valid.</p></figure>')


def motivation_observation(trace):
    if trace is None:
        return ""
    version, record, backend, _ = trace
    intervals = defaultdict(list)
    for _, _, op, a, b, accepted, _, _ in record["spans"]:
        if op in ("decode", "score", "train") and accepted is not False and b > a:
            intervals[op].append((a, b))
    if intervals.keys() != {"decode", "score", "train"}:
        return ""
    start = record["start"]
    score = min(a for a, _ in intervals["score"]) - start
    train = min(a for a, _ in intervals["train"]) - start
    last = max(b for _, b in intervals["decode"]) - start
    # Sweep wall time once. Several concurrent batches of one type count as one active phase.
    changes = sorted((t, op, delta) for op, spans in intervals.items()
                     for a, b in spans for t, delta in ((a, 1), (b, -1)))
    active = dict.fromkeys(intervals, 0)
    previous = changes[0][0]
    joint_seconds = 0.0
    for t, op, delta in changes:
        if all(active.values()):
            joint_seconds += t - previous
        active[op] += delta
        previous = t
    clock = {"sglang_fsdp2": "hardware worker times", "simulation": "the simulation clock",
             "simrl": "the simulation clock"}.get(backend, "executor times")
    return (f'<aside class="observation"><strong>Observed overlap</strong><p>In round v{version}, scoring started at'
            f' {score:.2f} s, microbatch training started at {train:.2f} s, and the last generation batch returned at {last:.2f} s. '
            f'All three operations were in flight together for {joint_seconds:.2f} s. '
            f'These times use {clock} from this round and include waiting within batches. <a href="#case-study">View the group timeline ↗</a></p></aside>')


def design_sections(trace):
    return [
        section("motivation", "Motivation", "Synchronous updates and per-trajectory dependencies",
            "In synchronous multi-teacher on-policy distillation, response lengths can vary widely within a "
            "batch. If scoring waits for the last response, groups that finish early sit idle while their "
            "teachers wait. Reserving a fixed teacher pool allows scoring to start earlier but leaves fewer "
            "replicas for initial generation. DynaMOPD starts every group on rollout and reassigns groups as "
            "work finishes.",
            '<h3>When scoring and backpropagation can start</h3>'
            '<p><strong>Scoring a prefix.</strong> For a frozen causal teacher, the score for sampled token yᵢ is '
            '<code>log p<sub>T</sub>(yᵢ | x, y<sub>&lt;i</sub>)</code>. Later tokens cannot change this conditional distribution. '
            'As soon as the student commits a prefix, the assigned teacher has all inputs for that segment. One '
            'teacher batch can score segments from several trajectories.</p>'
            '<p><strong>Training while long requests generate.</strong> Once a sample terminates and all teacher '
            'scores arrive, its loss depends only on its data and the parameters θᵥ at the start of the round. '
            'A free group can accumulate gradients using θᵥ while other GPUs generate. At round end, the '
            'optimizer group collects gradients for all valid tokens, normalizes once, takes one step and '
            'publishes θᵥ₊₁.</p>'
            + motivation_observation(trace) + example_timeline(),
            "Overlap shortens a round when it moves work off the critical path. Retaining the layout can finish "
            "sooner if switching roles, repeating prefill or borrowing generation GPUs adds more time. The "
            "scheduler therefore optimizes for the completion time of the next weight publication.",
            [("trainer/driver.py","fit","Controller("),("scheduler/controller.py","Controller.run_round","for a in self.policy.plan")]),
        section("state", "Data and control", "Trajectory state and resource ownership",
            "Each trajectory has a (rid, student_version) key. The controller stores sampled tokens, behavior "
            "log probabilities, the teacher route and delivered labels. It tracks the student request owner, the "
            "number of delivered scores len(labels), and the reusable teacher KV offset kv[group] independently. "
            "A student can migrate while teacher caches are evicted.",
            flow([("Controller", "Choose an action from confirmed state"), ("Group lease", "Reserve the one or two groups required"),
                  ("Executor", "Run local or Ray workers, or advance the built-in simulator"), ("Completion", "Check the version and offsets, update state and release the lease")])
            + prefix_example() +
                '<h3>Scoring boundaries</h3><p>For interval [a, b), the worker requests input log probabilities '
                'from SGLang starting at '
            '<code>len(prompt) + a − 1</code>. The first item contains the preceding token and an empty score. '
            'The worker checks the returned token IDs, '
            'skips the predecessor and delivers exactly b−a scores. For the first segment, the preceding token '
            'is the last token of the prompt.</p>',
            "A sample is ready for training once it has terminated, its required scores are complete and no "
            "worker has claimed it. EOS always qualifies. With train_on_length enabled, reaching the length "
            "limit also qualifies.",
            [("scheduler/protocol.py","Row.deliver","self.labels.extend"),("scheduler/protocol.py","Row.ready","return"),
             ("backends/sglang_engine.py","Inference.score","starts =")],
            notes=[
                ("Fixed intervals for in-flight scores",
                    "Dispatch fixes [start, end) and the target token IDs. The student may keep generating. The "
                    "controller validates the completion against the dispatched interval and starts the next "
                    "segment at the delivered offset. An active route with no pending prefix may soon have work.",
                 ("scheduler/controller.py","Controller.materialize","item.update(start=start")),
                ("Boundary checks on the running engine",
                    "At startup, SGLang scores two prefixes of different lengths. The check compares scores at "
                    "each position across a full pass, a cached continuation and a recomputation after the cache "
                    "is cleared. For rollout it also compares tokens sampled continuously and after clearing the "
                    "cache with the same seed. The check uses fixed probes and numerical tolerances.",
                 ("backends/checks.py","check_server","for batch in (warmed, rebuilt)")),
                ("Validating sampled IDs per route",
                    "Input checks compare student and teacher token mappings and EOS. With matching mappings, K1 "
                    "validates SGLang token IDs against the output widths of the student and the teacher "
                    "assigned to that request. A smaller output layer on another teacher cannot invalidate those "
                    "tokens. The trajectory keeps its sampling seed when it moves GPUs.",
                 ("trainer/data.py","vocabulary","sampled_id_limits")),
            ]),
        section("model", "Cost model", "Event simulation and online calibration",
            "DynaMOPD's built-in simulator advances each inference instance in virtual time and releases "
            "its group when the instance drains. Its cost models and paged KV accounting use only the Python "
            "standard library. The virtual executor implements the same action and completion interface as the hardware "
            "executor. Hardware workers return when the operation finishes. The simulator inserts the predicted "
            "completion into its event queue.",
            table(["Operation","Estimated costs","Parallel execution"],[
                ("Student generation","Decode steps, total tokens, attention context, uncached input and KV page limits","Estimate each inference replica separately. The slowest replica determines batch completion."),
                ("Teacher scoring","Fixed overhead and input cost fitted per route, allowing for cached prefixes on the target replica","Deliver segments in offset order within each trajectory. Batch across trajectories."),
                ("Microbatch training","Forward and backward time from the packed or padded input width on each FSDP rank","Form microbatches from ready samples and use the slowest rank."),
                ("Role changes and publication","Switch direction, resident state, teacher startup and shutdown, local preparation and shared transfer","The publication barrier waits for every group. Count each shared transfer once.")])
            + '<p class="equation"><b>Simulated time per step</b><code>T = max(F, W + G·N + Aₚ·P/b + A𝒹·C)</code>'
            '<i>F is the minimum step time, W the fixed cost, N the tokens in the step, P the prefill attention '
            'work, C the sum of decode context lengths, and b the batch size.</i></p>'
            '<p>For a batched decode RPC, the fit also accounts for context accumulated by each request in the '
            'segment. Given an initial context of c tokens and s more steps, its contribution is '
            '<code>s·c + s(s−1)/2</code>. Generating the same number of tokens can therefore take different amounts of time.</p>',
            "On a new allocation, SGLang discards old timing coefficients. When building wide TP meshes, it "
            "measures wake, sleep and wide decode. The first training round serves requests and collects batch "
            "and role-switch times. Later searches fit and update costs from this job. Offline replay may use "
            "known final lengths. The online policy uses only previously reported progress and history. The "
            "trace records the CPU time spent planning separately in planner_wall_s.",
            [("backends/sim_model.py","KvCacheManager.allocate","def allocate"),
             ("backends/sim_model.py","UnifiedRoofline.step_time","def step_time"),("backends/simulation.py","Simulator.wait","heappop"),
             ("backends/sim_profile.py","decode_features","ticks ="),("scheduler/calibration.py","update_profile","fit_profile(cfg")],
            notes=[
                ("Measurements in the first training round",
                    "The standard SGLang policies discard timing coefficients from previous jobs and retain "
                    "topology constraints such as TP width and eligible group pairs. The first round generates, "
                    "scores and trains normally. If migration is enabled, calibration attempts a real handoff "
                    "when a pair becomes eligible. If the first round yields no measurement, a later round "
                    "retries. Probes stop once the base cost is measured. For operations with sparse samples, "
                    "the model can use coarser timings observed in this job.",
                 ("scheduler/calibration.py","calibration_round","if candidates:")),
                ("Accounting per physical replica",
                    "A group can host several narrow TP replicas. The model splits each replica's work into "
                    "windows that fit its KV pages, accounts for cold input and accumulated context, and takes "
                    "the slowest completion. If all observed decode segments fill their quantum, the fit cannot "
                    "distinguish startup cost from per-step cost. It attributes that cost to steps to avoid "
                    "overstating the cost of a short tail.",
                 ("tools/worker_costs.py","fit_decode","inseparable")),
                ("Waiting and transfer time",
                    "The offline publication model aligns arrival times at the collective, accounts for local "
                    "preparation on each group, and counts one shared transfer. First-round calibration on "
                    "hardware measures elapsed time until all groups pass the publication barrier. When teacher "
                    "CPU preparation overlaps unloading the previous role, the lease model charges only the wait "
                    "that remains after handoff.",
                 ("backends/sim_profile.py","teacher_lease_window","def teacher_lease_window")),
                ("Timing errors and scheduling errors",
                    "For offline analysis, set sim_execution_profile to model the virtual worker separately from "
                    "the scheduler's estimates. Replaying the hardware action order tests service times and "
                    "cache-hit estimates. Letting the scheduler choose actions also tests its decisions. "
                    "sim_timing_seed samples measured latency multipliers by group and operation, independently "
                    "of trajectory sampling seeds.",
                 ("tools/trace_timing.py","replay","The recorded order")),
            ]),
        section("scheduling", "Scheduling", "Choosing actions to minimize time to weight publication",
            "At each decision, the controller processes completed batches. The cost_search policy builds "
            "feasible layouts from request ownership, group availability, in-flight work, outstanding scores, "
            "ready samples and available memory. For each layout it compares scoring groups, scoring and "
            "training batch sizes, and the choice of serving now or waiting for more prefixes.",
            table(["Candidate layout","Potential benefit","Costs in the forecast"],[
                ("Continue generation until groups drain","Keep current generation parallelism","Scoring and ready microbatches wait for a free group"),
                ("Keep teacher A resident while another group serves B or trains","Advance multiple routes or training batches concurrently","Draining existing work, loading roles, waiting to form batches and evicting teacher KV"),
                ("Merge requests from two instances","Free the source group earlier for scoring or training",
                    "Waiting for both safe points, destination admission, prefix recomputation and additional "
                    "generation load on the destination"),
                ("Widen TP for a tail request, then return the group","Shorten each remaining decode step on a long sequence","Entering and leaving the wide layout and delaying other work while GPUs are on loan")])
            +
                '<p>The predictor advances each layout along its data dependencies and group occupancy '
                'intervals. The controller chooses the layout with the earliest predicted publication and '
                'dispatches independent actions to their groups. '
            'It can dispatch decode work common to all candidates on uncontested groups before the CPU search finishes.</p>',
            "The calibration round uses its own allocation policy. At cold start, the response cap can stand in "
            "for a potential long tail. Later forecasts use length history per route and the current prefix. For "
            "the final active request they can also use an exponential moving average (EMA) of the round "
            "maximum. The layout and search budget limit the candidates considered.",
            [("scheduler/search.py","CostSearchPolicy.plan","f = Forecast("),("scheduler/search.py","Forecast.project","def project"),
             ("scheduler/search.py","CostSearchPolicy.plan","estimates.append"),("scheduler/controller.py","Controller.submit","overlapping exclusive leases")],
            notes=[
                ("Initial placement and replica load",
                    "Optional balanced placement estimates work from historical response lengths per route and "
                    "prompt cost. It assigns heavier requests first among replicas with the same index and "
                    "preserves each group's original slot count. It adopts the placement only if the predicted "
                    "peak load decreases. When the first round has no cost samples it uses the initial placement.",
                 ("scheduler/search.py","CostSearchPolicy.place_rollouts","if max(after)")),
                ("Length bias in early completions",
                    "Each round starts with a snapshot of lengths from completed rounds. A sample truncated at "
                    "an earlier cap can inform a forecast at the current cap only if the earlier cap was at "
                    "least as high. Optional conditional quantiles assign different percentiles to active "
                    "requests with the same route, progress and cap. For the last request, the forecast can use "
                    "the EMA of per-round maxima until its prefix exceeds the EMA, then return to the "
                    "conditional estimate.",
                 ("scheduler/search.py","Forecast.estimate_quantiles","order statistics")),
                ("Using migration waits for short decode segments",
                    "The two endpoints may reach safe points at different times. merge_backfill uses binary "
                    "search to find the longest decode segment predicted to fit the gap. It adds the extra "
                    "tokens and their recomputation cost to the forecast and compares the result with waiting. "
                    "The controller can cancel a paired reservation and replan.",
                 ("scheduler/search.py","Forecast.merge_backfill","while lo < hi")),
                ("Route order, preloading and claiming work",
                    "The search compares serving by ready time, grouping by teacher and draining one route "
                    "first. Scoring segments follow offset order within each trajectory. prepare_teacher can "
                    "load a role before claiming any prefixes. During a cold score with online calibration "
                    "disabled, a warm replica can claim the prefixes first. With online calibration enabled, "
                    "dispatch claims the batch and queues it behind the switch. Later rounds retain this "
                    "behavior.",
                 ("scheduler/controller.py","Controller.assign","Acquire only the engine lease")),
                ("Sharing a ready queue across groups",
                    "With measured costs, the search also considers smaller scoring and training batches sized "
                    "for the available groups. Each dispatched training microbatch claims its samples "
                    "immediately. Other groups take what remains. The controller reserves capacity for "
                    "outstanding generation and scoring. When many layouts are feasible, it dispatches their "
                    "common decode work before enumerating actions on contested groups.",
                 ("scheduler/calibration.py","share_ready","share = min")),
                ("Releasing idle teachers early",
                    "When publication costs are broken down by phase, the search can keep a teacher resident "
                    "until publication or switch it to standby while other groups work, hiding process shutdown. "
                    "The online calibration path uses the measured duration of the publication barrier and does "
                    "not estimate the savings from an early release separately.",
                 ("scheduler/search.py","Forecast.project","with_standby =")),
            ]),
        section("roles", "Role changes", "GPU handoffs and loading heterogeneous teachers",
            "A switch holds an exclusive group lease. At a safe point, the worker confirms that student requests "
            "have finished or migrated, disposes of old caches, checks available memory and loads the next role. "
            "The controller updates the group's role when it receives the matching completion. It can plan the "
            "next batch while the engine remains exclusive to the current operation.",
            table(["Condition","Execution path"],[
                ("Compatible inference layouts",
                    "Reuse the engine. With bucketed frozen-teacher synchronization enabled, load that teacher's "
                    "weights through local CUDA IPC."),
                ("Heterogeneous teacher with dedicated ranks",
                    "Prepare the runtime on the CPU and start active ranks after the GPU handoff. Active "
                    "processes exit when the teacher releases the GPUs. Optional teacher images reduce repeated "
                    "initialization."),
                ("Training state occupies device memory",
                    "Keep the trainer resident if the new role fits. Otherwise move state to the host. "
                    "Outstanding gradients constrain when the trainer can release its memory.")]),
            "Switch time includes releasing the previous role, preparing the next role, waiting and loading. The "
            "group reports elapsed wall time across these stages, including synchronization waits. It accepts "
            "scoring or training batches once loading and memory checks finish.",
            [("workers/group.py","GroupWorker.switch","if self.infer.owned"),("workers/model.py","inference_signature","def inference_signature"),
             ("backends/teacher_pool.py","TeacherRanks.activate","def activate"),("backends/teacher_rank.py","prepared_rank","def prepared_rank")],
            notes=[
                ("Reusing frozen teacher TP shards",
                    "With teacher_rank_images enabled, the first Hugging Face load of a dense, unquantized Qwen2 "
                    "teacher can produce native SGLang rank shards. The image key includes the model source, TP, "
                    "device type and software versions. One replica holds the write lock and publishes the "
                    "temporary directory atomically once every shard is present. Other groups can continue "
                    "loading from Hugging Face. On first use, fixed-prefix log probabilities are checked against "
                    "Hugging Face at each position.",
                 ("backends/teacher_image.py","TeacherImage.build","LOCK_NB")),
                ("Preparing on the CPU before a GPU lease",
                    "A heterogeneous teacher can import its runtime ahead of time. The active scheduler starts "
                    "once it gets the GPU lease. On release, its ranks and helper processes exit. Preflight "
                    "measures per-GPU memory requirements for each route. Later switches use these measurements "
                    "and available memory to decide whether the trainer can stay resident. The trace separates "
                    "remaining preparation time, loading and shutdown in standby_wait_s, teacher_rank_load_s and "
                    "teacher_rank_release_s.",
                 ("backends/teacher_pool.py","TeacherRanks.release","def release")),
                ("Preserving buffers captured by CUDA graphs",
                    "Models with compatible layouts still need their own weights. The optional Qwen2 path "
                    "rebuilds supported unscaled RoPE cosine and sine caches in place, preserving the addresses "
                    "captured by CUDA graphs. The request_seed launcher can restore batch-invariant prefill "
                    "after capturing the decode graph, supporting both graph decode and cached scoring "
                    "continuations.",
                 ("backends/qwen2_rope.py","restore","cache.copy_")),
                ("Versioned host copies",
                    "With sufficient GPU memory the trainer can stay resident. When host mirroring is enabled, "
                    "the FSDP process acknowledges publication, then copies master parameters or Adam state to "
                    "the host and records the version. When unloading later, it reuses matching copies and "
                    "refreshes stale ones. This consumes host memory and transfer bandwidth. The next command on "
                    "the rank may need to wait for the copy.",
                 ("backends/fsdp2.py","Shard.sleep","host_moments_version")),
            ]),
        section("migration", "Migration and KV", "Migrating requests, rebuilding caches and widening TP",
            "An ordinary merge changes which group owns a student request. The source exports the sampling seed "
            "and absolute position. The controller supplies the prompt and committed tokens. The destination "
            "checks the version, ownership and remaining KV capacity. The source releases the request after the "
            "destination accepts it and retains ownership if the destination declines.",
            flow([("hold", "Reserve a safe point at each end"),("export / install", "Transfer sampling state and verify destination capacity"),
                  ("release", "Release the source after destination acknowledgment"),("decode", "Resume sampling on the destination")])
            + table(["State","Reuse and recomputation"],[
                ("Student KV",
                    "The destination reuses cached context. It can prefill missing tokens at takeover or on the "
                    "next decode. Deferral frees the source sooner and leaves the prefill work to the "
                    "destination."),
                ("Teacher KV",
                    "Each inference replica has a separate budget. When a replica is flushed, the controller "
                    "withdraws its cached offsets and retains scores already delivered."),
                ("Wide TP for a tail request",
                    "SGLang prebuilds a wide engine for a pair of groups on one node and puts it to sleep with "
                    "the matching weight version. On takeover it wakes, replays the prefix and validates the log "
                    "probability of the last committed token. The simulator can also model the cost of moving KV shards "
                    "directly.")])
            +
                '<p class="equation"><b>Estimated local benefit of wider TP</b>Narrow decode time − wide decode '
                'time − costs of entering and leaving the wide layout − delayed service − cost of rebuilding '
                'evicted teacher KV</p>',
            "Expansion requires one active request across the eligible same-node pair. Other groups can continue "
            "generating. The controller permits one wide pair at a time, and the worker rechecks physical memory "
            "before waking the engine. It returns the group when the long request finishes or earlier if scoring "
            "or training pressure rises. tp_eligibility records screening criteria including ownership, capacity "
            "and in-flight work.",
            [("backends/executor.py","GroupExecutor.execute","src = self.call"),("backends/sglang_engine.py","Inference.install","contexts ="),
             ("backends/elastic_rollout.py","WideRollout.execute","handoff_logp_error"),("scheduler/search.py","Forecast.layouts","handoff ="),("backends/sim_tp.py","TailTP.transition","def transition")],
            notes=[
                ("Checking admission on each destination replica",
                    "For an ordinary merge, each destination replica budgets for resident requests, incoming "
                    "context and the next decode quantum. Spare space on one replica cannot make up a shortfall "
                    "on another. After a rejection, the controller remembers the pair's active request sets and "
                    "avoids retries until the sets change.",
                 ("backends/inference_replicas.py","InferenceReplicas.migration_fits","def migration_fits")),
                ("Teacher KV eviction and scoring admission",
                    "SGLang flushes teacher caches by replica. To cover a shortfall it chooses the smallest "
                    "replica large enough, or starts with the largest if none suffices. Dedicated teacher ranks "
                    "round up to whole pages, reserve allocator headroom per request and split large batches "
                    "into windows that fit. Each cold window fits even if the engine has evicted the cached "
                    "prefix.",
                 ("backends/score_cache.py","trim_score_cache","enough =")),
                ("Building the wide engine and coordinating startup",
                    "Before training, the worker establishes same-node wide TP communication and engines. With "
                    "CUDA graphs enabled, it captures a batch-one graph for single-request decode. It measures "
                    "wake, sleep, recomputation and wide decode. FSDP processes can import CPU libraries "
                    "concurrently and start CUDA initialization after the wide engine sleeps. idle_device_bytes "
                    "records memory retained during sleep and wake_bytes records the memory required to wake.",
                 ("workers/node.py","NodeRuntime.prepare","gate =")),
                ("Hardware handoff and simulated resharding",
                    "The wide engine validates the seed, absolute position and version, replays the prefix and "
                    "compares the last token's log probability. The completion reports wake_s, prefill_s and "
                    "rebuilt_tokens separately. An alternative simulation model estimates direct transfers by logical "
                    "KV heads, charges communication group setup on the first loan and returns only shards added "
                    "since expansion on early contraction. Transfer byte counts describe different operations in "
                    "these two paths.",
                 ("backends/simulation.py","Simulator.resize_tp","tp_prefix_tokens")),
            ]),
        section("publication", "Gradients and weights", "Round-wide gradient normalization and weight publication",
            "Each FSDP2 group claims ready samples and runs forward and backward passes with θᵥ. Sampled K1 "
            "computes one loss per response position. Microbatches accumulate sums of loss gradients and counts "
            "of valid response tokens. At round end the optimizer group checks the global denominator.",
            '<p class="equation"><b>Round-wide gradient</b><code>gᵥ = (∑ₘ ∑ⱼ∈Bₘ ∇ℓⱼ(θᵥ)) / N</code><i>Bₘ '
            'contains the valid response positions in microbatch m, and N counts them over the whole round. Each '
            'microbatch contributes in proportion to its valid token count.</i></p>'
            + flow([("Reduce", "Aggregate gradients across corresponding shard ranks"),("commit", "Divide by N, clip and take one AdamW step"),
                    ("Publish weights", "Build or receive BF16 buckets on each node"),("rollout_barrier", "All generation groups acknowledge v+1")]),
            "To publish tensors across replicas, workers first synchronize master parameter shards. Each node "
            "can then build rollout buckets for trainable parameters locally. The authoritative replica "
            "distributes frozen parameters and buffers when needed. Buffers can be reused once all receivers "
            "acknowledge them. Sampling waits for the new-version barrier, and checkpoint persistence is "
            "recorded separately.",
            [("backends/fsdp2.py","Shard.commit","g.div_(n)"),("backends/replica_sync.py","reduce_gradients","def reduce_gradients"),
             ("backends/weight_sync.py","TensorPublisher.bucket","def bucket"),("scheduler/controller.py","Controller.run_round",'self.emit("rollout_barrier"')],
            notes=[
                ("Policy loss on sampled tokens",
                    "The default on_policy_k1 objective recomputes old log probabilities on the frozen "
                    "round-start actor without gradients, then runs a differentiable forward pass. At sampled "
                    "positions, the difference between teacher and current actor log probabilities forms the "
                    "detached advantage. PPO ratio clipping and dual clipping yield the loss. Early backward "
                    "passes use the same objective as a fully synchronous schedule.",
                 ("algorithms/token_losses.py","clipped_k1","advantage =")),
                ("Training time on the slowest rank",
                    "With variable-length FlashAttention, longer samples are assigned first to the rank with the "
                    "fewest accumulated tokens. Each sequence resets position IDs. cu_seqlens keeps attention "
                    "within sequence boundaries, and only positions predicting response tokens need vocabulary "
                    "logits. Empty ranks join the same collectives with zero loss and zero valid tokens. The "
                    "search uses the matching packed or padded layout for train_width.",
                 ("backends/fsdp2.py","Shard.response_logits","logits_to_keep")),
                ("Fixed communication groups with changing training membership",
                    "Other groups may finish every backward pass while the optimizer group is still decoding a "
                    "tail. Before reduction, workers use small tensors to check the version, denominator and "
                    "gradient availability. Cross-group reduction uses corresponding FSDP ranks and bounded FP32 "
                    "buckets. Inference groups join with zero contributions and can leave the model on the CPU.",
                 ("backends/replica_sync.py","reduce_gradients","Exchange the actual")),
                ("Publishing one bucket to narrow and wide engines",
                    "After replicas confirm the updated FP32 shards, each node can rebuild BF16 buckets "
                    "containing trainable parameters locally, reducing cross-node broadcasts. Narrow and wide "
                    "engines can load a bucket concurrently. Each consumer receives its own CUDA IPC handle to "
                    "the same source tensor and acknowledges it before the memory is reused. The final round "
                    "skips wide-engine publication. wide_publish_exposed_s measures how much longer wide "
                    "publication takes than narrow publication on the same group.",
                 ("workers/group.py","GroupWorker.load_buckets","parallel =")),
                ("Measuring commit, publication and recovery separately",
                    "In-memory publication happens each round. HF exports and DCP checkpoints with optimizer "
                    "state follow the checkpoint configuration. Resumable checkpoints also store the EMA of "
                    "round maximum response lengths. Model-only exports support evaluation. Resuming training "
                    "requires a version with training state. End-to-end job time, round time and steady-state "
                    "step time excluding persistence cover different intervals.",
                 ("backends/fsdp2.py","Shard.save_checkpoint","scheduler_state")),
            ]),
        section("interfaces", "Extension points", "Topology, objectives and reproducible comparisons",
            "Resource reservations, the training objective and scheduling are configured separately. To study a "
            "scheduling change, hold the data, routes, initial parameters, engine layout and objective fixed and "
            "vary one scheduling option.",
            table(["Interface", "Implementation and experimental controls"], [
                ("Nodes and GPU groups",
                    "Ray reserves the job's resources before the engines start and checks physical GPU "
                    "ownership. Each node runs a local worker, and heterogeneous teachers can load locally "
                    "staged weights. Ordinary migrations can cross nodes. Wide TP uses prebuilt pairs on one "
                    "node."),
                ("Distillation objective",
                    "DistillationAlgorithm declares the required supervision and whether to recompute old log "
                    "probabilities. Its route selects a frozen teacher, and token_loss returns a loss per "
                    "position. Sampled KL and teacher top-k objectives can share the version barrier, microbatch "
                    "protocol and round-wide denominator."),
                ("Scheduling ablations",
                    "Keep cost_search and the workload fixed. Disable one feature at a time with "
                    "policy_options.live_prefix_scoring=false, early_training=false, max_pairs=0 or "
                    "allow_tp_expansion=false. merge_backfill can be disabled independently. When comparing "
                    "expansion policies, keep the wide engine, memory reservation and publication overhead the "
                    "same."),
                ("Observability",
                    "dispatch records the action sent, and complete.start/end records the worker interval. A "
                    "two-group action has one completion record. Ray aligns GPU samples with the driver clock "
                    "and records clock uncertainty. The activity timeline and NVML utilization use separate data "
                    "sources.")]),
            "Disabling one action within the same scheduler measures its marginal effect on a round. When "
            "comparing DynaMOPD runs, check GPU count, TP width, replica count, actual generated and training "
            "token counts, and the boundaries of each timing metric.",
            [("cluster/ray.py","RayExecutor.__init__","Check shared paths"),("cluster/ray.py","RayExecutor.gpu_monitor","uncertainty"),
             ("algorithms/base.py","DistillationAlgorithm.loss_sum","terms ="),("scheduler/search.py","CostSearchPolicy.__init__","allow_wide")]),
    ]


def read_trace(path, requested):
    """Project the event stream onto timing/decision fields before rendering anything."""
    with path.open(encoding="utf-8") as stream:
        return scan_trace(stream, path.name, requested)


def scan_trace(stream, name, requested):
    current = fallback = backend = groups = None
    for line_no, line in enumerate(stream, 1):
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except ValueError as exc:
            raise ValueError(f"{name} line {line_no}: {exc}") from exc
        kind, version = e.get("event"), e.get("version")
        if kind == "config":
            backend = e.get("backend")
        if kind == "executor_ready":
            groups = len(e.get("placements", [])) or None
        if not isinstance(version, int) or (requested is not None and requested != version):
            continue
        if kind == "round_start":
            current = dict(version=version, start=e["t"], end=None, samples=e.get("samples"), spans=[],
                           searches=[], pending={}, receipts=defaultdict(list), handoffs=[], teacher_formats=set())
        r = current
        if r is None or version != r["version"]:
            continue
        if kind == "round_end":
            r["end"] = e["t"]
            if r["end"] <= r["start"]:
                raise ValueError("round_end must be later than round_start")
            result = (version, r, backend, groups)
            if requested is not None or r["searches"]:
                return result
            if fallback is None:
                fallback = result
            current = None
        elif kind == "schedule_search":
            candidates = [(x["layout"], x.get("pair"), x["finish_s"]) for x in e.get("layouts", [])
                          if isinstance(x.get("finish_s"), (int, float)) and math.isfinite(x["finish_s"])]
            if len(candidates) > 1:
                r["searches"].append(dict(id=e.get("decision_id"), time=e["t"], chosen=e["chosen_layout"],
                                          pair=e.get("chosen_pair"), candidates=candidates))
        elif kind == "dispatch":
            r["pending"][e["gpu"], e.get("peer"), e["op"]] = e.get("decision_id")
        elif kind == "complete":
            g, peer, op = e["gpu"], e.get("peer"), e["op"]
            decision = r["pending"].pop((g, peer, op), None)
            if op not in OPS:
                continue
            a, b = e["start"], e["end"]
            if not (math.isfinite(a) and math.isfinite(b) and a <= b):
                raise ValueError(f"invalid worker interval on line {line_no}")
            info = (g, peer, op, a, b, e.get("accepted"), e.get("new"), e.get("transfer"))
            r["spans"].append(info)
            if op == "switch" and e.get("teacher_load_format"):
                r["teacher_formats"].add(str(e["teacher_load_format"]))
            if op in ("move", "tp_expand", "tp_shrink"):
                phases = []
                for key, label in (("export", "export"), ("install", "install"), ("release", "release")):
                    start, end = e.get(key + "_start"), e.get(key + "_end")
                    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                        phases.append(f"{label} {end-start:.3f} s")
                for key, label in (("wake_s", "wake"), ("prefill_s", "prefix recomputation"), ("sleep_s", "sleep"),
                                   ("kv_transfer_s", "KV transfer")):
                    if isinstance(e.get(key), (int, float)):
                        phases.append(f"{label} {e[key]:.3f} s")
                width = (f"TP{e['from_tp']} → TP{e['to_tp']}"
                         if e.get("from_tp") is not None and e.get("to_tp") is not None else OPS[op][0])
                if e.get("accepted") is False:
                    width += " (rejected)"
                r["handoffs"].append((a, b, g, peer, width, ", ".join(phases) or "No phase breakdown"))
            if decision is not None:
                r["receipts"][decision].append(info)
    if fallback is None:
        raise ValueError("The selected trace or version has no matching round_start and round_end events")
    return fallback


def group_label(g, peer):
    return f"G{g}" if peer is None else f"G{g} + G{peer}"


def actual_figure(r, size):
    zero, end = r["start"], r["end"]
    duration = end-zero
    lanes = defaultdict(list)
    for g, peer, op, a, b, accepted, role, transfer in r["spans"]:
        if b <= zero or a >= end:
            continue
        label, style = OPS[op]
        if role:
            label += f" → {role}"
        if accepted is False:
            label += " (rejected)"
        desc = f"{label} · {a-zero:.3f}–{b-zero:.3f} s"
        left, right = max(0,a-zero)/duration*100, min(duration,b-zero)/duration*100
        bar = (f'<span title="{esc(desc)}" class="bar {style}" '
               f'style="left:{left:.5f}%;width:{max(0,right-left):.5f}%"></span>')
        lanes[g].append(bar)
        if peer is not None and peer != g:
            lanes[peer].append(bar)
    html_rows = []
    for g in sorted(set(lanes) | set(range(size or 0))):
        html_rows.append(f'<div class="lane"><b>G{g}</b><div class="track">{"".join(lanes[g])}</div></div>')
    keys = ''.join(f'<span><i class="{k}"></i>{esc(v)}</span>' for k,v in LABELS.items())
    return (f'<figure class="actual"><figcaption>Worker activity by GPU group <small>Round duration {duration:.2f} s</small></figcaption>'
            f'<div class="legend">{keys}</div><div class="lane"><b>Seconds</b><div class="axis">'
            f'<span>0</span><span>{duration/2:.1f}</span><span>{duration:.1f}</span></div></div>'
            f'{"".join(html_rows)}<p class="caption">Hover over a bar for start and end times. A two-group operation appears on both rows. Gray intervals have no recorded worker action. Bar lengths include worker and communication waits.</p></figure>')


def trace_section(trace):
    v, r, backend, groups = trace
    chart = actual_figure(r, groups)
    if r["handoffs"]:
        handoffs = [(f"{a-r['start']:.3f}", group_label(g, peer), width, f"{b-a:.3f}", phases)
                    for a, b, g, peer, width, phases in sorted(r["handoffs"], key=lambda x: x[0])]
        chart += ('<h3>Request migrations and TP handoffs in this round</h3>'
                  + table(["Start in round (s)", "Groups", "Action", "Handoff time (s)", "Reported phases"], handoffs)
                  + '<p class="caption">Each paired handoff counts once. Workers time selected phases individually. Total duration also includes RPC overhead and waiting.</p>')
    else:
        chart += '<p>No request migration or TP handoff completed in the recorded trace for this round.</p>'
    if r["teacher_formats"]:
        chart += '<p>Teacher switches in this round used ' + esc(", ".join(sorted(r["teacher_formats"]))) + '.</p>'
    case = None
    for decision in r["searches"]:
        linked = r["receipts"].get(decision["id"], [])
        if any(op in ("move", "tp_expand", "tp_shrink") and accepted is not False
               for _, _, op, _, _, accepted, _, _ in linked):
            case = decision
            break
    if case is None and r["searches"]:
        case = r["searches"][0]
    if case:
        rows = sorted(case["candidates"], key=lambda x: x[2])
        # Show the best instance per layout, plus the exact chosen pair if different.
        shown = []
        seen = set()
        for mode, pair, seconds in rows:
            chosen = mode == case["chosen"] and pair == case["pair"]
            if mode in seen and not chosen:
                continue
            seen.add(mode)
            shown.append((mode, ", ".join(f"G{n}" for n in pair) if pair else "—", f"{seconds:.3f}", "Selected" if chosen else ""))
        chart += (f'<h3>Decision at {case["time"]-r["start"]:.3f} s into the round</h3><p>The table shows the fastest predicted group combination for each layout and the combination actually selected. '
                  f'Remaining time runs from the decision to weight publication.</p>'
                  + table(["Candidate layout", "Groups", "Predicted time remaining (s)", "Decision"], shown))
        selected = next((t for mode, pair, t in rows if mode == case["chosen"] and pair == case["pair"]), None)
        natural = next((t for mode, _, t in rows if mode == "natural"), None)
        if case["chosen"] != "natural" and selected is not None and natural is not None:
            delta = natural - selected
            direction = "shorter" if delta >= 0 else "longer"
            chart += f'<p>At decision time, the scheduler estimated that the selected layout would take {abs(delta):.3f} s {direction} than natural draining. Workers report actual action start and end times separately.</p>'
        linked = r["receipts"].get(case["id"], [])
        if linked:
            actions = []
            for g, peer, op, a, b, accepted, role, transfer in linked:
                outcome = "Rejected" if accepted is False else "Completed"
                detail = role if op == "switch" else transfer
                actions.append((group_label(g,peer), OPS[op][0], f"{a-r['start']:.3f}–{b-r['start']:.3f}",
                                f"{b-a:.3f}", outcome + (f" · {detail}" if detail else "")))
            chart += '<h3>Completed actions dispatched by this decision</h3>'+table(["Group","Action","Start–end (s)","Duration (s)","Outcome"],actions)
        else:
            chart += '<p>This decision has no completion record that matches a dispatch by decision_id. The timeline shows all recorded actions in the round.</p>'
    else:
        chart += '<p>The trace contains no schedule_search event with multiple candidates for this round. The timeline shows the recorded worker intervals.</p>'
    clock = "the simulation clock" if backend in ("simulation", "simrl") else ("hardware worker times" if backend == "sglang_fsdp2" else "executor times")
    group_count = f" across {groups} GPU groups" if groups else ""
    return section("case-study", "Run example", "Predicted layouts and observed execution",
        f"Student version v{v} → v{v+1}{group_count}. Times use {clock} and are relative to round_start.",
        chart, "Decisions match dispatches by version and decision_id, then match completions by action on the exclusively leased groups. This example uses the first multi-candidate decision with a completed migration or TP handoff, or the first multi-candidate decision if none resulted in a handoff. The predicted difference describes the estimate available at decision time. Measuring the actual change in round duration requires runs with matched workloads.",
        [("scheduler/controller.py","Controller.settle",'self.emit("complete"'),("tools/audit.py","audit","def audit")])


STYLE = """
:root {
  --ink:#193136; --muted:#526767; --line:#d7e1de; --green:#006557; --wash:#f1f6f3;
  --gen:#26786b; --score:#496ea5; --train:#984461; --move:#a46629; --update:#6c529a;
  color:var(--ink); background:white;
  font:16px/1.72 "Source Sans 3",system-ui,sans-serif;
}
*{box-sizing:border-box}body{margin:0}a{color:var(--green);text-underline-offset:3px}
a:focus-visible,summary:focus-visible,button:focus-visible{outline:2px solid var(--green);outline-offset:2px}
header{border-bottom:1px solid var(--line)}header>div,.layout{width:min(1120px,calc(100% - 48px));margin:auto}
header>div{display:flex;align-items:center;justify-content:space-between;padding:12px 0}
header a{color:var(--ink);font-weight:760;letter-spacing:-.02em;text-decoration:none;font-size:18px}
header button{background:white;border:1px solid var(--line);border-radius:4px;color:var(--muted);padding:4px 10px;font:inherit;font-size:13px;cursor:pointer}
.layout{display:grid;grid-template-columns:155px minmax(0,1fr);gap:clamp(24px,5vw,64px);padding:27px 0 64px;align-items:start}
nav{position:sticky;top:16px}nav a{display:block;font-size:14px;padding:4px 9px;color:var(--muted);border-left:2px solid var(--line);text-decoration:none}
nav a:hover,nav a[aria-current=location]{border-color:var(--green);color:var(--green)}
main{min-width:0}.title{padding:0 0 18px;border-bottom:2px solid #385c54}.title p{color:var(--muted);font-size:15px;margin:5px 0 0}
h1,h2,h3{line-height:1.35;letter-spacing:-.03em;margin:0}h1{font-size:clamp(27px,3.2vw,39px)}
h2{font-size:clamp(21px,2.1vw,26px);margin-bottom:12px}h3{font-size:17px;margin:20px 0 6px}p{margin:10px 0}
section{padding:25px 0 22px;border-bottom:1px solid var(--line);scroll-margin-top:12px}
figure,.observation,.equation{margin:16px 0;padding:13px 16px;background:var(--wash);border-radius:4px}
.observation,.equation{border-left:3px solid #669689}.observation strong,.equation b{display:block;font-size:13px;color:var(--green)}
.observation p{margin:2px 0;font-size:15px}.equation code{display:block;margin:3px 0}
.equation i,.caption,small{font-style:normal;font-size:12px;font-weight:450;color:var(--muted)}.equation i{display:block}
figcaption{display:flex;align-items:baseline;justify-content:space-between;gap:8px;flex-wrap:wrap;font-size:15px;font-weight:650;margin-bottom:10px}
.lane{display:grid;grid-template-columns:50px 1fr;gap:9px;align-items:center;margin:6px 0}.lane b{font-size:13px}
.track,.axis{position:relative;height:25px}.track{background:#e1eae6}.track.plain{background:transparent}
.axis{display:flex;justify-content:space-between;height:auto;font-size:12px}.bar{position:absolute;top:0;height:100%;display:grid;place-items:center;color:white;white-space:nowrap;overflow:hidden;font-size:12px}
.decode{background:var(--gen)}.score{background:var(--score)}.train{background:var(--train)}.handoff{background:var(--move)}.commit{background:var(--update)}
.actual .track{height:15px}.actual .lane{margin:3px 0}.legend{display:flex;gap:10px;flex-wrap:wrap;font-size:12px}.legend i{display:inline-block;width:10px;height:8px;margin-right:4px}
.flow{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:19px;list-style:none;padding:0;margin:20px 0}
.flow li{border-top:2px solid #79a395;position:relative;padding:4px 0}.flow li:not(:last-child)::after{content:"→";position:absolute;right:-17px;top:3px;color:var(--green)}
.flow strong,.flow small{display:block}.flow strong{font-size:15px}
.scroll{overflow-x:auto}table{width:100%;border-collapse:collapse;margin:8px 0 12px;font-size:14px;line-height:1.55}
td,th{text-align:left;vertical-align:top;border-bottom:1px solid var(--line);padding:7px 9px}th{background:#e8f0eb}td:first-child{font-weight:600}
figure table td{background:#fff}code,pre,.filename{font-family:"IBM Plex Mono",monospace}p code{font-size:.94em}
details{margin-top:12px}summary{font-size:13px;cursor:pointer;color:var(--green);font-weight:600}
.design-notes{border-left:2px solid #bdd0c8;padding-left:12px}.design-notes dl{margin:10px 0}
.design-notes dt{font-size:15px;font-weight:650;margin:12px 0 2px}.design-notes dd{margin:0;font-size:15px}
.src{margin:10px 0}.src a,.filename{font-size:13px;overflow-wrap:anywhere}.src pre{overflow:auto;font-size:12px;line-height:1.55;background:#f4f6f4;padding:9px}
@media(max-width:740px){
  header>div,.layout{width:calc(100% - 28px)}.layout{display:block;padding:10px 0 30px}
  nav{position:static;display:flex;overflow:auto;margin-bottom:16px}nav a{white-space:nowrap}
  section{padding:20px 0}.flow{grid-template-columns:1fr 1fr;gap:9px 15px}.flow li::after{display:none}
  figure{padding:10px 8px}.lane{grid-template-columns:32px 1fr;gap:4px}.bar{font-size:10px}
  td,th{padding:5px;font-size:13px}.scroll table{min-width:350px}
}
@media print{
  header,nav,.sources{display:none}.layout{display:block;width:100%;padding:0}.title{padding-bottom:8px}
  section{padding:15px 0;border:0}h2,h3{break-after:avoid}figure,.flow,tr,.observation{break-inside:avoid}
  .scroll,.src pre{overflow:visible}.bar,.legend i{print-color-adjust:exact;-webkit-print-color-adjust:exact}
}
"""

JS = """
const links = [...document.querySelectorAll('nav a')];
const sections = links.map(link => document.getElementById(link.hash.slice(1)));
let scheduled = false;
function markSection() {
  let index = 0;
  sections.forEach((section, i) => { if (section.getBoundingClientRect().top <= 70) index = i; });
  if (innerHeight + scrollY >= document.documentElement.scrollHeight-2) index=sections.length-1;
  links.forEach((link, i) => {
    if (i === index) link.setAttribute('aria-current', 'location');
    else link.removeAttribute('aria-current');
  });
  scheduled = false;
}
function onScroll() {
  if (!scheduled) { scheduled = true; requestAnimationFrame(markSection); }
}
addEventListener('scroll', onScroll, {passive:true});
addEventListener('resize', onScroll);
markSection();
let closedNotes = [];
addEventListener('beforeprint', () => {
  closedNotes = [...document.querySelectorAll('.design-notes:not([open])')];
  closedNotes.forEach(item => item.open = true);
});
addEventListener('afterprint', () => closedNotes.forEach(item => item.open = false));
document.getElementById('print').onclick = () => window.print();
"""


def write_guide(output, trace=None, version=None):
    output = output.resolve()
    if trace and trace.resolve() == output:
        raise ValueError("The output path would overwrite the input trace")
    record = read_trace(trace, version) if trace else None
    sections = design_sections(record)
    if record:
        sections.append(trace_section(record))
    navigation, blocks = [], []
    for item in sections:
        anchor = esc(item["anchor"])
        navigation.append(f'<a href="#{anchor}">{esc(item["nav"])}</a>')
        notes = item["notes"]
        detail = ""
        if notes:
            rows = "".join(f"<dt>{esc(title)}</dt><dd>{esc(reason)}</dd>" for title, reason, _ in notes)
            detail = (f'<details class="design-notes"><summary>Implementation notes ({len(notes)}) · {esc(notes[0][0])}'
                      f'</summary><dl>{rows}</dl></details>')
        refs = ''.join(source(ref, output) for ref in [*item["refs"], *(ref for _, _, ref in notes)])
        blocks.append(f'<section id="{anchor}"><h2>{esc(item["title"])}</h2>'
                      f'<p>{esc(item["lead"])}</p>{item["body"]}<p>{esc(item["closing"])}</p>{detail}'
                      f'<details class="sources"><summary>Source code</summary>{refs}</details></section>')
    document = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<title>DynaMOPD · System design for synchronous multi-teacher distillation</title><style>{STYLE}</style></head><body>'
                f'<header><div><a href="#top">DynaMOPD</a><button id="print" type="button">Print guide</button></div></header>'
                f'<div class="layout"><nav aria-label="Contents">{"".join(navigation)}</nav><main>'
                f'<div class="title" id="top"><h1>Intra-round parallelism in synchronous multi-teacher distillation</h1>'
                f'<p>Built-in cost simulation, dynamic GPU group scheduling and execution with SGLang and FSDP2</p></div>'
                f'{"".join(blocks)}</main></div><script>{JS}</script></body></html>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"runs"/"system-design.html", help="path for the self-contained HTML guide")
    parser.add_argument("--trace", type=Path, help="optional events.jsonl trace or run directory containing it")
    parser.add_argument("--version", type=int, help="student version at round start (default: first complete round with a multi-candidate decision, or first complete round)")
    args = parser.parse_args()
    if args.version is not None and not args.trace:
        parser.error("--version requires --trace")
    trace = args.trace/"events.jsonl" if args.trace and args.trace.is_dir() else args.trace
    try:
        write_guide(args.out, trace, args.version)
    except (KeyError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
