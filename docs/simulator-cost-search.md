# Calibrating the simulator from worker traces

The bundled H20 recipes select `cost_search` when the CLI runs with `--simulate`. From Python, set the policy explicitly with `fit(dict(config, policy="cost_search"), simulate=True, ...)`. Training on hardware uses the `policy` in the job configuration. Use a calibration profile that matches the models, training layout, and engine lifecycle. CPU simulation and replay use DynaMOPD's built-in Python models and do not require an external simulator package.

The measurements and replay tables below describe earlier experiments, including the previous SimRL-backed simulator. They have not been rerun with the current built-in simulator. Its cost equations use the configured coefficients; rerun matched workloads and validate against hardware before attributing these numbers to the current implementation.

## Validate individual action predictions

Five SGLang/FSDP2 experiments ran on the same eight-GPU H20 node. In each scenario, the model was fitted to worker receipts from v0 and v1, then evaluated on held-out v2. Each prediction could use state reported by previously completed actions. The table shows mean absolute error in seconds per worker action. A simultaneous weight publication across four groups produces four receipts.

| Scenario | Role switch, direction baseline → fitted model | Weight publication, direction baseline → fitted model | Training microbatch, total-token baseline → fitted model |
|---|---:|---:|---:|
| control | 0.406 → 0.350 | 0.472 → 0.153 | 0.064 → 0.043 |
| cap16k | 0.315 → 0.091 | 1.096 → 0.677 | 0.215 → 0.069 |
| gsm8k | 0.780 → 0.439 | 0.946 → 0.630 | 0.045 → 0.027 |
| math-student | 0.287 → 0.302 | 0.728 → 0.240 | 0.106 → 0.122 |
| teacher14b | 10.441 → 1.350 | 39.215 → 2.234 | 0.159 → 0.068 |

The direction baseline uses the median unload and load times for each source/destination role pair in the fitting set. It also reproduces the previous publication estimate's extra synchronization charge. The training baseline uses the measured median time per token. The results directory records coverage counts and individual errors.

- **Role switches.** The model distinguishes training-state residency, the group that owns the optimizer, weight replacement in a shared engine, legacy engine restarts, and starting or stopping ranks for heterogeneous teachers. Restarting the legacy single engine invalidates its previous wake-up memory calibration. Sparse groups of observations fall back to a coarser grouping for the same mechanism. Simulated receipts record the group in `cost_profile_key` and its sample count in `cost_profile_samples`. The `cost_profile_fallback` field identifies estimates taken from configuration.
- **Weight publication.** Groups publishing tensors share a rendezvous. The local preparation estimate excludes `tensor_publish_s` because that measurement includes waiting. Once the slowest group is ready, the simulator charges a single shared transfer. Each earlier group's wait appears in `publish_wait_s`. The transfer estimate uses the shortest participating tensor call in that round. Per-device wake-up and bucket processing near the end remain approximate.
- **Training.** The model calculates the input width on the slowest rank using the actual FSDP2 assignments. It accounts for length-balanced packed batches and per-rank padding separately. Historical training receipts determine the fixed microbatch cost and marginal cost per token.
- **Migration.** The model accounts for each destination inference replica's context, KV headroom for the next decode, and block capacity. Ordinary migration transfers request state and repeats prefill. The wide-TP KV resharding model charges for redistributing KV heads. Releasing teacher KV preserves supervision already delivered. Subsequent chunks pay to recompute context when the KV is missing.
- **Migration timing.** Each replay receipt records the source of its fixed `move_base_s` cost in `move_cost_source`. The simulator first tries fitted costs for accepted or rejected moves, then a scalar migration cost. With no measured migration time, it estimates two control calls at twice the round's median ordinary Ray RPC overhead (`rpc_overhead_proxy`). Without valid RPC timings, it uses twice the scoring startup time (`score_launch_proxy`). Fitting to just one scoring length can yield a zero intercept. The simulator then prefers the positive duration of a short scoring batch in the calibration round (`short_score_batch_proxy`). Older profiles can use a positive per-token scoring time (`single_score_token_proxy`). The simulator raises an error if it has no positive timing to use. These estimates may understate cache flushing after the source drains. Rebuilding KV at the destination is timed separately. Replays using proxy timings support scheduling sensitivity analysis. Measuring migration gains requires hardware runs.

Across four rounds in each of the five scenarios, only 10 of 71 switches into training reclaimed recorded teacher prefixes. The largest total context reclaimed in one switch was 15,191 tokens. The count uses trajectory IDs evicted by the worker and their previously reported prefix lengths. There are too few observations to isolate the cost of releasing a nearly full KV pool from the cost of moving weights or training state. The model therefore charges no additional switch cost per GB. The table's 14B scenario used the legacy engine restart path. The teacher activation section below covers the process-based path and its calibration.

## Scheduling decisions

After settling a completion event, the controller predicts the time remaining in the round using visible prefixes, current ownership, historical output lengths, and in-flight work. The replay engine uses recorded response lengths to advance generation. The scheduler sees only tokens that have already been reported.

Hardware and simulated runs both maintain an exponential moving average (EMA) of the maximum response length per round. Each successful round contributes its longest generated response, including responses that reached EOS or the length limit. The first observation initializes the EMA, which updates once per round thereafter. The JSON option `max_length_ema_alpha` weights the new observation and defaults to 0.2. Set it to `null` to reproduce the earlier estimator. The trace records the estimate available at the start in `round_start.max_length_ema`, and the observed maximum and updated estimate in `round_end.max_response_tokens` and `round_end.max_length_ema`.

In round 0, the default prior for the longest response in the batch is the response limit. Among active requests that can still reach the highest limit, the scheduler assigns this estimate to the one with the longest generated prefix. Other requests use the ordinary estimator. If a completed response has already reached the highest limit among all remaining requests, the scheduler predicts no additional long tail. Search events record the trajectory carrying the prior in `cold_start_tail`. Actual sampling ends at EOS or at the request's limit. Setting `max_length_ema_alpha=null` disables both the first-round prior and later EMA updates.

In later rounds, once only one active request remains, the scheduler estimates its remaining tokens by subtracting its generated length from the EMA and respecting its own response limit. When several requests remain, no historical EMA exists, or the prefix has reached the EMA, the scheduler uses a conditional length estimate for that route. Without suitable observations it uses `tail_prior`. A request that started decoding late can finish last even with a short response, so the round maximum can overestimate its length. Resumable training checkpoints store `scheduler.json` alongside the weights.

The search considers natural draining, directed migration between groups, different sets of scoring groups, batches ordered by readiness or by teacher, scoring a live prefix or waiting for more tokens, training a ready microbatch on an idle group, and releasing a leased teacher to standby early. A wide-TP profile also makes expansion, continued wide decoding, and shrinking available. The current wide-TP executor runs the last active request on a pair of groups on the same node. For each candidate, the planner estimates group drain times, role transitions, scoring and training dependencies, gradient reduction, and weight publication. It dispatches the actions in the selected plan that can start immediately. Independent groups can run concurrently.

`policy_options.max_pairs` limits how many directed migration candidates enter each comparison. The default is 12, ordered by estimated movement cost. Supply `score_group_sets` for additional combinations of scoring groups or `score_batch_sizes` and `train_batch_sizes` for additional batch sizes. Configured batch and memory limits still apply. The scheduler has no mandatory merge threshold based on the remaining request count. Inspect `layouts[].finish_s`, `chosen_layout`, `defer_live_prefix`, and the chosen service groups in `schedule_search` events, then check adjacent `dispatch` and `complete` events for actual execution. Predicted time to round completion ranks candidates and approximates future arrivals and availability. CPU search time is excluded from virtual GPU time.

## CPU replay on matched workloads

Both the natural-drain policy and cost search replayed four rounds of 512 recorded samples per round. Within a scenario, the policies generated and trained on identical token counts in each round. The table reports the mean virtual duration of the two middle rounds, v1 and v2, including gradient reduction and weight publication.

| Scenario | Natural draining | Cost search | Natural draining / cost search |
|---|---:|---:|---:|
| control | 94.53 s | 99.92 s | 0.95× |
| teacher14b, legacy engine restart | 205.81 s | 157.71 s | 1.31× |

In the 7B comparison, cost search completed one migration in v1 and two in v2, yet its rounds took longer than natural draining. Compare predictions and receipts by decision ID to inspect errors in estimated tail length and downstream service time. This comparison measures two complete policies, so migration accounts for only an unknown portion of the difference. Round v1 was also used to fit the model. The policy comparison therefore has different coverage from the held-out v2 prediction errors above.

Run the following commands from the repository root after installing DynaMOPD. Replay needs only the base package; install the `calibration` extra to fit decode coefficients. Use new output paths.

```bash
trace="${TRACE_DIR:?set TRACE_DIR to the recorded training output}"
python -m DynaMOPD.tools.worker_costs "$trace" \
  --fit 0 1 --holdout 2 --config-out runs/teacher14b-worker-profile.json
python -m DynaMOPD.run --simulate --rounds 4 \
  --config runs/teacher14b-worker-profile.json --replay "$trace" \
  --policy cost_search --out runs/teacher14b-cost-search
python -m DynaMOPD.tools.audit runs/teacher14b-cost-search
```

## Activating heterogeneous teachers on a GPU group

`preload_teacher_imports` allows each physical rank to import SGLang on the CPU ahead of activation. The preparation process reads device properties supplied by the local FSDP rank and waits for controller authorization at the first CUDA operation or scheduler construction. When it starts waiting, it queries the CUDA driver for the current process's primary context count. Once authorized, it builds the teacher scheduler, loads that teacher's weights, and allocates KV. On exit, a supervisor reaps the GPU child and helper processes, then uses NVML on each GPU to verify that host PIDs that appeared during activation have exited. CPU preparation for the next activation overlaps subsequent work on the group.

For hardware role changes, `unload_s` includes releasing the previous engine after it drains and moving any training state as needed. `teacher_rank_load_s` includes both the remaining CPU preparation wait (`standby_wait_s`) and the GPU engine load. `teacher_rank_timings` breaks out per-rank CPU imports, imports after activation, and initialization. `teacher_kv_capacity` reports the actual capacity allocated by the engine. The worker uses the memory requirement measured at startup to decide whether to keep training state resident. If retaining it reduces KV capacity below the calibrated value, the worker releases the teacher, moves training state off the device, and rebuilds the teacher before accepting scoring requests.

When heterogeneous teachers share devices with automatically managed training state, FSDP copies optimizer state and FP32 parameter shards to the host after rollout weight publication is acknowledged. A matching-version CPU copy can be reused on the next offload. `reused_host_optimizer_bytes` and `reused_host_parameter_bytes` record the bytes actually reused. These optimizations require extra host memory and can be controlled separately with `mirror_optimizer_on_rollout` and `mirror_parameters_on_rollout`. Initial allocation, transfers, and subsequent activation all count toward job wall time.

The calibration tool separates preparation wait from GPU load time in traces that provide both. When a teacher releases its group, the simulator starts the next CPU preparation on that group, so the next activation waits only for outstanding preparation. The planner evaluates candidate schedules using the group's measured teacher activation time and estimates offload time from the previous role, training-state residency, and optimizer location. [Teacher executor](../backends/teacher_pool.py) · [Simulator](../backends/simulation.py) · [Calibration tool](../tools/worker_costs.py)

### Validate prediction errors by operation

Validate a worker cost model against an independent training run with the same models, parallel widths, and teacher loading mechanism. The output includes the count of each operation, the number covered by the fit, its measured mean duration, and mean absolute error. Times are in seconds. Install the `calibration` extra to fit decode coefficients.

```bash
python -m DynaMOPD.tools.worker_costs "$FIT_TRACE" --fit 0 1 \
  --validation-trace "$OTHER_TRACE" --holdout 0 1 2 3
```

Use `--extra-fit-trace "$LARGE_BATCH_TRACE"` to include other training shapes from the same engine layout. The tool checks round completeness separately for each run and aggregates cross-group publication receipts within the run that produced them. A validation run cannot also supply extra fitting data. Decode fitting charges for each replica's full RPC. The sum of the longest generation step counts in each KV window estimates additional serialization under capacity pressure. Existing five-coefficient `decode` profiles remain supported. New six-coefficient fits are stored in `decode_rpc`.

To validate an execution model independently, copy `sim_worker_profile` from another fit into the configuration's `sim_execution_profile` and keep the original scheduler and its priors. Set `sim_preflight_activation_s` to measured initial activation times by group index and teacher name. This information is available before the first decode. The virtual worker subsequently refreshes its report from completed role loads on its own group.

Use `python -m DynaMOPD.tools.trace_timing "$OTHER_TRACE" --config "$FITTED_CONFIG"` to hold observed actions and receipt dependencies fixed and recompute decode, scoring, and training times with either measured or estimated KV hits. Other actions, controller overhead, and transfer latency use the analyzed run's observations. The tool first reconstructs the whole round on its original clock and checks the result. For each round and compute operation, `compute_coverage` counts predictions and fallbacks to measured times. If the fixed-action error is small while freely scheduled replay diverges, compare group completion order and the actions selected after each completion.

Replay uses deterministic service times by default. The fit also stores empirical quantiles of the ratio of observed to predicted time for decode, scoring by teacher, and training. With `sim_timing_seed` set, the simulator uses independent reproducible random streams per group and operation to sample service-time multipliers. Receipts record `nominal_service_s` and `service_ratio`. Replay the same trajectories with several timing seeds to see whether variations in completion order change the chosen actions. Timing multipliers leave trajectory seeds unchanged. The fit does not currently capture correlated jitter across groups on the same node.

Within a decode RPC, the slowest inference replica determines the group's completion time. The model uses generation steps, total output from active requests, cumulative context, and input tokens that missed the cache. If every training observation used the same number of steps, the fit folds fixed overhead into the per-step time to avoid charging a full chunk to a short tail. Under memory pressure, the simulator splits requests according to each replica's KV pages, reclaims inactive prefixes, and charges for prefill if it encounters the trajectory again. Inspect `replica_decode_windows`, `rollout_kv_evicted`, and `cached_tokens` in the trace. Page-based batching approximates SGLang's continuous batching.

Scoring cost is fitted separately for each teacher, using the number of uncached prefix tokens on the slowest replica. A route with no offline coefficients can learn from completed scoring calls in the current job. While loading a teacher, a slow group holds only a role lease. It claims prefixes when scoring is dispatched. Simulation and hardware follow the same dispatch protocol. With phase barriers disabled, search can compare waiting for prefixes before loading with loading early during an idle interval. The completion receipt determines when the new role is available.

The eight-H20 TP1 recipe enables `length_forecast=conditional_quantiles` and includes deferred KV reconstruction as a candidate action. Each round draws a per-route length distribution from completed historical rounds. Conditional quantiles of observations that exceeded the current prefix length estimate the order in which requests on the same replica will finish. For a single tail request, the scheduler uses the configured EMA of maximum response length. The process rebuilds its historical distribution after a restart. Omitting the field selects per-request `conditional_median`. Both estimators supply timing predictions. The planner selects merges and expansions by the predicted round completion times of candidate layouts. [Eight-GPU configuration](../configs/sglang-tp1-fsdp2-8xh20.json)

`schedule_search.planner_wall_s` records actual CPU seconds spent planning. Offline simulation uses a virtual clock for GPU time. When comparing predicted and measured remaining time, or estimating throughput from virtual replay, also check wall time spent on the controller, communication, and file export.

The earliest availability of a future prefix comes from the next receipt of the current decode batch. The search compares scoring segments immediately with waiting for termination to build a batch, preserving each trajectory's offset dependencies in the predicted queue. For consecutive scores on the same teacher replica, it deducts prefill time only when the prefix still fits within predicted KV capacity. Estimates of migration memory and tokens to rebuild use the projected progress of both in-flight decode batches at their safe points.

Set `deferred_migration_prefill=true` in an SGLang job to include a merge candidate that defers reconstruction. The destination checks capacity, seed, and sampling position and acknowledges ownership. The source can then change roles. On the next generate call, the destination rebuilds missing KV from the full prefix. Immediate reconstruction and natural draining remain candidates in the same decision. In the hardware trace, `transfer=seed_position_then_decode` identifies this handoff, and `deferred_context_tokens` records context volume at migration. The subsequent decode's `migration_prefill_contexts` and `cached_tokens` give per-request context and engine cache hits. Decode RPC time includes reconstruction and generation. The trace has no separate GPU kernel timings for the two.

In the 14B experiment, fitting on the first two rounds and holding out the third gave a mean absolute error of 0.54 seconds across 12 role changes. The role-pair estimate gave 0.91 seconds. Error for the four groups' publication actions was 0.40 seconds. The two model fits ran separately from the training job.

### Inspect decode gaps and TP expansion

Add `--rollout-tp-profile <profile>` to a `--policy cost_search --simulate --replay` command, or set `rollout_tp_expansion` in the configuration, to make a wide-TP layout available. Match the profile to the student model, devices, and engine. To compare expansion on and off, use the same profile, reserved memory, and replay inputs in both runs. Set `policy_options.allow_tp_expansion=false` in the control run to disable only the expansion candidate.

For each wide-TP mesh, the planner checks request ownership, memory, and in-flight operations on the primary and lending groups. Other groups can continue generating. In `schedule_search` events, `tp_eligibility` reports candidate eligibility and `early_decode_groups` lists independent groups dispatched during the search. Use the `tp_expand` dispatch and completion to identify the actual handoff interval. When investigating decode gaps, inspect both idle intervals across groups and the per-replica work inside each RPC. Replicas can have uneven workloads even while their group generates continuously.
