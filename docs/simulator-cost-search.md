# 按真实 worker 回执校准模拟器

DynaMOPD 的 H20 配置在 CLI 中使用 `--simulate` 时默认选择 `cost_search`。Python 入口可显式传入 `fit(dict(config, policy="cost_search"), simulate=True, ...)`。真机训练沿用配置中的 `policy`。校准文件与模型、训练布局、引擎生命周期配套使用。

## 单动作预测与真机核对

使用 同一台八卡 H20 上的五组 SGLang/FSDP2 实验，各自在 v0、v1 的回执上拟合，在同一场景留出的 v2 预测。每次预测可以使用该动作之前已回执的状态。下表是按 worker 动作统计的平均绝对误差，单位为秒。四组卡同时发布会产生四条回执。

| 场景 | 服务换岗，方向基线 → 新模型 | 权重发布，方向基线 → 新模型 | 训练微批，总 token 基线 → 新模型 |
|---|---:|---:|---:|
| control | 0.406 → 0.350 | 0.472 → 0.153 | 0.064 → 0.043 |
| cap16k | 0.315 → 0.091 | 1.096 → 0.677 | 0.215 → 0.069 |
| gsm8k | 0.780 → 0.439 | 0.946 → 0.630 | 0.045 → 0.027 |
| math-student | 0.287 → 0.302 | 0.728 → 0.240 | 0.106 → 0.122 |
| teacher14b | 10.441 → 1.350 | 39.215 → 2.234 | 0.159 → 0.068 |

方向基线使用同一拟合集的角色对卸载、加载中位数，并复现原先发布额外叠加的同步时间。训练基线使用实测每 token 中位耗时。分项覆盖数和原始误差保存在结果目录。

- 换岗模型区分训练状态驻留、优化器所在组、共享引擎换权重、旧引擎重启和异构教师 rank 启停。旧单引擎重启会使原唤醒显存标定失效。样本少的组合退回同类机制的较粗分组。模拟回执中的 `cost_profile_key` 和 `cost_profile_samples` 指出所用分组，`cost_profile_fallback` 表示使用配置估计。
- 多组 tensor 发布使用同一个 rendezvous。本地准备时间剔除回执中包含等待的 `tensor_publish_s`。模拟器在最慢组就绪后计一次共享传输，`publish_wait_s` 记录较早就绪的组等了多久。共享传输时间由同一轮耗时最短参与者的 tensor 调用估计，尾部逐卡唤醒与分桶细节仍是近似。
- 训练按实际 FSDP2 分片计算最慢 rank 的输入宽度，分别处理长度均衡的 packed batch 和逐 rank 的 padding。微批固定成本与 token 边际成本从历史训练回执拟合。
- 迁移按目标推理副本核算上下文、下一次 decode 的 KV 余量和块容量。普通迁移搬运请求状态并重新 prefill，宽 TP 转换按 KV 头重分片计费。释放 teacher KV 后，已交付监督仍有效，后续 chunk 在缺少 KV 时支付上下文重算成本。

五个场景四轮中，71 次切入训练的动作里只有 10 次回收了已记录的 teacher 前缀，最大合计上下文为 15,191 token。这个计数来自 worker 驱逐的轨迹 ID 和此前回执的前缀长度。现有样本不足以区分 KV 池接近满载时的释放耗时与权重或训练状态搬运，因此当前没有按每 GB 额外收取一项切换费。表中的 14B 数据采用旧引擎重启流程。异构教师按进程接卡的实测回执和校准数据见下文。

## 调度动作

每次结算完成事件后，控制器用可见前缀、当前归属、历史输出长度和在途工作预测剩余轮时。回放器使用保存的真实长度推进生成。调度器只看到已经公布的 token。

真机和 SimRL 共用轮最大回复长度的指数移动平均（EMA）。每个成功轮次取所有请求实际生成的最大 token 数，包括 EOS 和达到上限的回复。首轮最大值初始化 EMA，之后每轮更新一次。作业 JSON 中的 `max_length_ema_alpha` 是新观测的权重，默认 0.2，可设为 `null` 复现原估计策略。每轮开始的 `round_start.max_length_ema` 是本轮可用的估计，结束的 `round_end.max_response_tokens` 和 `round_end.max_length_ema` 是实际最大值与更新结果。

剩最后一条活跃请求时，调度器用 EMA 减去已生成长度估计剩余 token，并按该请求自己的回答上限截断。多请求并行、首次冷启动或前缀已经达到 EMA 时，沿用同路由已结束样本的条件剩余长度中位数，没有合适样本时使用 `tail_prior`。最后完成的请求有可能较晚才开始解码，这时轮最大值可能高估它的长度。可恢复的训练检查点随权重保存 `scheduler.json`；恢复旧检查点时从冷启动开始。

搜索空间包含自然排空、有方向的双组迁移、不同评分组集合、按就绪时间或按教师安排批次、提前评分或等待后续前缀、在空闲组执行就绪微批，以及租用教师提前退回 standby。提供宽 TP profile 时，也比较借卡扩容、继续宽 TP 和缩回。宽 TP 执行器目前只支持同节点一对卡组承载最后一条活跃请求。每个候选都估算组排空时间、角色代价、评分和训练依赖、梯度归约和权重发布，再派发所选计划当前可执行的动作。多个互不冲突的组可以同时执行。

`policy_options.max_pairs` 限定每次比较的有向迁移候选数，默认 12，按估计搬运成本排序。`score_group_sets` 可补充评分组组合，`score_batch_sizes` 和 `train_batch_sizes` 可加入预设批量。配置里的 batch 和显存上限仍生效。没有按剩余请求数强制合并的触发条件。可以在 `schedule_search` 事件中检查 `layouts[].finish_s`、`chosen_layout`、`defer_live_prefix` 和所选服务组，再用相邻的 `dispatch`、`complete` 检查动作是否执行。预测的剩余轮时用于排序，包含对未来到达和可用性的近似；CPU 搜索时间不计入虚拟 GPU 时间。

## 同负载 CPU 回放

分别以保留的自然排空策略和新策略回放四轮，每轮 512 条真实样本。两种策略在同一场景下的每轮生成与训练 token 一致。表中取中间两轮 v1、v2 的虚拟耗时均值，含梯度归约和发布。

| 场景 | 自然排空 | 成本搜索 | 自然排空时间 / 搜索时间 |
|---|---:|---:|---:|
| control | 94.53 s | 99.92 s | 0.95× |
| teacher14b，旧引擎重启流程 | 205.81 s | 157.71 s | 1.31× |

7B 对照里的搜索在 v1、v2 分别完成了 1、2 次迁移，轮时仍比自然排空长。控制器对尾长和后续服务的预测存在误差，可按决策 ID 对照预测与回执。这里比较的是两个完整调度策略，不能把全部差值归因给迁移。v1 同时属于校准集，这张策略比较表与上面的 v2 留出误差使用不同口径。

在已能导入 SimRL 的环境中，从仓库根目录运行，输出使用新路径。

```bash
trace="${TRACE_DIR:?set TRACE_DIR to the recorded training output}"
python -m DynaMOPD.tools.worker_costs "$trace" \
  --fit 0 1 --holdout 2 --config-out runs/teacher14b-worker-profile.json
python -m DynaMOPD.run --simulate --rounds 4 \
  --config runs/teacher14b-worker-profile.json --replay "$trace" \
  --policy cost_search --out runs/teacher14b-cost-search
python -m DynaMOPD.tools.audit runs/teacher14b-cost-search
```


## 异构教师重新接卡

`preload_teacher_imports` 允许每个物理 rank 在 CPU 上提前导入 SGLang。准备进程读取本地 FSDP rank 提供的设备属性，到首次 CUDA 操作或调度器构造处等待控制器授权。开始等待时通过 CUDA driver 查询本进程的 primary context 数。接卡后才建立教师调度器、装载该教师自己的权重并分配 KV。退出时监督进程回收 GPU 子进程与辅助进程，逐卡用 NVML 检查激活时新出现的宿主机 PID 已经消失。下一次 CPU 准备与该卡组接下来的工作并行。

真实换岗的 `unload_s` 包含原角色排空后释放引擎及必要的训练状态搬运。`teacher_rank_load_s` 包含等待 CPU 准备的剩余时间 `standby_wait_s` 及加载 GPU 引擎的时间。`teacher_rank_timings` 提供每个 rank 的 CPU 导入、接卡后导入和初始化计时。`teacher_kv_capacity` 是引擎实际建立的 KV 容量。worker 用启动时测得的显存需求判断能否保留驻留的训练状态。保留训练状态后若引擎 KV 容量低于标定值，先释放教师、迁出训练状态，再重建教师并接受评分。

异构教师配合自动训练状态驻留时，FSDP 在 rollout 权重发布回执后将优化器状态和 FP32 参数分片复制到主机。版本相同的 CPU 副本可直接用于后续迁出，`reused_host_optimizer_bytes` 和 `reused_host_parameter_bytes` 分别统计实际复用量。这两项优化占用额外的主机内存，可通过 `mirror_optimizer_on_rollout` 和 `mirror_parameters_on_rollout` 分别控制。首次分配 CPU 副本、实际传输和下一次激活都计入作业墙钟。

校准工具从新 trace 分离准备等待与 GPU 加载时间。模拟器在教师释放时启动同卡组的下一次 CPU 准备，后续激活只等待剩余时间。搜索在备选调度序列中使用该组的实际教师激活耗时，并按原角色、训练状态驻留及优化器位置估计迁出时间。[执行器](../backends/teacher_pool.py) · [模拟器](../backends/simulation.py) · [校准工具](../tools/worker_costs.py)

### 校验每种操作的预测误差

可以用独立训练运行校验同一份 worker 成本模型。两个 trace 需要有相同的模型、并行宽度和教师加载机制。输出给出每种操作的次数、拟合覆盖数、实测均值与平均绝对误差，单位均为秒。拟合 decode 系数需要安装 `calibration` 依赖。

```bash
python -m DynaMOPD.tools.worker_costs "$FIT_TRACE" --fit 0 1 \
  --validation-trace "$OTHER_TRACE" --holdout 0 1 2 3
```

用 `--extra-fit-trace "$LARGE_BATCH_TRACE"` 加入同一引擎布局的其他训练形状。工具逐运行检查轮次完整性，并把跨卡发布回执按原始运行分别聚合。同一个验证运行不能作为额外训练数据。decode 拟合对每个副本的完整 RPC 计费，另用各 KV 窗口最长生成步数之和估计容量压力导致的额外串行时间。配置中已有的五项 `decode` 系数仍可使用，新拟合的六项系数保存在 `decode_rpc`。

独立校验执行模型时，可以将另一份拟合结果中的 `sim_worker_profile` 对象写入配置的 `sim_execution_profile`，保留原来的调度器与先验。`sim_preflight_activation_s` 可按卡组编号和教师名称提供实测初始激活时间，该输入在首个 decode 前可见。虚拟 worker 后续用本组已经完成的角色加载刷新回报。

还可以用 `python -m DynaMOPD.tools.trace_timing "$OTHER_TRACE" --config "$FITTED_CONFIG"` 固定实际动作与回执依赖，分别以真实 KV 命中量和估算命中量重算 decode、评分和训练算时。其余动作、控制器开销和传输时延采用被分析运行的观测值。工具先用原始耗时重建并校验整轮时钟。`compute_coverage` 逐轮给出各类计算操作中实际使用了模型预测和回退到实测值的次数。若固定动作的误差小而自由调度回放的误差大，应对照卡组完成顺序和之后的动作选择。

缺省回放使用确定性的服务时间。拟合结果还保存 decode、各教师 score 和 train 的实测耗时与预测耗时之比的经验分位数。显式设置 `sim_timing_seed` 后，模拟器按卡组和操作使用独立的可复现随机流，为服务时间采样倍率，回执记录 `nominal_service_s` 和 `service_ratio`。可在同一批轨迹上换多个 timing seed，检查先后完成顺序的波动是否改变选中动作。采样倍率不影响轨迹 seed，当前未拟合同节点不同卡组之间的相关抖动。

decode 以一个 RPC 内最慢的推理副本决定卡组完成时间。模型使用生成步数、活跃请求实际输出总量、累积上下文和未命中的输入 token。训练数据全部为固定步数时，固定开销归入每步耗时，以免将整个 chunk 的开销外推给短尾。显存压力下模拟器按各副本 KV 页数拆分请求，回收非活动前缀，在下次读到相同轨迹时计入重新 prefill。trace 中的 `replica_decode_windows`、`rollout_kv_evicted` 和 `cached_tokens` 可用于核对。页级批次划分是对 SGLang 连续批处理的近似。

评分成本按教师独立拟合，输入为最慢副本未命中的前缀 token 数。缺少离线系数的路由可以从当次作业已完成的评分学习。慢卡组在加载教师时只持有角色租约，前缀在真正派发评分时才被认领。模拟器和真机执行相同的派发协议。在不使用阶段屏障时，搜索也可比较等前缀到达再加载和利用卡组空档提前加载。完成回执决定新角色何时可用。

八张 H20 的 TP1 配置启用 `length_forecast=conditional_quantiles`，并把延后重建 KV 加入可选动作。每轮从已完整结束的历史轮次提取各路由的长度分布，对超过当前前缀长度的历史观测取条件分位数，估算同一副本内请求结束的先后。单条尾部请求沿用配置的最大长度 EMA。历史分布会随进程重启重新积累。配置省略该字段时沿用逐条条件中位数 `conditional_median`。两者给搜索提供时间估计，合并和扩容仍由备选布局的预计完轮时间决定。[八卡配置](../configs/sglang-tp1-fsdp2-8xh20.json)

`schedule_search.planner_wall_s` 记录规划器消耗的实际 CPU 秒。离线模拟的 GPU 时间使用虚拟时钟。在比较预计剩余时间与实测余时，或将虚拟回放用于吞吐估计时，需要另外核对控制器、通信和文件导出的墙钟开销。

未来前缀的首个可用时间取自当前 decode 批次的下一次回执。搜索比较立即分段评分和等到终止再组批，并在预测队列中维持每条轨迹的 offset 依赖。预计在同一教师副本接续评分时，只有该前缀仍在预计的 KV 容量内才抵扣 prefill 时间。估算迁移显存和待重建 token 时，使用两端在途 decode 批次到达安全点后的进度。

对 SGLang 作业设置 `deferred_migration_prefill=true` 会加入一种合并候选。目标组检查容量、seed 和采样位置，然后确认接管；源组收到确认后可以切换角色。目标组在下一次 generate 时用完整前缀重建缺失的 KV。立即重建和自然排空仍参与同一决策。真机 trace 中 `transfer=seed_position_then_decode` 表示此交接，`deferred_context_tokens` 为迁移时的上下文量，后续 decode 的 `migration_prefill_contexts` 和 `cached_tokens` 给出各请求的上下文与引擎命中量。decode RPC 的持续时间同时包含重建和生成，二者目前没有单独的 GPU kernel 计时。

在 14B 实验中，用前两轮拟合、第三轮留出验证，12 次服务换岗的平均绝对误差为 0.54 秒，按角色对估计为 0.91 秒。四组发布动作的误差为 0.40 秒。两次模型拟合与训练作业分开运行。
