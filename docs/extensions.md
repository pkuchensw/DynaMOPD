# 扩展 DynaMOPD

## 组装一个作业

在已有相同版本 CUDA、PyTorch 和 SGLang 的环境里，可用 `python -m pip install --no-deps --no-build-isolation -e .` 安装本地包。新环境的 `gpu`、`cluster`、`parquet` 可选依赖在 pyproject 中声明。SimRL 重放另需提供含 `srl` 的 SimRL 环境。直接在共享工作区运行也可使用原有 `launch.sh`。[包定义](../pyproject.toml)

```python
import json
from pathlib import Path
from DynaMOPD.trainer import fit

cfg = json.loads(Path("configs/sglang-fsdp2-h20.json").read_text())
cfg["algorithm"] = "topk_cross_entropy"
cfg["algorithm_options"] = {"k": 8}
rounds = fit(cfg, data_path="data/native-smoke.jsonl",
             output="runs/topk-example", state_dir="states/topk-example")
```

`fit` 复制传入的配置，每轮返回已提交版本、有效 token 数及执行器提交回执。命令行与该入口使用相同的实现。已有配置省略 `algorithm`、`policy` 和 `cluster` 时会使用训练端 K1、merge 策略和本地执行。

## 添加 MOPD 算法

继承 `DistillationAlgorithm`，实现 `token_loss(TokenBatch)`。返回每个响应 token 的可微损失，形状与 `student_logp` 相同。算法可以用 `route(example, config)` 为一条请求选择冻结教师，也可以沿用数据里的 `teacher_routing_key`。在数据读入时确定路由，此后它会随轨迹跨卡传递。各 rank 只对本地样本做反向，整轮的有效 token 数由 trainer 统一结算。[算法协议](../algorithms/base.py) · [全局归一化](../backends/fsdp2.py)

| 注册名 | 所用监督与目标 |
|---|---|
| `on_policy_k1` | 使用 FSDP2 当前前向的 sampled logprob 和冻结 teacher 计算 K1。每轮一次更新，旧策略以冻结 actor 单独前向重算，按全批有效 token 归一 |
| `sampled_reverse_kl` | 使用实际采样 token 的行为和教师 logprob，计算 reverse KL 的 score-function 梯度 |
| `sampled_forward_kl` | 在 student 访问的前缀上用 `exp(teacher_logp - behavior_logp)` 加权交叉熵。可设置 `max_importance_weight` 以截断权重，截断会改变估计量 |
| `topk_cross_entropy` | SGLang 对每个前缀位置返回教师前 K 个 token ID 与 logprob。FSDP2 在完整 student softmax 下取相应位置，以教师前 K 项归一化后计算交叉熵 |

例如，已有代码包可以按教师设置不同的训练权重。

```python
from DynaMOPD.algorithms.sampled import SampledReverseKL

class WeightedReverseKL(SampledReverseKL):
    def token_loss(self, batch):
        weight = self.options["route_weights"][batch.sample["teacher"]]
        return weight * super().token_loss(batch)
```

将 `algorithm` 设置为 `my_lab.objectives:WeightedReverseKL`，将 `algorithm_options.route_weights` 设置为路由名到权重的映射。driver 与所有训练 rank 通过相同的模块路径导入实现。这个例子保留按整轮有效 token 总数归一的约定。插件代码须在所有节点可导入。

当前监督协议提供 sampled logprob 和 teacher top-k 两种形式。新增一种形式涉及算法声明、评分 worker 的生产逻辑、按 token 区间的交付和训练 worker 的消费逻辑。driver 在分配 GPU 前检查算法与执行器的能力是否匹配。HF 参考执行器支持 sampled 监督，SGLang/FSDP2 执行器支持两种形式。SimRL 可模拟两种监督的区间和传输成本，其中的合成分数只用于调度。当前每条轨迹路由到一位教师，同一轨迹的多教师集成还需要扩展监督日志的路由维度。[监督日志](../scheduler/protocol.py) · [评分实现](../backends/sglang_engine.py)

## 添加调度策略或执行器

将 `policy` 设置为内置的 `natural`、`merge`，或自定义策略类的 `package.module:Class`。类接收 `policy_options` 字典，`plan(state)` 可以在一个决策点依次 yield 多个兼容动作。观察包括各组放置位置、当前角色、在途工作、可见轨迹、训练状态及历史成本。每个 yield 后 controller 先领取该动作的资源，再恢复策略迭代。模拟器中的真实尾长不在这个接口中。[策略接口](../scheduler/base.py) · [内置策略](../scheduler/policy.py)

策略通过 `Action(kind, gpu, version, payload, peer)` 选择逻辑卡组。decode、train、move 的 `items` 只需提供轨迹 ID。score 另提供教师路由和每条轨迹的右开 `end`。controller 从已结算的日志重建实际 token、版本及监督。迁移先 hold 一对组，再 move 或 unhold。优化器提交与轮末全卡换权由 controller 发起。

新增执行器时，将 `engine` 设置为其类的模块路径。构造器接收作业配置，报告 `placements`、支持的监督与检查点能力，并实现 `now / submit / wait / close`。支持恢复时还须校验版本目录。`wait` 返回原动作与完成回执。所有动作的 start/end 使用同一个计时域，远端单次执行耗时可作为单独字段记录。本地执行器和 Ray actor 共用 `workers.group.GroupWorker`。自定义传输可以直接复用它和 `GroupExecutor` 的迁移流程。[执行器接口](../backends/base.py) · [组件解析](../backends/registry.py)

## 放置到 Ray 集群

32 卡配置在四台机器各预留 8 卡，64 卡配置在八台各预留 8 卡。每个节点的一个 Ray actor 持有四个双卡组，组内的 SGLang TP1 副本独立生成，各组可以并发评分或执行 FSDP2 微批。节点内用 actor 实际获得的可见卡序号启动训练 rank，日志按全局组号隔离。跨节点迁移传输已提交的前缀和采样状态，目标重新 prefill。可选的 TP1→TP4 扩容只在同一节点的两个双卡组间执行。现有长尾策略面向全局最后一条活跃轨迹，一次租用一对组。[节点执行](../workers/node.py) · [Ray 放置](../cluster/ray.py)

轮末，每个 FSDP 本地 rank 与其他组的对应 rank 组成一条 NCCL 通信组。所有组参加梯度归约，优化器所有者按本轮有效 token 总数归一化并更新一次。随后同步 FP32 训练参数，广播供推理使用的 BF16 权重。SGLang 通过同机 CUDA IPC 接收各桶。跨节点通信组使用第一个分配节点上的 TCPStore 建连，本地 FSDP 组在节点的临时目录建连。输出、模型暂存和持久检查点仍使用相同绝对路径的共享文件系统。节点临时目录由各节点的 `TMPDIR` 或系统默认值决定。[PyTorch TCPStore](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.TCPStore) · [梯度与参数同步](../backends/replica_sync.py) · [权重发布](../backends/weight_sync.py)

### 在分配 GPU 前核对拓扑

从仓库根目录运行。把 `size` 设为 32 或 64。命令打印节点、逻辑组、各 bundle 的 CPU/GPU 预留和候选 TP4 配对，不连接 Ray。两个配置分别需要 4 或 8 个节点，每个节点至少有 8 个可预留的整卡 GPU 和 16 个 Ray 逻辑 CPU。

```bash
size=32
base=configs/sglang-tp1-fsdp2-8xh20.json
cluster=configs/ray-${size}gpu.json
tp=configs/h20-rollout-tp1-tp4.json
bash ./launch.sh --config "$base" --cluster-config "$cluster" \
  --rollout-tp-profile "$tp" --check-layout
```

### 启动真实训练

各节点使用同版本 Ray、SGLang、PyTorch、CUDA 和训练依赖，包括配置选用的 FlashAttention。共享代码、模型、数据、状态和输出目录，节点内核缓存可放本地。各节点在启动 Ray 前设置适合本机的 NCCL 网卡和环境路径。Ray 节点地址、PyTorch TCPStore 动态端口和 NCCL 网络须在分配的节点间可达。节点还需有容纳 FSDP 参数、优化器和休眠引擎的主机内存，`--check-layout` 只核对逻辑预留。

在分配的 GPU 主机上设置 `HEAD_IP` 为 Ray 头节点的集群网 IP，`THIS_IP` 为当前节点的集群网 IP。使用同一训练环境的解释器启动 Ray。如果已有可用的 Ray 集群，直接提交后面的作业。

```bash
PY=${PYTHON:-python}
# 头节点
"$PY" -m ray.scripts.scripts start --head --node-ip-address="$HEAD_IP" --port=6379 --num-gpus=8
# 每台其余 GPU 节点
"$PY" -m ray.scripts.scripts start --address="$HEAD_IP:6379" --node-ip-address="$THIS_IP" --num-gpus=8
```

在头节点从仓库根目录提交一次，沿用上面的 `size`、`base` 和 `cluster`。示例沿用八卡作业的 K1、全局 batch 512、回答上限 8192、TP1 和每个 FSDP 组两卡。训练组上限放开至本次分配的组数，实际参与训练的组由控制器决定。若实验需要和固定 4 个训练组的结果对齐，将上限设为 4。

```bash
job="$PWD/runs/dapo-${size}gpu"
bash ./launch.sh --config "$base" --cluster-config "$cluster" \
  --data "${TRAIN_DATA:?set TRAIN_DATA to your training JSONL or Parquet}" \
  --max-train-groups "$((size / 2))" --out "$job" --state-dir "$job/state"
```

默认沿用 `merge` 策略。要评估长尾扩容，在作业命令中加入 `--policy elastic_tp --rollout-tp-profile "$tp"` 并使用独立作业目录。省略成本文件中的 `group_pairs` 会按每台节点内部的实际组顺序配对，也可显式给出合法子集。成本先验来自单机 H20。多机梯度归约、权重广播耗时和吞吐仍需实测。[32 卡配置](../configs/ray-32gpu.json) · [64 卡配置](../configs/ray-64gpu.json) · [TP 成本先验](../configs/h20-rollout-tp1-tp4.json)

需要固定节点时，用 `cluster.nodes` 指定 Ray 节点 IP 和各节点组数。传入 `--cluster-config` 时 CLI 从节点配置推导组数，使用 Python `fit` 时显式令 `groups` 等于各节点组数之和。指定节点或节点数时按节点严格分散预留，省略拓扑时保留按组自动放置。trace 的 `executor_ready.placements` 记录节点、设备和全局组，`wide_mesh_ready` 记录实际备用引擎和启动开销。SimRL 使用相同的节点布局，`costs.cross_node_move_s` 可配置跨节点迁移的附加成本。[Ray placement group](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html) · [单节点 Ray 示例](../configs/ray-cluster.json) · [拓扑协议](../cluster/topology.py)

### 调度插件的交接边界

`score` 派发时固定每条轨迹的教师和响应区间 `[start, end)`。在途生成可以继续延长轨迹。控制器先用本次派发的区间核对回执，再向轨迹日志交付分数。`decode` 回执从已派发的前缀终点起步，新增 token 数受该动作步长和样本剩余额度约束。[动作与轨迹](../scheduler/protocol.py) · [结算](../scheduler/controller.py)

`hold` 可以在两组仍在解码时预约迁移。`move` 要等两组到达安全点，源组每条活跃请求都已有采样位置。目标确认安装后，源组确认释放同一批 ID，控制器才更新 owner 并把两组的剩余显存交给下一次决策。[跨组迁移](../backends/executor.py) · [组内执行](../workers/group.py)
