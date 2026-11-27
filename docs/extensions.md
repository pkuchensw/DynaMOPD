# Extending DynaMOPD

## Configure a training job

If your environment already provides compatible versions of CUDA, PyTorch, and SGLang, install the local package with `python -m pip install --no-deps --no-build-isolation -e .`. For a new environment, the package declares `gpu`, `cluster`, and `parquet` extras. CPU simulation and replay use the base package and Python standard library, with no external simulator dependency. You can use `launch.sh` to run directly from a shared checkout. [Package configuration](../pyproject.toml)

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

`fit` copies the supplied configuration. It returns one report per round with the committed version, effective token count, and executor's commit receipt. The CLI calls the same function. If you omit `algorithm`, `policy`, and `cluster`, `fit` uses the on-policy K1 objective and local execution. The default policy is `cost_search` for training and `merge` for simulation. The CLI also honors a recipe's `sim_policy` when running a simulation.

## Add a distillation objective

Subclass `DistillationAlgorithm` and implement `token_loss(TokenBatch)`. Return a differentiable loss for each response token, with the same shape as `student_logp`. Override `route(example, config)` to select a frozen teacher per request, or use a route from the data, such as `teacher_routing_key`. The driver resolves the route at input time and keeps it with the trajectory across GPU transfers. Each rank backpropagates through its local samples. The trainer counts effective tokens across the entire round. [Objective interface](../algorithms/base.py) · [Global normalization](../backends/fsdp2.py)

| Objective | Supervision and loss |
|---|---|
| `on_policy_k1` | Uses sampled token log probabilities from the current FSDP2 forward pass and the frozen teacher to compute the K1 policy loss. A separate forward pass through the frozen actor recomputes the old policy's log probabilities. The optimizer updates once per round, with normalization over all effective tokens. |
| `sampled_reverse_kl` | Estimates the reverse KL score-function gradient using the behavior policy and teacher log probabilities of the tokens actually sampled. |
| `sampled_forward_kl` | Weights cross-entropy at student-visited prefixes by `exp(teacher_logp - behavior_logp)`. Set `max_importance_weight` to cap the weights. Capping changes the estimator. |
| `topk_cross_entropy` | SGLang returns the teacher's top K token IDs and log probabilities at each prefix position. FSDP2 looks up those IDs under the student's full softmax and computes cross-entropy using the renormalized teacher top K. |

For example, an external package can assign a different weight to each teacher.

```python
from DynaMOPD.algorithms.sampled import SampledReverseKL

class WeightedReverseKL(SampledReverseKL):
    def token_loss(self, batch):
        weight = self.options["route_weights"][batch.sample["teacher"]]
        return weight * super().token_loss(batch)
```

Set `algorithm` to `my_lab.objectives:WeightedReverseKL` and `algorithm_options.route_weights` to a mapping from route names to weights. The driver and every training rank import the same module path. Install the plugin on all nodes. This example keeps the round's total effective token count as the normalization denominator.

The supervision protocol supports sampled log probabilities and teacher top-k distributions. To add a representation, declare it on the objective, produce it in the scoring worker, deliver it for the corresponding token intervals, and consume it in the training worker. The driver checks the objective and executor capabilities before allocating GPUs. The Hugging Face reference executor supports sampled supervision. The SGLang/FSDP2 executor supports both forms. The built-in simulator models intervals and transfer costs for both and uses synthetic scores for scheduling. Each trajectory currently has one teacher. An ensemble of teachers on the same trajectory would need an additional route dimension in the supervision log. [Supervision log](../scheduler/protocol.py) · [Scoring implementation](../backends/sglang_engine.py)

## Add a scheduling policy or executor

Set `policy` to a built-in name such as `natural`, `merge`, `elastic_tp`, or `cost_search`, or supply a custom policy as `package.module:Class`. Its constructor receives `policy_options`. At each decision point, `plan(state)` can yield several compatible actions. The observation includes group placements, roles, in-flight work, visible trajectories, training state, and historical costs. After each yield, the controller reserves the action's resources before resuming the iterator. The simulator keeps the true remaining response length private. [Policy interface](../scheduler/base.py) · [Built-in policies](../scheduler/policy.py)

A policy selects logical GPU groups with `Action(kind, gpu, version, payload, peer)`. For `decode`, `train`, and `move`, each entry in `items` needs only a trajectory ID. For `score`, also supply the teacher route and the exclusive `end` position for each trajectory. The controller reconstructs tokens, versions, and supervision from the settled event log. Migration starts with a `hold` on a pair of groups, followed by `move` or `unhold`. The controller initiates the optimizer commit and publishes the new weights to every group at the end of the round.

To add an executor, set `engine` to its class's module path. The constructor receives the job configuration. The executor reports `placements`, supported supervision types, and checkpoint capabilities, and implements `now`, `submit`, `wait`, and `close`. An executor that supports resuming must also validate the version directory. `wait` returns the original action and its completion receipt. All action start and end times share one clock. Record remote execution duration in a separate field if needed. Local execution and Ray actors use `workers.group.GroupWorker`. A custom transport can reuse this worker and the migration sequence in `GroupExecutor`. [Executor interface](../backends/base.py) · [Component resolution](../backends/registry.py)

## Placement on a Ray cluster

The 32-GPU configuration reserves eight GPUs on each of four nodes, and the 64-GPU configuration does the same on eight nodes. One Ray actor per node owns four two-GPU groups. Each group's SGLang TP1 replicas generate independently. Separate groups can score or run FSDP2 microbatches concurrently. Training ranks use the visible devices assigned to the actor, and logs are separated by global group ID. For inter-node migration, the destination receives the committed prefix and sampling state and reruns prefill. Optional TP1-to-TP4 expansion uses two groups on the same node. The tail policy leases one pair for the last active trajectory across all groups. [Node worker](../workers/node.py) · [Ray placement](../cluster/ray.py)

At the end of the round, FSDP ranks with the same local index form an NCCL group across replicas. All groups participate in gradient reduction. The optimizer owner normalizes by the round's effective token count and updates once. The workers synchronize FP32 training parameters and publish BF16 inference weights. SGLang receives each bucket over node-local CUDA IPC. Inter-node communication uses a TCPStore on the first allocated node. Node-local FSDP groups rendezvous in a temporary directory under `TMPDIR` or the system default. Output, staged models, and durable checkpoints must be accessible at identical absolute paths on the shared filesystem. [PyTorch TCPStore](https://docs.pytorch.org/docs/stable/distributed.html#torch.distributed.TCPStore) · [Gradient and parameter synchronization](../backends/replica_sync.py) · [Weight publication](../backends/weight_sync.py)

### Check the layout before allocating GPUs

From the repository root, set `size` to 32 or 64. The command prints the nodes, logical groups, resource reservations for each bundle, and candidate TP4 pairs without connecting to Ray. The configurations require four or eight nodes respectively, each with at least eight reservable whole GPUs and 16 logical Ray CPUs.

```bash
size=32
base=configs/sglang-tp1-fsdp2-8xh20.json
cluster=configs/ray-${size}gpu.json
tp=configs/h20-rollout-tp1-tp4.json
bash ./launch.sh --config "$base" --cluster-config "$cluster" \
  --rollout-tp-profile "$tp" --check-layout
```

### Run training

Install matching versions of Ray, SGLang, PyTorch, CUDA, and training dependencies on each node, including FlashAttention when selected by the configuration. Share the code, models, data, state, and output directories. Kernel caches can stay on each node. Configure the NCCL network interface and environment paths before starting Ray. The Ray node addresses, dynamic PyTorch TCPStore ports, and NCCL network must be reachable across the allocation. Allow enough host memory for FSDP parameters, the optimizer, and sleeping engines. `--check-layout` verifies logical reservations only.

On each allocated host, set `HEAD_IP` to the cluster-network address of the Ray head node and `THIS_IP` to the current node's address. Start Ray using the interpreter from the training environment. If the cluster is already running, go straight to job submission.

```bash
PY=${PYTHON:-python}
# Head node
"$PY" -m ray.scripts.scripts start --head --node-ip-address="$HEAD_IP" --port=6379 --num-gpus=8
# Each additional GPU node
"$PY" -m ray.scripts.scripts start --address="$HEAD_IP:6379" --node-ip-address="$THIS_IP" --num-gpus=8
```

Submit once on the head node from the repository root, using the `size`, `base`, and `cluster` values above. This example uses the eight-GPU recipe's K1 objective, global batch size of 512, 8,192-token response limit, TP1 inference replicas, and two GPUs per FSDP group. The command allows up to the full allocation to train. The controller decides which groups participate. Use a limit of four when comparing with a job capped at four training groups.

```bash
job="$PWD/runs/dapo-${size}gpu"
bash ./launch.sh --config "$base" --cluster-config "$cluster" \
  --data "${TRAIN_DATA:?set TRAIN_DATA to your training JSONL or Parquet}" \
  --max-train-groups "$((size / 2))" --out "$job" --state-dir "$job/state"
```

This recipe uses `cost_search` by default. To evaluate tail expansion, add `--policy elastic_tp --rollout-tp-profile "$tp"` and use a separate job directory. Omit `group_pairs` from the profile to pair groups in placement order within each node, or supply an explicit subset of valid pairs. The cost priors come from a single H20 node. Multi-node gradient reduction time, weight broadcast time, and throughput still need hardware measurements. [32-GPU configuration](../configs/ray-32gpu.json) · [64-GPU configuration](../configs/ray-64gpu.json) · [TP cost priors](../configs/h20-rollout-tp1-tp4.json)

To pin placements, set `cluster.nodes` to the Ray node IPs and their group counts. With `--cluster-config`, the CLI derives the total number of groups from that configuration. When calling `fit` directly, set `groups` to the sum of per-node counts. An explicit node list or node count enforces placement on distinct nodes. Omitting the topology allows automatic per-group placement. The trace records nodes, devices, and global groups in `executor_ready.placements`, and actual standby engines and startup costs in `wide_mesh_ready`. The built-in simulator uses the same node layout. Set `costs.cross_node_move_s` to model additional migration cost between nodes. [Ray placement groups](https://docs.ray.io/en/latest/ray-core/scheduling/placement-group.html) · [Single-node Ray example](../configs/ray-cluster.json) · [Topology interface](../cluster/topology.py)

### Handoff boundaries for scheduling plugins

A `score` dispatch fixes the teacher and response interval `[start, end)` for each trajectory. In-flight generation may continue extending that trajectory. The controller checks the receipt against the dispatched interval before appending scores to the trajectory log. A `decode` receipt starts at the end of the dispatched prefix. The action's step size and the sample's remaining allowance bound the number of new tokens. [Actions and trajectories](../scheduler/protocol.py) · [Receipt handling](../scheduler/controller.py)

`hold` can reserve a migration while both groups are decoding. `move` waits until both groups reach a safe point and every active source request has a sampling position. The destination first confirms installation, then the source confirms release of the same request IDs. The controller updates ownership and exposes the groups' remaining memory to the next scheduling decision. [Migration between groups](../backends/executor.py) · [Group worker](../workers/group.py)
