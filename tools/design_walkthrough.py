"""Render an offline Chinese design document for the DynaMOPD training system (stdlib only)."""
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
OPS = {"decode": ("学生生成", "decode"), "score": ("教师评分", "score"),
       "train": ("微批训练", "train"), "move": ("请求迁移", "handoff"),
       "switch": ("角色切换", "handoff"), "tp_expand": ("TP 扩容", "handoff"),
       "tp_shrink": ("TP 缩容", "handoff"), "reduce_grads": ("梯度归约", "commit"),
       "commit": ("优化器提交", "commit")}
LABELS = dict(decode="生成", score="评分", train="训练", handoff="交接", commit="归约 / 发布")


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


def source(ref, out, simrl):
    path, name, marker = ref
    target = (simrl / path[6:]) if path.startswith("SimRL/") else ROOT / path
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
    chart = ['<figure><figcaption>轮内卡组分工与提交边界 <small>机制示意 · 无实测时间比例</small></figcaption>']
    for i, row in enumerate(groups):
        bars = ''.join(f'<span class="bar {k}" style="left:{a}%;width:{b-a}%">{LABELS[k]}</span>' for k,a,b in row)
        chart.append(f'<div class="lane"><b>卡组 {i}</b><div class="track">{bars}</div></div>')
    chart.append('<div class="lane"><b>全局</b><div class="track plain"><span class="bar commit" style="left:89%;width:11%">提交</span></div></div></figure>')
    return ''.join(chart)


def prefix_example():
    return ('<figure><figcaption>一次学生迁移和教师缓存回收后的轨迹状态'
            '<small>同一 rid，偏移为已提交的响应 token 数</small></figcaption>'
            + table(["事件", "学生组", "已生成", "监督已交付", "教师 KV", "可训练"], [
                ("学生提交 t₀…t₃", "G0", 4, 0, "—", "否"),
                ("教师返回 [0, 4)", "G0", 4, 4, "G1 至偏移 4", "否"),
                ("G2 接管并生成 t₄", "G2", 5, 4, "G1 至偏移 4", "否"),
                ("G1 回收 KV", "G2", 5, 4, "—", "否"),
                ("G2 提交 t₅（EOS）", "G2", 6, 4, "—", "否"),
                ("教师补齐 [4, 6)", "G2", 6, 6, "—", "是")])
            + '<p class="caption">G1 回收缓存后，下一次评分需重读上下文。前四个标签仍然有效。</p></figure>')


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
    clock = {"sglang_fsdp2": "真机回执", "simrl": "模拟时钟"}.get(backend, "执行器回执")
    return (f'<aside class="observation"><strong>运行观测</strong><p>在 v{version} 的这一轮中，首次评分始于'
            f' {score:.2f} s，首次微批训练始于 {train:.2f} s，最后一个生成批次在 {last:.2f} s 返回。'
            f'三类 worker 动作同时在途的墙钟时间合计 {joint_seconds:.2f} s。'
            f'时间取自同轮{clock}，含批次内等待。<a href="#case-study">本轮卡组时间线 ↗</a></p></aside>')


def design_sections(trace):
    return [
        section("motivation", "动机与依赖", "整批同步与轨迹级数据依赖",
            "在同步多教师 on-policy distillation 中，同一批请求的回答长度可以相差很大。如果整个生成池必须等最后一个请求结束才进入评分，短请求所在的卡组已经排空，对应教师却仍在等待。固定教师卡池可以提前评分，但提前划走的卡也会减少开轮时的生成副本。DynaMOPD 让全部卡组从 rollout 开始，随本轮实际进度重新分配资源。",
            '<h3>评分与反向传播的就绪条件</h3>'
            '<p><strong>前缀生成与教师评分。</strong> 对冻结的因果教师，已采样 token yᵢ 的监督为 '
            '<code>log p<sub>T</sub>(yᵢ | x, y<sub>&lt;i</sub>)</code>。后续生成不会改变这个条件分布。'
            '因此学生一旦提交前缀，路由对应的教师就有了该段评分所需的全部输入。一个教师批次可以包含来自不同轨迹的已提交片段。</p>'
            '<p><strong>长请求生成与短样本反向。</strong> 一个样本终止且教师监督齐备后，该样本的损失只依赖自己的数据和轮初参数 θᵥ。'
            '一个空闲训练组即可用 θᵥ 累积梯度，同时其他卡继续生成。轮末汇齐所有有效 token 的梯度、统一归一化并更新，随后发布 θᵥ₊₁。</p>'
            + motivation_observation(trace) + example_timeline(),
            "重叠能缩短轮时的条件是提前执行的工作原本位于关键路径上。若换岗、重新 prefill 或借走生成卡造成的延迟更长，保持当前布局可能更快。因此调度目标取下一次权重发布的完成时刻。",
            [("trainer/driver.py","fit","Controller("),("scheduler/controller.py","Controller.run_round","for a in self.policy.plan")]),
        section("state", "数据与控制", "轨迹状态与资源所有权",
            "轨迹以 (rid, student_version) 标识。控制器保存采样 token、行为 logprob、教师路由及已交付的标签。学生请求的执行归属 owner、监督交付偏移 len(labels)、各教师执行位置的可复用 KV 偏移 kv[group] 分别更新。这使学生迁移和教师缓存驱逐可以独立发生。",
            flow([("控制器", "从已结算状态选择动作"), ("卡组租约", "锁定动作所需的一组或两组卡"),
                  ("执行器", "本地或 Ray worker，或 SimRL 虚拟执行器"), ("回执", "校验版本和偏移，更新状态后释放租约")])
            + prefix_example() + '<h3>评分边界</h3><p>评分区间 [a, b) 使用 SGLang 的输入 logprob。引擎从 '
            '<code>len(prompt) + a − 1</code> 读取，首项为前驱 token 和空分数。worker 校验返回的 token ID，'
            '跳过前驱并交付恰好 b−a 个分数。对于首段，前驱是 prompt 的最后一个 token。</p>',
            "训练门槛同时检查终止标志、所需监督的完整性和样本认领状态。EOS 总是有效终止，启用 train_on_length 后，达到长度上限的样本也纳入训练。",
            [("scheduler/protocol.py","Row.deliver","self.labels.extend"),("scheduler/protocol.py","Row.ready","return"),
             ("backends/sglang_engine.py","Inference.score","starts =")],
            notes=[
                ("在途评分固定区间", "派发时冻结本次 [start, end) 与目标 token。学生可以继续延长前缀。回执按当时派发的区间校验，下一段从已交付位置续接。若当前路由仍有活跃轨迹，暂时没有待评前缀也可能很快产生新工作。",
                 ("scheduler/controller.py","Controller.materialize","item.update(start=start")),
                ("用真引擎探测边界", "启动检查向 SGLang 提交长短不同的两条前缀，逐位置比较整段评分、缓存续评和清空缓存后的重算。对 rollout 还用相同 seed 比较连续采样与清缓存后续采的 token。检查覆盖固定探针和数值容差。",
                 ("backends/checks.py","check_server","for batch in (warmed, rebuilt)")),
                ("按路由核验采样 ID", "输入先检查学生和教师的 token 映射与 EOS。K1 在映射一致时，用学生与该条请求的教师输出层宽度核验 SGLang 已采样的 ID。这样即使另一个教师的输出层更小，这条轨迹的有效 token 也能交付。固定的轨迹 seed 独立于接下来使用的卡。",
                 ("trainer/data.py","vocabulary","sampled_id_limits")),
            ]),
        section("model", "成本建模", "事件模拟与在线校准",
            "原 SimRL 的弹性 rollout 模拟按时间推进每个推理实例，实例排空时释放卡组。DynaMOPD 复用 UnifiedRoofline、ScoringCost 和分页 KV 管理器，并将虚拟执行器接到与真机相同的 Action / 回执接口。对同一动作，真机等待 worker 返回，模拟器将预测完成时刻插入事件队列。",
            table(["操作","成本模型","并行边界"],[
                ("学生生成","生成步数、token 总量、注意力上下文、未命中的输入及 KV 分页约束","各推理副本分别估时，批次完成于最慢副本"),
                ("教师评分","按路由拟合固定开销和输入成本，复用仍在目标副本中的前缀","同一轨迹按偏移交付，跨轨迹可以成批"),
                ("微批训练","从每个 FSDP rank 的 packed 或 padded 输入宽度估算前反向耗时","取关键 rank，用就绪样本生成微批"),
                ("角色切换与发布","区分角色方向、状态驻留、教师启停、本地准备与共享传输","发布屏障等待最晚到达的组，同一段共享传输只计一次")])
            + '<p class="equation"><b>SimRL 单步模型</b><code>T = max(F, W + G·N + Aₚ·P/b + A𝒹·C)</code>'
            '<i>F 为最低步时，W 为固定项，N 为本步 token 数，P 为 prefill 注意力工作量，C 为 decode 上下文长度之和，b 为模型的块大小参数。</i></p>'
            '<p>对整批 decode RPC，拟合特征还包括各请求在本段累积的上下文工作量。初始上下文为 c，继续生成 s 步时，该项为 '
            '<code>s·c + s(s−1)/2</code>。因此，相同的生成 token 数可以对应不同的执行时间。</p>',
            "SGLang 真机路径在新的资源分配上清空旧时间系数。预建宽 TP 网格时测量唤醒、休眠和宽解码，首个训练轮在服务请求的同时收集批次和转岗回执。后续搜索采用本作业拟合出的成本并持续修正。已知终长可驱动离线回放，在线策略仅访问决策前已公布的进度和历史。规划器的 CPU 墙钟耗时独立记录为 planner_wall_s。",
            [("SimRL/srl/simulator/mopd_elastic.py","ElasticMopdSimulator._collect_batch","def advance"),
             ("SimRL/srl/cost/rollout/roofline.py","UnifiedRoofline._raw","return ("),("backends/simulation.py","Simulator.wait","heappop"),
             ("backends/sim_profile.py","decode_features","ticks ="),("scheduler/calibration.py","update_profile","fit_profile(cfg")],
            notes=[
                ("在首轮训练中测量", "标准 SGLang 策略启动时清除旧作业的时间系数，保留 TP 宽度和可用组对等拓扑约束。首轮照常生成、评分和训练。允许迁移且出现可行组对时，标定策略尝试一次真实交接，采到基础费用后停止探测。某类动作缺样本时，模型可用本作业已观测的粗粒度耗时估算。",
                 ("scheduler/calibration.py","calibration_round","Measure one real")),
                ("按物理副本计费", "同一个卡组可运行多个窄 TP 副本。模型在各副本内按 KV 页容量切分窗口，计入冷输入和累计上下文，再取最慢副本的完成时刻。若观测中的 decode 段总是跑满 quantum，固定启动成本与逐步成本无法单独辨识，拟合时将这部分归入每步，防止严重高估很短的尾段。",
                 ("tools/worker_costs.py","fit_decode","inseparable")),
                ("区分等待和实际传输", "离线发布模型对齐所有参与组到达 collective 的时刻，将各组本地准备与一次共享传输拆开。真机首轮标定路径直接观测全部组通过发布屏障的墙钟时间。异构教师的 CPU 准备若与旧角色卸载重叠，租约模型只把交接后仍需等待的部分计入。",
                 ("backends/sim_profile.py","teacher_lease_window","def teacher_lease_window")),
                ("区分估时误差和决策误差", "离线可为虚拟 worker 单设 sim_execution_profile，保留调度器自己的估计。固定真机动作顺序回放可以检验服务时间与缓存命中估算，自由调度回放还能检验动作选择。设置 sim_timing_seed 可以按组和操作抽取实测时延倍率，轨迹的采样 seed 保持独立。",
                 ("tools/trace_timing.py","replay","The recorded order")),
            ]),
        section("scheduling", "联合调度", "以权重发布时间为目标的动作搜索",
            "在每个决策点，控制器先结算已返回的批次。cost_search 根据请求归属、可用卡组、在途工作、监督积压、就绪样本和显存，构造一组可行布局。每种布局再比较评分组、评分与训练批量，以及立即服务或等待更多前缀的排程。",
            table(["候选布局","潜在收益","计入预计余时的代价"],[
                ("继续生成，等待自然排空","保留当前生成并行度","评分和就绪微批等待卡组释放"),
                ("教师 A 保持驻留，另一组服务 B 或训练","多个路由或训练批次并行推进","旧工作排空、角色加载、成批等待以及教师 KV 驱逐"),
                ("合并两个实例上的请求","提前释放源组用于评分或训练","等待两端安全点、目标接纳、前缀重算以及合并后的生成负载"),
                ("扩大尾部请求的 TP，再归还卡组","降低较长剩余序列的逐步解码时间","宽布局交接、借卡期间推迟的工作和缩容")])
            + '<p>对每个候选，预测器沿数据依赖和卡组占用时间向前推演，选择预计发布最早的布局。可并行的动作按各自卡组派发。'
            '各候选都需要、且与争用组无关的 decode 可以先发，以减少 CPU 搜索造成的空档。</p>',
            "首轮标定采用单独的资源分配策略。长度预测在冷启动时可用回答上限估计一条潜在长尾，后续利用已完成轮的路由分布和当前前缀，最后一条活跃请求还可参考轮最大长度的 EMA。候选数随布局与搜索预算裁剪，预计最优限于本次评估的候选集合。",
            [("scheduler/search.py","CostSearchPolicy.plan","f = Forecast("),("scheduler/search.py","Forecast.project","def project"),
             ("scheduler/search.py","CostSearchPolicy.plan","estimates.append"),("scheduler/controller.py","Controller.submit","overlapping exclusive leases")],
            notes=[
                ("开轮放置与副本负载", "可选 balanced 放置按路由历史长度和 prompt 成本估计工作量，在同一副本序号内先分较重的请求，保留各组原始槽数。仅当预计最大负载下降时采用新归属。首轮缺成本样本时沿用初始放置。",
                 ("scheduler/search.py","CostSearchPolicy.place_rollouts","if max(after)")),
                ("校正先完成样本的短偏差", "轮初冻结已完成轮的长度历史。被旧上限截断的样本，只在旧上限足以覆盖当前上限时才能用于推算该上限下的长度。可选条件分位数对同路由、同进度、同上限的活跃请求分配不同分位，避免把一批请求全预测成中位长度。最后一条活跃请求可用轮最大长度 EMA，前缀超过 EMA 后回退到条件估计。",
                 ("scheduler/search.py","Forecast.estimate_quantiles","order statistics")),
                ("用空档推进短 decode", "两端到达迁移安全点的时间可能不同。merge_backfill 用二分搜索找出预计能塞进等待窗口的最长 decode 段，把多生成的 token 及其重算成本放回候选排程，再与直接等待比较。两端已预订时，控制器仍可撤销预约并重新规划。",
                 ("scheduler/search.py","Forecast.merge_backfill","while lo < hi")),
                ("路由顺序、提前加载和认领", "搜索比较按就绪时间、按教师聚集或优先排空某一路由。各轨迹的评分片段始终服从偏移依赖。prepare_teacher 可以先加载角色而不认领前缀。普通 score 触发冷加载时，关闭在线标定允许热副本先取走前缀。开启在线标定会认领该批并排在换岗之后，后续轮次也沿用这个模式。",
                 ("scheduler/controller.py","Controller.assign","Acquire only the engine lease")),
                ("多卡共享就绪队列", "实测成本路径会把按可用组数拆小的评分批次和训练微批也纳入比较。派发每个微批后立即认领对应样本，后续组从剩余队列取数。若仍需生成或评分，控制器为这些工作保留服务能力。候选较多时先下发各布局共有的 decode，再枚举争用组的动作。",
                 ("scheduler/calibration.py","share_ready","share = min")),
                ("提前释放空闲教师", "使用分项发布成本的搜索路径会比较两种选择：教师保持驻留直到发布，或者先退到 standby，把退出进程的时间藏在其他组的工作期间。在线标定路径直接使用实测整轮发布屏障耗时，当前这个分支没有单独搜索提前退租的收益。",
                 ("scheduler/search.py","Forecast.project","with_standby =")),
            ]),
        section("roles", "角色与加载", "卡组交接与异构教师加载",
            "switch 占用卡组的独占租约。worker 在安全点确认学生请求已结束或迁出，再处置旧缓存、核对显存并加载目标角色。控制器收到匹配的完成回执后更新组角色。这允许提前规划下一批任务，同时维持引擎级的互斥。",
            table(["条件","执行路径"],[
                ("推理布局兼容","复用同一推理引擎。启用冻结教师分桶同步时，经本地 CUDA IPC 装入该教师自己的权重。"),
                ("异构教师需要独立 rank","在 CPU 准备启动环境，GPU 租约交接后启动活动 rank。释放角色时退出活动进程，可选的 teacher image 减少重复初始化。"),
                ("训练状态占用显存","按新角色的显存需求决定 trainer 是否驻留，必要时将状态迁回主机。有待归约梯度时限制释放。")]),
            "切换成本可拆为旧角色释放、新角色准备、等待和加载。组级回执覆盖这些步骤的墙钟区间，其中可以包含同步等待。加载完成和显存检查通过后，该组才接收对应的评分或训练批次。",
            [("workers/group.py","GroupWorker.switch","if self.infer.owned"),("workers/model.py","inference_signature","def inference_signature"),
             ("backends/teacher_pool.py","TeacherRanks.activate","def activate"),("backends/teacher_rank.py","prepared_rank","def prepared_rank")],
            notes=[
                ("重用冻结教师的 TP 分片", "可选 teacher_rank_images 将首次从 HF 装入的稠密、非量化 Qwen2 教师保存为 SGLang 原生 rank 分片。镜像按模型来源、TP、设备类型与软件版本区分。仅一个副本持锁写临时目录，分片齐全后原子发布，其他组可继续从 HF 加载。首次使用镜像时按固定前缀对照 HF 的逐位置 logprob。",
                 ("backends/teacher_image.py","TeacherImage.build","LOCK_NB")),
                ("把 CPU 准备移出 GPU 租期", "异构教师可提前导入运行时。给予 GPU 租约后才启动活动 scheduler，退出时回收 rank 及辅助进程。预检按路由、按卡记录显存需要，后续切换结合当前余量决定能否保留 trainer。trace 的 standby_wait_s、teacher_rank_load_s 和 teacher_rank_release_s 可区分准备未完成、加载和退出。",
                 ("backends/teacher_pool.py","TeacherRanks.release","def release")),
                ("保留图捕获的缓冲区", "兼容布局仍须装入所选模型的权重。可选 Qwen2 路径允许在受支持的未缩放 RoPE 上原位重建 cos/sin 缓存，保留 CUDA Graph 捕获的地址。启用 request_seed 启动器时，可在捕获 decode 图后恢复 batch-invariant prefill，兼顾解码图和缓存续评。",
                 ("backends/qwen2_rope.py","restore","cache.copy_")),
                ("按版本保留主机副本", "显存充足时 trainer 可以驻留。启用主机镜像后，FSDP 进程在回复发布完成后复制主参数或 Adam 状态，并标记版本。以后必须卸载时直接接回匹配版本的副本，过期部分重新拷贝。这会占用主机内存和传输带宽，下一条需要该 rank 的命令也可能等待它完成。",
                 ("backends/fsdp2.py","Shard.sleep","host_moments_version")),
            ]),
        section("migration", "迁移与 KV", "请求迁移、缓存重建与弹性 TP",
            "普通合并移动学生请求的执行归属。源组导出采样 seed 和绝对位置，控制器提供 prompt 和已提交的 token。目标组核对版本、归属与剩余 KV 容量，接受后源组才释放请求。目标拒绝时，源组保留所有权。",
            flow([("hold", "预约两端安全点"),("export / install", "传递采样状态，目标核验容量"),
                  ("release", "目标确认后释放源端"),("decode", "在目标组上继续采样")])
            + table(["对象","复用与重建策略"],[
                ("学生 KV","目标利用已有缓存，未命中的前缀可在接管时预填充，也可延后至下一次 decode。后者能使源组更早开始新工作，但仍消耗目标组算力。"),
                ("教师 KV","按独立推理副本核算预算，超限时只清理受影响副本的缓存并撤回相应偏移。控制器继续保留已经交付的监督。"),
                ("尾部宽 TP","真实 SGLang 路径预建同节点配对的宽引擎，发布同版权重后休眠。接管时唤醒并重放前缀，核验末个已提交 token 的 logprob。SimRL 还支持单独研究直接搬运 KV 分片的成本。")])
            + '<p class="equation"><b>局部 TP 收益估计</b>窄解码时间 − 宽解码时间 − 进出宽布局成本 − 被推迟的服务 − 丢失教师 KV 的重算成本</p>',
            "扩容要求同节点的这一对组只剩一个活跃请求，其他组仍可生成。控制器同一时刻只允许一对组占用宽布局，worker 在唤醒前复查物理显存。长请求终止后归还卡组，若评分或训练压力先上升，也可提前缩容。tp_eligibility 记录所有权、容量和在途工作等候选筛选原因。",
            [("backends/executor.py","GroupExecutor.execute","src = self.call"),("backends/sglang_engine.py","Inference.install","contexts ="),
             ("backends/elastic_rollout.py","WideRollout.execute","handoff_logp_error"),("scheduler/search.py","Forecast.layouts","handoff ="),("backends/sim_tp.py","TailTP.transition","def transition")],
            notes=[
                ("接管前按副本验容量", "普通合并在每个目标推理副本上分别核算已驻留请求、迁入上下文和下一个 quantum 的额度。一个副本的余量不能抵扣另一个副本的短缺。若目标拒绝，控制器记住这对组当时的活跃请求集合，避免在相同集合上反复尝试。",
                 ("backends/inference_replicas.py","InferenceReplicas.migration_fits","def migration_fits")),
                ("控制 KV 驱逐和评分准入", "SGLang 以副本为单位清教师缓存。超预算时先找能覆盖缺口的最小副本，若单个副本不足，先清占用最大的。独立教师 rank 按页向上取整，并为每条请求预留分配器空间，把大批拆成能容纳的窗口。即使实际缓存已被引擎逐出，冷启动窗口仍可接纳。",
                 ("backends/score_cache.py","trim_score_cache","enough =")),
                ("预建宽引擎和启动门控", "开训前建立同节点宽 TP 通信和引擎，按单请求解码捕获 batch 1 图（启用 CUDA Graph 时），测量唤醒、休眠、重算和宽解码。FSDP 进程可并行做 CPU 导入，宽引擎休眠后才开放 CUDA 初始化。休眠后的实际残留与再次唤醒的显存分别记录为 idle_device_bytes 与 wake_bytes。",
                 ("workers/node.py","NodeRuntime.prepare","gate =")),
                ("真机交接与模拟重分片", "真机宽引擎核对 seed、绝对位置和版本，重放前缀并比较末个 token 的 logprob，回执分别记录 wake_s、prefill_s 和 rebuilt_tokens。另一个仅供 SimRL 研究的方案按逻辑 KV 头估计分片传输，第一次借卡另计建通信组，提前缩容只搬回扩容后新增的部分。两个方案的传输字节含义不同。",
                 ("backends/simulation.py","Simulator.resize_tp","tp_prefix_tokens")),
            ]),
        section("publication", "梯度与发布", "全轮梯度归一化与版本发布",
            "每个 FSDP2 组只认领就绪样本，并以 θᵥ 完成本轮前反向。对齐 verl 的 sampled K1 路径逐响应位置计算损失。各微批累积损失梯度之和与有效响应 token 数，优化器组在轮末校验全局分母。",
            '<p class="equation"><b>全轮梯度</b><code>gᵥ = (∑ₘ ∑ⱼ∈Bₘ ∇ℓⱼ(θᵥ)) / N</code><i>Bₘ 为微批的有效响应位置，N 为全轮这些位置的总数。不同长度微批按其有效 token 数贡献梯度。</i></p>'
            + flow([("归约", "同一分片 rank 汇集梯度"),("commit", "除以 N、裁剪、执行一次 AdamW"),
                    ("权重发布", "节点内构造或接收 BF16 桶"),("rollout_barrier", "所有生成组确认 v+1")]),
            "多副本张量发布先同步主参数分片，各节点可从本地分片构造训练参数的 rollout 桶，需分发的冻结参数或缓冲区由权威副本提供。所有接收端确认后才能复用发布缓冲区。下一轮采样等待版本屏障，checkpoint 持久化另行记录。",
            [("backends/fsdp2.py","Shard.commit","g.div_(n)"),("backends/replica_sync.py","reduce_gradients","def reduce_gradients"),
             ("backends/weight_sync.py","TensorPublisher.bucket","def bucket"),("scheduler/controller.py","Controller.run_round",'self.emit("rollout_barrier"')],
            notes=[
                ("对齐 verl 的损失输入", "默认 on_policy_k1 在同一轮初 actor 上无梯度重算 old logprob，再做有梯度的前向。已采样位置的 teacher 与当前 actor 的 logprob 差构成停止求导的 advantage，沿 verl 对照的 PPO clip 和 dual clip 计算损失。这让提前反向仍使用与同步对照相同的目标。",
                 ("algorithms/verl_k1.py","policy_loss","advantage =")),
                ("按最慢 rank 优化训练", "启用变长 FlashAttention 时，较长样本先分给累计 token 最少的 rank。各条序列重置 position ID，用 cu_seqlens 隔开注意力，只对预测回答的位置投影词表 logits。空 rank 走相同 collective，以零损失和零有效 token 参与。搜索中的 train_width 使用对应的 packed 或 padded 布局。",
                 ("backends/fsdp2.py","Shard.response_logits","logits_to_keep")),
                ("固定通信组接受动态训练成员", "优化器所在组仍可能在跑长尾，其他组已能完成全部反向。归约前先在小张量上检查版本、分母和梯度存在性。跨组用同序号 FSDP rank 及有上限的 FP32 分桶，推理组以零贡献加入 collective，模型主体可留在 CPU。",
                 ("backends/replica_sync.py","reduce_gradients","Exchange the actual")),
                ("一次构桶，窄宽引擎并发消费", "各副本确认更新后的 FP32 分片后，可在本地重建只含可训练参数的 BF16 桶，减少跨节点重复广播。窄、宽引擎并发加载时，同一源张量为每个消费者各导出独立 CUDA IPC 句柄，均确认后才复用内存。最后一轮无需再给宽引擎发新权重。wide_publish_exposed_s 单列宽发布超过同组窄发布的时间。",
                 ("workers/group.py","GroupWorker.load_buckets","parallel =")),
                ("分开计量提交、发布与恢复", "内存发布每轮发生，HF 导出和包含优化器的 DCP 检查点按配置写入。可恢复检查点同时保存轮最大长度 EMA。只导出模型的版本适合评估，继续训练要从含训练状态的版本恢复。端到端作业、轮时、去掉持久化的稳态步时有不同边界。",
                 ("backends/fsdp2.py","Shard.save_checkpoint","scheduler_state")),
            ]),
        section("interfaces", "扩展与对照", "拓扑、算法与可复现对照",
            "系统把资源预留、训练目标和决策策略分开。比较一种调度改动时，可以固定数据、路由、初始参数、引擎布局与训练目标，只替换这一项。",
            table(["边界", "实现与实验含义"], [
                ("节点和卡组", "Ray 在启动引擎前预留作业所需资源，检查物理 GPU 归属。每个节点执行本地 worker，异构教师可读节点本地暂存权重。常规迁移可跨节点，宽 TP 限于同节点的已建组对。"),
                ("蒸馏目标", "DistillationAlgorithm 声明监督类型和是否重算 old logprob，route 选择冻结教师，token_loss 返回逐位置损失。可接 sampled KL 或 teacher top-k，复用同一版本屏障、微批协议与全轮分母。"),
                ("调度消融", "固定 cost_search 与负载，分别设置 policy_options.live_prefix_scoring=false、early_training=false、max_pairs=0 或 allow_tp_expansion=false。merge_backfill 可单独关。扩容对照保留相同宽引擎、预留显存和发布开销。"),
                ("观测", "dispatch 记录派发，complete.start/end 记录 worker 动作区间，双组动作共用一次回执。Ray 的 GPU 采样对齐 driver 时钟并保留时钟不确定度。活动时间线与 NVML 利用率各有独立数据来源。")]),
            "同一调度器内关闭一项动作可以测它对整轮的边际影响。与原版 KDFlow、verl 比较时还要分别核对 GPU 数、TP、副本数、实际输出与训练 token，以及计时边界。",
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
                raise ValueError("round_end 必须晚于 round_start")
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
                for key, label in (("export", "导出"), ("install", "接管"), ("release", "释放")):
                    start, end = e.get(key + "_start"), e.get(key + "_end")
                    if isinstance(start, (int, float)) and isinstance(end, (int, float)):
                        phases.append(f"{label} {end-start:.3f} s")
                for key, label in (("wake_s", "唤醒"), ("prefill_s", "前缀重算"), ("sleep_s", "休眠"),
                                   ("kv_transfer_s", "KV 传输")):
                    if isinstance(e.get(key), (int, float)):
                        phases.append(f"{label} {e[key]:.3f} s")
                width = (f"TP{e['from_tp']} → TP{e['to_tp']}"
                         if e.get("from_tp") is not None and e.get("to_tp") is not None else OPS[op][0])
                if e.get("accepted") is False:
                    width += "（拒绝）"
                r["handoffs"].append((a, b, g, peer, width, "，".join(phases) or "未单列"))
            if decision is not None:
                r["receipts"][decision].append(info)
    if fallback is None:
        raise ValueError("所选 trace 或版本缺少成对的 round_start / round_end")
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
            label += "（未接纳）"
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
    return (f'<figure class="actual"><figcaption>每组的 worker 动作区间 <small>轮初至轮末 {duration:.2f} s</small></figcaption>'
            f'<div class="legend">{keys}</div><div class="lane"><b>秒</b><div class="axis">'
            f'<span>0</span><span>{duration/2:.1f}</span><span>{duration:.1f}</span></div></div>'
            f'{"".join(html_rows)}<p class="caption">悬停查看起止时间。同一双组操作标在两行，灰底表示没有对应动作回执。条长含 worker 和通信等待。</p></figure>')


def trace_section(trace):
    v, r, backend, groups = trace
    chart = actual_figure(r, groups)
    if r["handoffs"]:
        handoffs = [(f"{a-r['start']:.3f}", group_label(g, peer), width, f"{b-a:.3f}", phases)
                    for a, b, g, peer, width, phases in sorted(r["handoffs"], key=lambda x: x[0])]
        chart += ('<h3>本轮全部请求迁移和 TP 交接</h3>'
                  + table(["轮内开始 / s", "参与组", "动作", "交接耗时 / s", "回执中的分项"], handoffs)
                  + '<p class="caption">每次双组交接计一次。分项只覆盖回执单列的阶段，总耗时还包括 RPC 与等待。</p>')
    else:
        chart += '<p>这一轮没有记录请求迁移或 TP 交接的完成回执。</p>'
    if r["teacher_formats"]:
        chart += '<p>本轮教师换岗回执记录的加载格式为 ' + esc("、".join(sorted(r["teacher_formats"]))) + '。</p>'
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
            shown.append((mode, "、".join(f"G{n}" for n in pair) if pair else "—", f"{seconds:.3f}", "选中" if chosen else ""))
        chart += (f'<h3>轮内 {case["time"]-r["start"]:.3f} s 的决策</h3><p>每种布局列出预测最快的卡组组合，'
                  f'并保留实际选中的组合。预计余时从决策点算到本轮发布。</p>'
                  + table(["候选布局", "参与组", "预计余时 / s", "决策"], shown))
        selected = next((t for mode, pair, t in rows if mode == case["chosen"] and pair == case["pair"]), None)
        natural = next((t for mode, _, t in rows if mode == "natural"), None)
        if case["chosen"] != "natural" and selected is not None and natural is not None:
            delta = natural - selected
            direction = "短" if delta >= 0 else "长"
            chart += f'<p>按决策时的估计，选中布局的剩余轮时比自然排空{direction} {abs(delta):.3f} s。动作耗时由 worker 的起止回执另行计量。</p>'
        linked = r["receipts"].get(case["id"], [])
        if linked:
            actions = []
            for g, peer, op, a, b, accepted, role, transfer in linked:
                outcome = "未接纳" if accepted is False else "完成"
                detail = role if op == "switch" else transfer
                actions.append((group_label(g,peer), OPS[op][0], f"{a-r['start']:.3f}–{b-r['start']:.3f}",
                                f"{b-a:.3f}", outcome + (f" · {detail}" if detail else "")))
            chart += '<h3>该决策实际派发并收到回执的动作</h3>'+table(["组","动作","起止 / s","用时 / s","回执"],actions)
        else:
            chart += '<p>此决策没有可按 decision_id 与派发配对的完成回执。上图保留本轮全部已记录动作。</p>'
    else:
        chart += '<p>该轮未记录多候选的 schedule_search。图中展示实际返回的动作区间。</p>'
    clock = "模拟时钟" if backend == "simrl" else ("真机回执" if backend == "sglang_fsdp2" else "执行器回执")
    group_count = f"，共 {groups} 个卡组" if groups else ""
    return section("case-study", "运行案例", "布局预测与执行回执",
        f"学生版本 v{v} → v{v+1}{group_count}。同轮{clock}以 round_start 为零点。",
        chart, "决策按同一版本与 decision_id 关联到派发，再用独占组上的动作关联到完成回执。优先取第一条实际完成迁移或 TP 交接的多候选决策，否则取第一条多候选决策。预测差值描述决策时的估计。实际轮时收益需要通过匹配负载的对照运行测量。",
        [("scheduler/controller.py","Controller.settle",'self.emit("complete"'),("tools/audit.py","audit","def audit")])


STYLE = """
:root {
  --ink:#193136; --muted:#526767; --line:#d7e1de; --green:#006557; --wash:#f1f6f3;
  --gen:#26786b; --score:#496ea5; --train:#984461; --move:#a46629; --update:#6c529a;
  color:var(--ink); background:white;
  font:16px/1.72 "Source Sans 3","Noto Sans CJK SC","Microsoft YaHei",system-ui,sans-serif;
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


def write_guide(output, trace=None, version=None, simrl=None):
    output = output.resolve()
    if trace and trace.resolve() == output:
        raise ValueError("输出文件不能覆盖输入 trace")
    simrl = simrl or ROOT.parent / "SimRL"
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
            detail = (f'<details class="design-notes"><summary>实现取舍 · {esc(notes[0][0])}等 {len(notes)} 项'
                      f'</summary><dl>{rows}</dl></details>')
        refs = ''.join(source(ref, output, simrl) for ref in [*item["refs"], *(ref for _, _, ref in notes)])
        blocks.append(f'<section id="{anchor}"><h2>{esc(item["title"])}</h2>'
                      f'<p>{esc(item["lead"])}</p>{item["body"]}<p>{esc(item["closing"])}</p>{detail}'
                      f'<details class="sources"><summary>源码位置</summary>{refs}</details></section>')
    document = (f'<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">'
                f'<meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<title>DynaMOPD · 同步多教师蒸馏系统设计</title><style>{STYLE}</style></head><body>'
                f'<header><div><a href="#top">DynaMOPD</a><button id="print" type="button">打印文档</button></div></header>'
                f'<div class="layout"><nav aria-label="目录">{"".join(navigation)}</nav><main>'
                f'<div class="title" id="top"><h1>同步多教师蒸馏的轮内并行</h1>'
                f'<p>SimRL 成本建模、动态卡组调度与 SGLang / FSDP2 执行</p></div>'
                f'{"".join(blocks)}</main></div><script>{JS}</script></body></html>')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(document, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT/"runs"/"system-design.html", help="输出自包含 HTML")
    parser.add_argument("--trace", type=Path, help="可选的 events.jsonl 或包含它的运行目录")
    parser.add_argument("--version", type=int, help="轮初 student 版本，默认选择有多候选决策的首个完整轮")
    parser.add_argument("--simrl-root", type=Path, help="可选 SimRL 源码目录，用于显示原项目代码摘录")
    args = parser.parse_args()
    if args.version is not None and not args.trace:
        parser.error("--version 需要 --trace")
    trace = args.trace/"events.jsonl" if args.trace and args.trace.is_dir() else args.trace
    try:
        write_guide(args.out, trace, args.version, args.simrl_root)
    except (KeyError, ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
