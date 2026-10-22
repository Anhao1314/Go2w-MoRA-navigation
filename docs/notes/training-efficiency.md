# 训练效率调研笔记

[返回首页](../README.md) · [文档索引](README.md) · [研究问题](RESEARCH.md) · [执行计划](PLAN.md)

> 目的：在不破坏实验有效性的前提下，缩短 M1/A-B 对比的训练周期。调研时间 2026-10-23。
> 记录原则：区分"能自由优化的"（并行度、包装器、日志）与"会改变训练动态的"（env 数、rollout 长度）。

## 1. 我们的基线（实测，RTX 5060 Ti 16 GB）

| 指标 | 数值 | 来源 |
| --- | --- | --- |
| 单种子 1500 iters | 1204.7 s（seed 0）/ 1231.5 s（seed 1） | `m1_training.log` |
| 每迭代耗时 | 0.80 / 0.82 s | 同上 |
| 每迭代样本数 | 2048 envs × 24 steps = 49,152 | 配置 |
| 端到端吞吐 | ≈ **61,000 env-steps/s** | 推算 |
| GPU 利用率 | 36–56%，显存峰值 ~5.2 GB | `nvidia-smi` 采样 |
| CPU | 总占用 12–17%，主要 2 个核 | 进程采样 |
| 关键干扰 | 残留导出进程 + 并发 pyright 使速度掉到 1.80 s/iter | journal |

**初步判断**：61k steps/s 与官方在 L40S 上报告的 43–77k（4096 envs，G1）处于同一量级；考虑 5060 Ti 的算力差距，训练链路本身没有明显低效。GPU 未饱和、CPU 只用两个核，说明瓶颈是**单线程延迟**（仿真步进 + Python 侧 rollout 处理），不是算力总量。

## 2. 权威来源与关键发现

### 2.1 Rudin et al., "Learning to Walk in Minutes Using Massively Parallel Deep RL"（CoRL 2021）

- 单工作站 GPU 上，平地策略训练 **不到 4 分钟**，崎岖地形 **20 分钟**（ANYmal）。
- 关键结论：并行规模极大时，**每次更新的步数要保持很小**；论文报告少于约 25 步（0.5 s 仿真时间）会因 GAE 等原因不收敛。我们用的是 24 步——正好在论文的下限附近，值得做一个 32/48 的对照。
- **大 mini-batch**（数万样本）能稳定学习且不增加训练时间。
- **Reset 语义**：必须区分"失败重置"与"超时重置"，超时用 critic bootstrapping。这直接影响价值估计的偏差，也是复现时最常被忽略的细节。
- 引入**游戏式课程**（game-inspired curriculum），按机器人能力逐步加难度。

来源：[arXiv:2109.11978](https://arxiv.org/abs/2109.11978) · [PMLR v164](https://proceedings.mlr.press/v164/rudin22a.html)

### 2.2 Isaac Lab 官方性能基准

官方把吞吐拆成三段，这个拆分方式直接可用：

| 阶段 | 含义 |
| --- | --- |
| `environment_step_fps` | 纯仿真步进 |
| `environment_step_fps + inference` | 加上策略推理 |
| `total_fps` | 再加 PPO 更新 |

官方在 RTX 4090 上报告 4096 环境、G1 平地任务的 end-to-end **>100k steps/s**；L40S 上 4096 环境约 43–77k。文档特别提醒：**更快的环境步进不等于更快的端到端训练**，必须看学习曲线。

本地已有工具：`IsaacLab/scripts/benchmarks/benchmark_rsl_rl.py` 与 `benchmark_non_rl.py`，可分别测"完整训练"与"纯环境"。

来源：[Performance Benchmarks](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/overview/reinforcement-learning/performance_benchmarks.html) · [Run benchmarks](https://isaac-sim.github.io/IsaacLab/main/source/developer-tools/benchmarking/run_benchmarks.html)

### 2.3 包装器开销可能很可观

Isaac Lab 的 SB3 集成讨论中，有贡献者发现 **只处理已终止的环境**（而不是每个环境的完整数据）可把速度从约 10k 提升到 30–35k steps/s（约 3×）。这说明向量化环境的适配层常是隐藏瓶颈。

对 RSL-RL 的启示：`RslRlVecEnvWrapper` 的观测/动作转换、dones 处理值得单独计时，而不是假设官方实现一定最优。

（社区讨论：IsaacLab issue/discussion #1749、#1769；未找到可稳定引用的公开链接，作为线索记录。）

### 2.4 课程与域随机化的取舍

2025 年的 legged locomotion 综述与多篇工作反复指出：

- 奖励塑形、策略表示、训练稳定性对最终性能的影响大于算法选择；
- 域随机化能提升鲁棒性，但"可能用最优性换保守性"，过宽的随机化会让策略学得更慢；
- 课程学习（平地 → 不规则地形，或按能力推进）能显著加速；
- 分布式/大规模并行采样是提升样本效率最直接的手段。

来源：多篇综述（MDPI 2025 legged DRL 系统综述、ScienceDirect 四足综述）；具体条目未取得稳定链接，结论按"综述共识"记录，不作为单篇引用。

### 2.5 行为参数化：与本项目研究直接相关

"Walk These Ways"（CoRL 2022）训练一个**条件策略** π(a | o, b)，b 是少量行为参数（步频、抬腿高度、躯干姿态），单一策略在平坦地面训练后，部署时调 b 就能应对台阶、障碍、推搡。

对本项目的意义：分层架构里 **S1 的输出可以是行为参数而不是直接速度指令**——这给"分层接口选择"提供了一个具体的备选设计，也可能提升 S1 的学习效率（动作空间更小、更结构化）。

来源：[arXiv:2212.03238](https://arxiv.org/abs/2212.03238)

## 3. 瓶颈假设与验证方法

| 假设 | 验证方法 | 预期结果 |
| --- | --- | --- |
| H1：仿真步进是主要开销 | `benchmark_non_rl.py` 测纯环境 FPS，与端到端 61k 对比 | 若纯环境显著高于 61k，则训练/推理有可优化空间 |
| H2：Python 侧适配层有固定开销 | 对比 2048 与 4096 envs 的每样本耗时 | 若 4096 摊薄后每样本更快，则固定开销显著 |
| H3：rollout 太短 | 24 vs 48 steps 各跑 300 iters，比较收敛 | 论文预测 24 偏短；48 可能更稳但更慢 |
| H4：多核空闲 | 2 路并行跑两个种子，比较总墙钟时间 | 预期接近 2× 加速 |

## 4. 可实施实验清单（按预期收益/风险排序）

| 编号 | 实验 | 预期收益 | 风险 | 是否改变实验语义 |
| --- | --- | --- | --- | --- |
| E1 | 清理残留进程 + 训练进程 High 优先级 | **已实测 2.4×** | 无 | 否 |
| E2 | 训练期间禁止 CPU 密集任务（pyright/编译） | 已实测（1.80 → 0.80 s/iter） | 无 | 否 |
| E3 | 2 路并行训练（显存 2×5.2 GB，CPU 仅用 4 核） | 接近 2× | OOM（需先测） | 否（需记录并发度） |
| E4 | `benchmark_non_rl` + `benchmark_rsl_rl` 分解瓶颈 | 定位优化点 | 占 GPU，需等训练结束 | 否 |
| E5 | 评估并行化：20 个种子用 20 个并行环境 | 5–10×（评估） | 需改 `eval_policy.py` | 否（同一批种子） |
| E6 | `num_steps_per_env` 24 → 48（300 iters 对照） | 收敛更稳或更快 | 改变训练动态 | **是**（需在报告中声明） |
| E7 | 4096 envs（仅效率研究，不进正式协议） | 摊薄固定开销 | 显存/协议冲突 | **是** |
| E8 | 检查 VewEnvWrapper 的 dones/obs 处理开销 | 未知（参照 SB3 的 3×） | 改官方代码风险 | 否（若只包一层） |

**优先级**：E1/E2（已完成）→ E3（下一步，收益确定）→ E4（用官方工具定位）→ E5（评估阶段）→ E6/E7（需要单独的消融实验，不能混入正式结果）。

## 5. 纪律：效率优化不能污染研究结论

- **可自由改**：并行度、进程优先级、日志频率、评估并行化。
- **需谨慎**：`num_envs`、`num_steps_per_env`、网络结构——改变训练动态，必须在报告中声明为不同配置，不能与主实验混报。
- **不可改**：评估种子集 E={0..19}、指标定义、验收阈值（协议冻结）。
- 任何效率实验都单独记录 `run_config.json`，并注明"效率研究，不进入主结果表"。

## 6. 下一步

1. seed 2 训练结束后，先跑 `benchmark_non_rl.py` 与 `benchmark_rsl_rl.py`，拿到环境步进 vs 端到端的分解（E4）。
2. 用 2 路并行跑后续 A/B 训练（E3），先做一次显存与吞吐验证。
3. 评估脚本改为 20 环境并行（E5），缩短验证周期。
4. 若 A/B 训练仍偏慢，再考虑 E6（rollout 长度）的消融。

## 7. 参考文献

- Rudin, Hoeller, Reist, Hutter. *Learning to Walk in Minutes Using Massively Parallel Deep Reinforcement Learning.* CoRL 2021. [arXiv:2109.11978](https://arxiv.org/abs/2109.11978) · [PMLR](https://proceedings.mlr.press/v164/rudin22a.html)
- Margolis, Agrawal. *Walk These Ways: Tuning Robot Control for Generalization with Multiplicity of Behavior.* CoRL 2022. [arXiv:2212.03238](https://arxiv.org/abs/2212.03238)
- NVIDIA. *Isaac Lab Performance Benchmarks.* [v2.3.2 docs](https://isaac-sim.github.io/IsaacLab/v2.3.2/source/overview/reinforcement-learning/performance_benchmarks.html) · [Run benchmarks](https://isaac-sim.github.io/IsaacLab/main/source/developer-tools/benchmarking/run_benchmarks.html)
- 2025 legged locomotion DRL 系统综述（MDPI）：奖励塑形/训练稳定性/DR 与课程的取舍
