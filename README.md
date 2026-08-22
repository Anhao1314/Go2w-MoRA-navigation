# Unitree Go2w 分层导航强化学习（MoRA-inspired）

> **项目定位**：以四足机器人导航为试验台，研究 MoRA 分层智能架构（System 0/1/2）与量化决策闭环方法。历经纯 PPO/BC/DAgger/课程学习共 12 轮失败，通过消融实验定位动作空间设计为核心瓶颈，最终在弯道/10m 长程导航/岔路口决策任务上实现 100% 通过率。
> 从"单段弯道"到"10m 两段导航"再到"岔路口决策"。
> 核心主线：用三层架构（System 2 决策 / System 1 导航 / System 0 控制）解决单层 PPO 学不会的长程导航问题。

## English Abstract

This project is a research vehicle for hierarchical agent architectures:
we use Unitree Go2w navigation in MuJoCo as a testbed to study the MoRA
Agentic-Native stack (System 2 decision / System 1 policy / System 0 control)
and an experiment-driven quantitative decision loop. After 12 failed rounds of
single-layer methods (pure PPO v1-v5, BC fine-tuning, DAgger, fixed curriculum),
ablation studies isolated **action-space design** as the core bottleneck.
The hierarchical agent then reached **100% success on a 10 m two-subgoal
navigation task with precise stopping** and **100% success on a junction
decision task**, with 0 falls in formal evaluation.

## 🔬 核心研究发现

项目先经历 12 轮失败（纯 PPO v1~v5、BC 微调、DAgger 5 轮、固定课程 4 阶段），
再用消融实验逐一定位根因。以下是四条最有价值的方法论发现，全部来自
`reports/` 与 `rl/runs/*/eval_log.csv` 的真实数据。

### 1. 动作空间设计是分层控制的核心瓶颈

单一变量消融：`speed_scale` 下界 0.5 → 0.9，弯道成功率从 **0% → 100%**
（seed00 站桩 0m，seed01 在 50k 评估点即 100%，至 150k 三个评估点全部 100%）。
给策略"停车"的合法出口会导致站桩局部最优；禁止停车后，System 0 教师控制器
本身能走通的路线，高层策略也能学会。

### 2. 课程学习的中间态必须匹配自然状态分布

junction 三阶段课程：stage1（正常起点）250k 达标；stage2（人工中间态：
岔路口 yaw=0、零速度出发）10 个 eval 点全 0%；stage3（完整任务正常起点）
约 1.008M 即 100%。**更难的完整任务成功了，更简单的中间态反而失败**——
人工中间态与策略自然到达该状态时的分布不匹配。结论：课程学习的中间态起始
分布必须与自然状态匹配，否则不如直接学完整任务。

### 3. Goal-Conditioned 在简单任务上无独立价值

B/B+ 随机目标消融（`traverse_curve_high_level_bplus` vs `..._no_goal`）：
正式验收有目标输入版 90%/0.834m，无目标版 100%/1.3522m。训练曲线显示两者
都会在 100k 附近出现站桩回退；无目标版 300k 后固定/随机成功率稳定 100%，
有目标版 250k 后随机成功率在 80%~100% 波动。结论：在固定终点单段弯道上，
目标输入未被策略真正利用，GC 的价值依赖任务复杂度（多段/岔路口任务才体现）。

### 4. 精确停止奖励设计决定长程任务成败

multi_segment v1（docking 奖励 +30）：0% 成功，冲过 B 点不停车；
multi_segment v2（docking +100、过冲惩罚 -1×距离、B 点到达即终止、
stage2 单独 BC 预热）：**100% 成功，A 点精确停止达标 100%**。
奖励结构与初始化分布的修复，使长程导航从完全失败到完全成功。

### 亮点速览

- **三层控制架构**：System 0 前视点差速控制器 → System 1 高层 PPO（2 维动作）→
  System 2 规则决策（A→左 / B→右）。
- **动作空间降维**：6 维力矩 → `[speed_scale, turn_adjust]`，并禁止停车
  （`speed_scale≥0.9`），消除"站桩"局部最优。
- **BC 预热 + 三阶段课程**：脚本演示监督 actor → 段1/段2/完整任务逐级微调，
  norm 冻结 + 学习率 warmup 防止 negative transfer。
- **精确停止**：位置 ±0.15m + 停留 0.3s + 速度 <0.2m/s，A/B 子目标均达标。
- **工程闭环**：197 项单元测试、Pyright 0 errors、训练守护、实时 Web 面板、
  自动报告与演示视频。

### 消融实验汇总

| 实验 | 变量 | 结果 | 结论 |
|---|---|---|---|
| 动作空间下界 | speed_scale≥0.5 vs ≥0.9 | 0% vs 100% | 禁止停车是分层控制关键 |
| Goal-Conditioned | 有目标输入 vs 无 | 90% vs 100%（均达标） | 简单弯道下GC无独立增益 |
| 课程中间态（junction） | 人工起始 vs 正常起始 | stage2 0% vs stage3 100% | 中间态分布必须匹配自然状态 |
| 精确停止奖励（multi_segment） | docking +30 vs +100+过冲惩罚 | 0% vs 100% | 奖励设计决定长程任务成败 |

### 实验演进（失败 → 根因 → 修复）

1. 单层 PPO / BC / DAgger：弯道全部 ≤0.7 m → 证明需要目标与进度概念。
2. 分层第一步：动作允许停车 → 学到站桩 0 m → 禁止停车后单段 100%。
3. 多段 A→B 首版：冲过 B 点不停车 → 停靠奖励、过冲惩罚、B 点到达即终止。
4. 岔路口：System 2 规则决策 + System 1 分支导航 → 40-episode 验收 100%。

## 🏗️ MoRA 三层架构

```text
System 2 规则决策（岔路口锁定分支）
        │ target/branch 观测
System 1 高层 PPO（61+7 维观测 → [speed_scale, turn_adjust]）
        │ 2 维导航指令
System 0 前视点跟踪 + 差速转向（→ 6 维力矩）
        │
     MuJoCo Go2w 仿真
```

- **System 0（反射层）**：脚本控制器，前视点跟踪 + 差速转向，生成演示轨迹，
  稳定可靠，是 BC 标签与低层执行来源。
- **System 1（技能层）**：Agentic-Native 高层 PPO，2 维动作
  `[speed_scale, turn_adjust]`，注入目标/记忆/进度（观测 61 → 67 → 68 维演进）。
- **System 2（Agent 块）**：离散决策，岔路口选左/右，目标条件驱动，
  决策锁定并写回观测，供 System 1 使用。

三层架构首次在岔路口任务上完整验证，System 2 决策正确率 100%。
完整设计见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 📊 训练曲线与消融验证

### 动作空间消融：0% → 100% 的单一变量

![ablation](docs/images/curve_high_level_ablation.png)

*speed_scale 下界 0.5（红，站桩 0m）vs 0.9（绿，50k 即 100%）*

### 课程学习的意外发现：stage2 失败但 stage3 成功

![multi-curriculum](docs/images/multi_segment_curriculum.png)
![junction-curriculum](docs/images/junction_curriculum.png)

*multi_segment v2 三阶段全部成功（P0/P1 修复后）；junction stage2 人工起始失败 vs stage3 完整任务成功。*

junction 的 stage2 从人工中间态（岔路口 yaw=0、零速度）出发全部失败，但
stage3 从正常起点出发的完整任务 100% 成功，根因是人工中间态的起始分布与
策略自然到达该状态时的分布不匹配。multi_segment 的 v2 重训（停靠奖励 +100、
过冲惩罚、B 点到达即终止、stage2 单独 BC 预热）则让三个阶段全部 100% 通过，
验证了奖励设计与阶段专用 BC 的价值。

## 🎬 核心任务演示（实验结果可视化）

| 任务 | 演示 | 关键指标 |
|---|---|---|
| 弯道导航（分层控制） | ![curve](media/rl_traverse_curve_high_level_seed01.gif) | 1.355m，50k即100%，0摔倒 |
| 10m长程导航A→B | ![multi-seg](media/rl_traverse_curve_multi_segment_seed00_v2.gif) | 10.73m，A停止100%，15.2s |
| 岔路口决策 | ![junction](media/rl_traverse_curve_junction_seed00.gif) | 40/40，决策100%，7.0s |
| 随机目标泛化 | ![bplus](media/rl_traverse_curve_high_level_bplus_seed00.gif) | 90%成功率，0.834m |

高清版见 `media/`（`*.mp4`）。

## 📈 实验结果总表（含失败实验）

以下列出所有实验，包括失败实验。失败实验的根因分析见上方"核心研究发现"章节。

| 任务 | 方法 | 成功率 | 关键指标 |
|---|---|---|---|
| 单段弯道（MoRA 分层） | System 0 + System 1 PPO | 100% | 1.355 m，0 摔倒 |
| 单段弯道 B+ | 随机目标 + 轻量 HER | 90% | 0.834 m |
| 单段弯道 B+ 无目标 | 消融对照组 | 100% | 1.352 m |
| **10m 两段导航 A→B** | 课程学习 + 精确停止 | **100%** | A 停止 100%，15.2 s，0 摔倒 |
| **岔路口 A→B 决策** | System 2 规则 + System 1 | **100%** | 分支正确 100%，7.0 s，0 摔倒 |

对照组：单层 PPO（v1~v5）、BC、DAgger、课程学习早期版本全部失败（≤0.7 m），
详见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 🚀 快速开始

```bash
# 1. 环境
conda create -n mujoco_env python=3.10
conda activate mujoco_env
pip install -r requirements.txt

# 2. 生成弯道演示轨迹（System 0 脚本控制器）
python scripts/gen_demo_trajectory.py --seed 0

# 3. 录制演示视频（无显示环境加 MUJOCO_GL=egl）
bash scripts/make_demo.sh traverse_curve

# 4. 全量测试
python -m unittest discover -s tests -p 'test_*.py'
```

训练入口：

```bash
# 单任务 PPO（traverse_curve 等）
python rl/train.py --task traverse_curve --seed 0 --total-steps 2000000

# 高层分层训练（单段/多段/岔路口课程学习）
python scripts/train_high_level_curve.py
python scripts/train_multi_segment_curriculum.py
python scripts/train_junction_curriculum.py

# 本地 Web 面板（训练进度 / 资源 / 报告）
python rl/webpanel.py --host 127.0.0.1 --port 8787
```

完整复现步骤见 [docs/USAGE.md](docs/USAGE.md)。

## 📁 项目结构

```text
├── rl/                  # 环境、PPO 训练、低层控制器、评估、Web 面板
│   ├── go2w_env.py      # Gymnasium 环境（traverse/多段/岔路口）
│   ├── low_level_controller.py   # System 0 脚本控制器
│   ├── high_level_env_wrapper.py # System 1 高层包装器（2 维动作）
│   └── train.py         # SB3 PPO 训练入口
├── scripts/             # 演示、BC 预热、课程学习、报告脚本
├── tests/               # 197 项单元测试
├── models/go2w/         # 宇树官方 Go2w 模型（BSD-3）
├── data/demo_trajectories/  # 演示轨迹、BC 模型、实验报告
├── reports/             # 各任务验收报告与指标
├── media/               # GIF / MP4 演示
└── docs/                # 架构、实验索引、使用指南、第三方许可
```

## ✅ 测试与质量

- **197 项单元测试**：`tests/` 覆盖环境、低层控制器、高层包装器、BC/DAgger、
  课程学习、Web 面板与安全路径校验（`python -m unittest discover -s tests`）。
- **Pyright 0 errors**：全仓库静态类型检查通过。
- **工程闭环**：训练守护（崩溃续训）、实时 Web 面板、自动验收报告、
  自动演示视频、量化数据入库。

## 📝 研究文档

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — MoRA 三层架构、观测演进、奖励设计
- [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) — 全部实验结论与失败教训
- [docs/USAGE.md](docs/USAGE.md) — 环境搭建/复现/评估/自定义任务指南
- [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md) — 第三方许可
- `data/demo_trajectories/` 下另有 10 份实验报告（BC 消融、B+ 评估、
  多段 P0 验证、DAgger 训练等），索引见 EXPERIMENTS.md

## 📄 许可证与致谢

- 本项目自身代码：MIT（见 [LICENSE](LICENSE)）。
- 宇树 Unitree Go2w 模型：BSD-3-Clause（`models/go2w/LICENSE`）。
- 高度场地形算法参考 westonrobot/unitree_mujoco（BSD-3），见
  [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md)。
