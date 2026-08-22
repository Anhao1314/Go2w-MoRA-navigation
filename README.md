# Unitree Go2w 分层导航强化学习（MoRA-inspired）

> MuJoCo 轮腿机器人作品集：从"单段弯道"到"10m 两段导航"再到"岔路口决策"。
> 核心主线：用三层架构（System 2 决策 / System 1 导航 / System 0 控制）解决
> 单层 PPO 学不会的长程导航问题。

## English Abstract

This project builds a **hierarchical navigation agent** for the Unitree Go2w
wheel-legged robot in MuJoCo, inspired by the MoRA Agentic-Native architecture.
A low-level scripted controller (System 0) tracks the path centerline; a
high-level PPO policy (System 1) outputs 2-D navigation commands; and a rule
based System 2 makes branch decisions at junctions. Pure single-layer PPO/BC/DAgger
failed on the curve task (≤0.7 m), while the hierarchical agent reached **100%
success on a 10 m two-subgoal navigation task with precise stopping** and
**100% success on a junction decision task**, with 0 falls in formal evaluation.

## 核心亮点

- **三层控制架构**：System 0 前视点差速控制器 → System 1 高层 PPO（2 维动作）→
  System 2 规则决策（A→左 / B→右）。
- **动作空间降维**：6 维力矩 → `[speed_scale, turn_adjust]`，并禁止停车
  （`speed_scale≥0.9`），消除"站桩"局部最优。
- **BC 预热 + 三阶段课程**：脚本演示监督 actor → 段1/段2/完整任务逐级微调，
  norm 冻结 + 学习率 warmup 防止 negative transfer。
- **精确停止**：位置 ±0.15m + 停留 0.3s + 速度 <0.2m/s，A/B 子目标均达标。
- **工程闭环**：197 项单元测试、Pyright 0 errors、训练守护、实时 Web 面板、
  自动报告与演示视频。

## 实验结果（reports/ 实测数据）

| 任务 | 方法 | 成功率 | 关键指标 |
|---|---|---|---|
| 单段弯道（MoRA 分层） | System 0 + System 1 PPO | 100% | 1.355 m，0 摔倒 |
| 单段弯道 B+ | 随机目标 + 轻量 HER | 90% | 0.834 m |
| 单段弯道 B+ 无目标 | 消融对照组 | 100% | 1.352 m |
| **10m 两段导航 A→B** | 课程学习 + 精确停止 | **100%** | A 停止 100%，15.2 s，0 摔倒 |
| **岔路口 A→B 决策** | System 2 规则 + System 1 | **100%** | 分支正确 100%，7.0 s，0 摔倒 |

对照组：单层 PPO（v1~v5）、BC、DAgger、课程学习早期版本全部失败（≤0.7 m），
详见 [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)。

## 架构

```text
System 2 规则决策（岔路口锁定分支）
        │ target/branch 观测
System 1 高层 PPO（61+7 维观测 → [speed_scale, turn_adjust]）
        │ 2 维导航指令
System 0 前视点跟踪 + 差速转向（→ 6 维力矩）
        │
     MuJoCo Go2w 仿真
```

完整设计见 [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)。

## 🎬 演示与成果

### 核心任务演示

| 任务 | 演示 | 关键指标 |
|---|---|---|
| 弯道导航（分层控制） | ![curve](media/rl_traverse_curve_high_level_seed01.gif) | 1.355m，50k即100%，0摔倒 |
| 10m长程导航A→B | ![multi-seg](media/rl_traverse_curve_multi_segment_seed00_v2.gif) | 10.73m，A停止100%，15.2s |
| 岔路口决策 | ![junction](media/rl_traverse_curve_junction_seed00.gif) | 40/40，决策100%，7.0s |
| 随机目标泛化 | ![bplus](media/rl_traverse_curve_high_level_bplus_seed00.gif) | 90%成功率，0.834m |

### 失败→成功：12轮探索的关键转折

单层方法（纯PPO v1-v5、BC初始化、DAgger 5轮、固定课程4阶段）在弯道任务上全部失败（≤0.7m或站桩不动）。
通过消融实验定位到**动作空间设计**是核心瓶颈：给策略"停车"的合法出口（speed_scale≥0.5）会导致站桩局部最优；禁止停车（speed_scale≥0.9）后50k即100%通过。

![ablation](docs/images/curve_high_level_ablation.png)

*单一变量消融：speed_scale下界0.5（红，站桩0m）vs 0.9（绿，50k即100%）*

### 课程学习的意外发现

junction 的三阶段课程出现了"stage2失败但stage3成功"的现象——stage2从人工中间态（岔路口yaw=0、零速度）出发全部失败，但stage3从正常起点出发的完整任务100%成功。根因：人工中间态的起始分布与策略自然到达该状态时的分布不匹配。结论：课程学习的中间态起始分布必须与自然状态匹配，否则不如直接学完整任务。

multi_segment 的 v2 重训（停靠奖励+100、过冲惩罚、B点到达即终止、stage2单独BC预热）则让三个阶段全部100%通过，验证了奖励设计与阶段专用BC的价值。

![multi-curriculum](docs/images/multi_segment_curriculum.png)
![junction-curriculum](docs/images/junction_curriculum.png)

*左：multi_segment v2 三阶段全部成功（P0/P1 修复后）；右：junction stage2 人工起始失败 vs stage3 完整任务成功。*

### 消融实验汇总

| 实验 | 变量 | 结果 | 结论 |
|---|---|---|---|
| 动作空间下界 | speed_scale≥0.5 vs ≥0.9 | 0% vs 100% | 禁止停车是分层控制关键 |
| Goal-Conditioned | 有目标输入 vs 无 | 90% vs 100%（均达标） | 简单弯道下GC无独立增益 |
| 课程中间态（junction） | 人工起始 vs 正常起始 | stage2 0% vs stage3 100% | 中间态分布必须匹配自然状态 |
| 精确停止奖励（multi_segment） | docking +30 vs +100+过冲惩罚 | 0% vs 100% | 奖励设计决定长程任务成败 |

## 快速开始

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

## 目录结构

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
└── docs/                # 架构、实验索引、第三方许可
```

## 实验演进（失败 → 根因 → 修复）

1. 单层 PPO / BC / DAgger：弯道全部 ≤0.7 m → 证明需要目标与进度概念。
2. 分层第一步：动作允许停车 → 学到站桩 0 m → 禁止停车后单段 100%。
3. 多段 A→B 首版：冲过 B 点不停车 → 停靠奖励、过冲惩罚、B 点到达即终止。
4. 岔路口：System 2 规则决策 + System 1 分支导航 → 40-episode 验收 100%。

## 许可证

- 本项目自身代码：MIT（见 [LICENSE](LICENSE)）。
- 宇树 Unitree Go2w 模型：BSD-3-Clause（`models/go2w/LICENSE`）。
- 高度场地形算法参考 westonrobot/unitree_mujoco（BSD-3），见
  [docs/THIRD_PARTY_NOTICES.md](docs/THIRD_PARTY_NOTICES.md)。
