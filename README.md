# Unitree Go2w 分层导航强化学习（MoRA-inspired）

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.12-3776AB?logo=python&logoColor=white)](requirements.txt)
[![MuJoCo](https://img.shields.io/badge/MuJoCo-3.11.0-0B7285)](https://mujoco.org/)
[![Tests](https://img.shields.io/badge/tests-197-2EA44F)](tests/)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

> **公开作品集版本**：以 Unitree Go2w 轮腿机器人和 MuJoCo 仿真为试验台，研究“任务决策—导航策略—底层控制”的分层闭环。仓库保留可复现源码、精选实验资产、失败记录、验收报告和演示，不包含持续训练服务、完整历史 `rl/runs/` 或任何设备凭据。

**一句话成果**：在单层 PPO、BC、DAgger 和早期课程学习多轮失败后，通过动作空间降维、自然状态分布课程、阶段专用 BC 与精确停止奖励，将单段弯道、10 m 两段导航和岔路口决策任务分别推进到本仓库验收集上的 100% 成功率。

[作品集讲解](docs/PORTFOLIO.md) · [架构设计](docs/ARCHITECTURE.md) · [实验与失败记录](docs/EXPERIMENTS.md) · [复现指南](docs/USAGE.md)

## 30 秒项目卡

| 项目维度 | 内容 |
| --- | --- |
| 研究问题 | 单层策略在长程、分阶段导航中容易站桩、过冲或丢失任务进度 |
| 我的工作 | 环境与任务设计、System 0/1/2 分层实现、PPO/BC/DAgger/课程实验、奖励消融、评估与工程工具链 |
| 核心方案 | System 2 规则决策 + System 1 高层 PPO + System 0 前视点/差速控制 |
| 关键改进 | 6 维底层动作降为 2 维导航动作；`speed_scale≥0.9`；阶段专用 BC；精确停靠闭环 |
| 代表结果 | 弯道 20/20；10 m A→B 导航 100%；岔路口 A/B 共 40/40；正式验收均 0 摔倒 |
| 技术栈 | Python、MuJoCo、Gymnasium、Stable-Baselines3、PyTorch、NumPy |
| 工程质量 | 197 项单元测试、Pyright 0 errors、无头渲染、训练监控、自动报告与视频 |
| 项目边界 | 当前结论来自 MuJoCo 与仓库内验收集，不代表真实机器人部署或开放世界泛化 |

## 核心演示

| 单段弯道 | 10 m 两段导航 A→B | 岔路口目标决策 |
| --- | --- | --- |
| ![弯道分层导航](media/rl_traverse_curve_high_level_seed01.gif) | ![多段精确停止](media/rl_traverse_curve_multi_segment_seed00_v2.gif) | ![岔路口分支决策](media/rl_traverse_curve_junction_seed00.gif) |
| 20/20，平均 1.355 m | A 点停止 100%，平均 10.73 m | A/B 共 40/40，分支正确 100% |

MP4 高清版本与其他对照实验位于 [`media/`](media/)。

## 问题与解法

### 为什么单层 PPO 不够

初始方案直接学习“61 维状态 → 6 维底层动作”。在弯道与长程任务中，策略反复落入两个局部最优：

- **站桩**：允许低速或停车时，不移动比承担摔倒风险更容易获得稳定回报；
- **只顾眼前**：策略缺少子目标、进度与停止状态，能够短程前进，却无法完成 A→B 的阶段切换。

项目没有隐藏这些失败，而是把它们保留为对照证据，并通过单变量消融逐步定位根因。

### 分层闭环

```text
System 2  任务/分支决策（规则实现，锁定 A→左 / B→右）
    │ target / branch / progress
System 1  高层 PPO（目标与进度观测 → [speed_scale, turn_adjust]）
    │ 2 维导航指令
System 0  前视点跟踪 + 差速转向（脚本/PD 控制器 → 6 维底层动作）
    │
MuJoCo Go2w 仿真与任务环境
```

- **System 0 / 反射执行层**：稳定完成前视点跟踪和差速转向，同时生成 BC 演示数据；
- **System 1 / 技能层**：只学习速度缩放与转向修正，维护目标距离、航向误差和停靠进度；
- **System 2 / 任务层**：在岔路口按目标选择分支并写回观测，让导航策略执行已锁定的意图。

这里的 **MoRA-inspired** 表示对公开三层思想的独立工程化映射，不是官方 MoRA 模型的复现；System 2 目前是可审计的规则模块，而不是 VLM/LLM。实现细节见 [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)。

## 可核验结果

| 任务 | 评估规模 | 成功率 | 关键指标 | 证据 |
| --- | ---: | ---: | --- | --- |
| 单段弯道（分层控制） | 20 episodes | 100% | 平均 1.355 m，0 摔倒 | [seed01 报告](reports/traverse_curve_high_level/seed01/report.md) |
| 随机目标 B+ | 20 episodes | 90% | 平均 0.834 m | [B+ 报告](reports/traverse_curve_high_level_bplus/seed00/report.md) |
| 无目标输入消融 | 20 episodes | 100% | 平均 1.352 m | [no-goal 报告](reports/traverse_curve_high_level_bplus_no_goal/seed00/report.md) |
| 10 m 两段导航 A→B | 仓库正式验收 | 100% | A 点停止 100%，平均 15.2 s，0 摔倒 | [multi-segment v2 报告](reports/traverse_curve_multi_segment/seed00_v2/report.md) |
| 岔路口 A/B 决策 | 40 episodes | 100% | 分支正确 100%，平均 7.0 s，0 摔倒 | [junction 报告](reports/traverse_curve_junction/seed00/report.md) |

这些数字描述的是**给定仿真配置和仓库验收协议**，不是跨场景 SOTA 声明。完整成功/失败矩阵见 [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md)。

## 四个最重要的实验发现

1. **动作空间比继续堆训练步数更关键**：仅将 `speed_scale` 下界从 0.5 调到 0.9，单段弯道由站桩 0% 变为 100%。
2. **课程中间态必须匹配自然状态分布**：junction 人工中间态连续失败，而从正常起点训练完整任务成功，说明“更简单的课程”不一定更有效。
3. **Goal-Conditioned 的价值依赖任务复杂度**：固定终点弯道中，有/无目标输入均能达标；到多段和岔路任务后，目标与进度才成为必要状态。
4. **精确停止必须进入奖励与终止条件**：将 docking 奖励提高、加入过冲惩罚并在 B 点达标时终止后，多段任务从 0% 提升到 100%。

![动作空间消融](docs/images/curve_high_level_ablation.png)

## 工程闭环

- 环境、控制器、训练、评估、报告和演示脚本均在仓库内；
- 197 项单元测试覆盖环境、分层包装器、课程、BC/DAgger、监控和 Web 面板；
- Pyright 静态检查为 0 errors；
- MuJoCo 支持 EGL 无头渲染，CPU 环境可完成 smoke 训练；
- 训练产物默认与源码隔离，公开仓库只保留精选证据资产；
- 失败实验与修复路径一并归档，避免只展示“最好的一次”。

## 快速开始

推荐 Linux 或 WSL2，Python 3.10/3.12：

```bash
git clone https://github.com/Anhao1314/go2w-MoRA-navigation.git
cd go2w-MoRA-navigation

python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 运行测试
python -m unittest discover -s tests -p 'test_*.py'

# 生成 System 0 弯道演示轨迹
python scripts/gen_demo_trajectory.py --seed 0

# 无头 Linux 录制演示
MUJOCO_GL=egl bash scripts/make_demo.sh traverse_curve
```

训练入口：

```bash
# 单任务 PPO
python rl/train.py --task traverse_curve --seed 0 --total-steps 2000000

# 分层/课程训练
python scripts/train_high_level_curve.py
python scripts/train_multi_segment_curriculum.py
python scripts/train_junction_curriculum.py

# 本地监控面板；不要直接暴露到公网
python rl/webpanel.py --host 127.0.0.1 --port 8787
```

长训练前请先按 [`docs/USAGE.md`](docs/USAGE.md) 完成 smoke、环境和磁盘检查。

## 仓库结构

```text
├── rl/                         # 环境、训练、控制器、评估与本地监控
│   ├── go2w_env.py             # Gymnasium 环境与任务逻辑
│   ├── low_level_controller.py # System 0
│   ├── high_level_env_wrapper.py # System 1/目标与进度包装
│   └── train.py                # SB3 PPO 入口
├── scripts/                    # BC、DAgger、课程训练、演示和报告脚本
├── tests/                      # 197 项单元测试
├── reports/                    # 精选正式验收报告
├── data/demo_trajectories/     # 精选演示数据、模型与消融记录
├── media/                      # GIF / MP4 演示
├── models/go2w/                # Unitree Go2w 模型与第三方许可
└── docs/                       # 作品集、架构、实验和复现文档
```

## 项目边界与下一阶段

当前仓库是**机器人强化学习研究与工程作品集**，不是实机控制产品，也不是量化交易系统。它为后续量化研究复用了同一套方法论：

```text
状态/观测 → 分层决策 → 动作执行 → 结果反馈 → 可追溯评估 → 下一轮迭代
```

下一阶段计划把训练/评估结果标准化为可增量读取的数据契约，再建设 Windows 端数据仓库、特征计算、回测与风险控制。任何交易信号、回测结论或实盘接入都必须在独立模块中验证，不能由本仓库的机器人成功率直接推导。

## 文档导航

- [`docs/PORTFOLIO.md`](docs/PORTFOLIO.md) — 项目贡献、面试讲解顺序、能力映射与限制
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — 三层架构、观测演进、奖励与实现边界
- [`docs/EXPERIMENTS.md`](docs/EXPERIMENTS.md) — 成功/失败实验、证据入口与方法论结论
- [`docs/USAGE.md`](docs/USAGE.md) — 环境、演示、训练、评估和常见问题
- [`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md) — 第三方模型与算法许可

## 许可证

- 本项目自研代码采用 [MIT License](LICENSE)。
- Unitree Go2w 模型采用 BSD-3-Clause，见 [`models/go2w/LICENSE`](models/go2w/LICENSE)。
- 其他第三方说明见 [`docs/THIRD_PARTY_NOTICES.md`](docs/THIRD_PARTY_NOTICES.md)。
