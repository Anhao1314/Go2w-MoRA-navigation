# MoRA-Train：面向具身智能体的分层训练系统

### 第一个 adapter：Unitree Go2W 轮式四足长程导航

*MoRA-Train: A Hierarchical Training System for Embodied Agents · First adapter: Unitree Go2W*

- **文档性质：** 系统工程与研究记录（非同行评议论文）
- **项目主体：** MoRA-Train（MoRA-inspired 分层具身训练系统）
- **本仓库角色：** 第一个 Task Adapter / 实验床 + S0 C++ 控制核心
- **训练侧：** Windows（Go2W 全部训练与评估）
- **量化侧：** RL-Sentinel（Mac，时间正确的训练可靠性分析）
- **适用范围：** 仿真-only；非真实机器人部署；不声称复现官方 MoRA

[English](README.en.md) · [架构与接口](docs/ARCHITECTURE.md) · [实验报告索引](docs/EXPERIMENTS.md) · [复现指南](docs/USAGE.md) · [完整性升级说明](docs/INTEGRITY_UPGRADE.md)

## 0. 项目定位

MoRA-Train 要解决的不是“让一只机器狗走完一条路”，而是：

> 如何训练一个由 S2 规划、S1 目标条件执行（多粒度记忆 + 进度闭环）、S0 可验证控制组成的具身智能体，并在多个任务上训练、评估和量化，而不是只针对单一平台调参。

Go2W 是 MoRA-Train 的**第一个 task adapter 和实验床**，用来验证：

- 层次接口是否可组合：S2 规则决策 → S1 学习策略 → S0 C++ 控制；
- S0 是否数值可信：FK/IK/Jacobian、质心动力学、MPC 与 MuJoCo 一致；
- S1 是否能完成多段、岔路、目标点等长程任务；
- 训练过程是否能被 RL-Sentinel 以时间正确的方式量化。

| 层 | 职责 | 本仓库现状 |
| --- | --- | --- |
| S2 | 任务分解、分支/技能选择 | Go2W adapter 内用规则实现 |
| S1 | 目标条件策略 | PPO / BC / DAgger / 课程学习 |
| Memory | 短期/情节/任务记忆 | `mora/` 骨架已实现；神经 Memory Token 待做 |
| Progress | 进度估计、停滞检测、重规划 | `mora/` 启发式已实现；学习型 progress head 待做 |
| S0 | 运动控制、MPC、安全状态机 | P0 已交付并做数值验证 |
| Observability | 事件流、as-of 早停分析 | `EventWriter` + SB3 callback 已接入 multi-segment；RL-Sentinel ingester 待接 |

系统层接口与当前能力见 [MoRA-Train 系统层](docs/MORA_TRAIN.md)。本仓库的结果只说明“第一个 adapter 在当前协议下可行”，**不等于 MoRA-Train 整体已被验证**，也不等于官方 MoRA 被复现。

## 摘要

**背景：** MoRA-Train 需要一个可复现的具身任务来验证分层训练接口。Go2W 轮式四足在已知路径仿真中提供了长程导航、岔路决策和底层控制验证。**目的：** 在本仓库中交付第一个 adapter，并检验“规则 S2 + 学习型 S1 + 可验证 S0”的组合能否完成指定任务。**方法：** 将规则目标/分支决策、PPO 高层导航指令、脚本路径跟踪/C++ S0 执行组合，记录 PPO、BC、DAgger、课程学习和探索性对照。**结果：** 在各自归档的固定仿真协议下，弯道、多段 A→B、岔路分别记录 **20/20、20/20、40/40** 成功；C++ S0 与 scripted S0 在 multi-segment 上逐步一致，10/10 通过。**结论：** 这些证据支持第一个 adapter 的工程可行性，但不单独归因 PPO/BC 的收益，也不证明 MoRA-Train 的记忆/进度模块、未知场景泛化或真实机器人可靠性。

**关键词：** MoRA-inspired；分层具身智能体；机器人导航；强化学习；PPO；MuJoCo；训练可靠性

## 1. 第一个 adapter：研究问题

以下 RQ 是 **Go2W adapter 层面**的问题，不是 MoRA-Train 的全部研究问题；系统级的记忆、进度闭环与跨任务问题在后续迭代中回答。

机器人在弯道和连续任务中运动，既要保持自身稳定，又要沿路线调整方向、控制速度并在目标点停下。如果把这些责任全部交给一个底层强化学习策略，学习过程就可能陷入“不动”“乱动”或“无法稳定完成目标”的局部方案。本项目的早期实验确实记录了类似结果，参见[负结果总览](docs/EXPERIMENTS.md)。

这项研究不尝试发明新的 PPO 算法，而是提出一个**系统工程假设**：把已知的路径跟踪与制动能力留给传统控制器，让学习策略只负责少量导航调整，并向它提供明确的任务目标和进度。

具体研究三个问题：

- **RQ1：** 规则决策、PPO 和传统控制构成的分层系统，能否完成指定的弯道、多段停靠和岔路导航任务？
- **RQ2：** 原地停滞、终点过冲与阶段训练失败，提示了哪些动作接口、奖励设计和状态分布问题？
- **RQ3：** 现有实验能否将 PPO、BC、DAgger 和脚本控制器的**独立贡献**可靠地区分开？

**命名与范围边界：** `MoRA-inspired` 指独立借鉴分层思想的工程实现，并非官方 MoRA 模型或训练流程复现。当前 System 2 是规则模块，没有实现学习型 VLM/VLA、语义规划或开放场景导航；仓库也没有真实 Go2W 的部署结果。

## 2. Go2W adapter 的方法

以下方法是 Go2W adapter 的具体实现，不是 MoRA-Train 通用训练系统的完整设计；系统级 Task API、记忆与进度模块在后续迭代中抽象。

### 2.1 仿真环境与任务

本研究使用 [`Go2wEnv`](rl/go2w_env.py) 封装 MuJoCo 机器人模型、Gymnasium 状态/动作接口、奖励、终止条件和任务进度。主要任务是单段弯道、带 A 点停靠的多段 A→B 导航，以及已知目标 A/B 的岔路导航。

当前代码中物理步长为 **0.002 秒（500 Hz）**；每 5 个物理步更新一次策略指令，即 **100 Hz**。这描述的是仿真设置，而不是实机 SDK 控制频率。

底层穿越环境的动作接口是 6 维：**前四维是四轮力矩相关命令，后两维是前后腿组姿态命令**。原始观测包含机器人姿态、角速度、轮速、腿关节状态、历史动作以及导航状态等。实现参见[环境代码](rl/go2w_env.py)。

### 2.2 分层控制结构

```text
已知目标 / 任务状态
  ↓
System 2：规则目标与岔路选择（A→左、B→右）
  ↓
System 1：PPO MLP → [speed_scale, turn_adjust]
  ↓
System 0：前视路径跟踪 + 差速转向 + 脚本停靠 / 腿部 PD 控制
  ↓
MuJoCo：执行动作 → 返回状态、奖励、任务结果
  └──────────────────→ 更新下一步决策
```

| 层级 | 具体实现 | 主要职责 | 源代码 |
| --- | --- | --- | --- |
| System 2 | 规则映射与决策区分支锁定 | 将 A/B 任务目标映射到左/右岔路 | [高层包装器](rl/high_level_env_wrapper.py) |
| System 1 | Stable-Baselines3 PPO，MLP `[256, 256]` | 输出 2 维导航指令 | [高层 PPO 训练](scripts/train_high_level_curve.py) |
| System 0 | 已知路径前视点跟踪、差速和脚本/PD 控制 | 将导航指令转换为实际底层动作 | [底层控制器](rl/low_level_controller.py)、[环境](rl/go2w_env.py) |

**观测空间：** 基础为 61 维；附加目标条件后为 63 维，多段任务为 67 维，岔路任务为 68 维。额外信息分别包含目标距离/方向、任务阶段或已选分支状态。

**动作空间：** 高层输出 `[speed_scale, turn_adjust]`，当前范围分别为 **[0.9, 1.5]** 和 **[-0.5, 0.5]**，由底层控制器展开为 6 维执行动作；正常导航的速度倍率下界不同于脚本制动或零动作。

**关键限制：** 岔路的“往左还是往右”由规则决定，实际转向使用已知路径的脚本教师几何；当前岔路分支**未使用高层策略的 `turn_adjust`**，学习策略主要改变速度。多段停靠也具有显式的脚本行为。不能将全部岔路决策、转向和刹车声称为 PPO 自主学会的能力。

### 2.3 策略训练与实验方法

| 方法 | 实际作用 | 对应入口 |
| --- | --- | --- |
| PPO [1] | 最初的单层基线，以及后续分层导航策略学习 | [低层训练](rl/train.py)、[高层训练](scripts/train_high_level_curve.py) |
| BC（行为克隆） | 用脚本教师演示的状态—动作对做监督学习，为部分课程阶段初始化策略 | [BC](scripts/bc_pretrain.py)、[多段 BC](scripts/bc_pretrain_multi_segment.py) |
| DAgger [2] | 在当前策略访问的偏差状态中收集教师纠正，再加入训练集迭代学习 | [DAgger](scripts/dagger_train.py) |
| 课程学习 | 先到 A，再从 A 到 B，最后完成整个 A→B 任务 | [多段课程](scripts/train_multi_segment_curriculum.py)、[岔路课程](scripts/train_junction_curriculum.py) |

奖励函数随任务版本变化，主要涉及前进进度、航向与横向偏差、稳定性、子目标到达和失败惩罚等。不同实验不能简单视为使用同一个 Reward Function；精确逻辑应以[`_get_reward` 实现](rl/go2w_env.py)和对应归档配置为准。

在多段任务中，**A 点停靠**的判定包含位置容差约 0.15 m、速度低于 0.2 m/s 和持续 0.3 秒；但**B 点**的完成逻辑是进入终点范围后结束回合，并没有独立验证与 A 点相同的静止保持条件。这一区别影响对“精确停靠”结果的解释。

### 2.4 评价指标与证据口径

归档报告使用任务成功率、进度/距离指标、回合用时、摔倒次数、A 点停靠和岔路分支正确率等指标。以下全部结果来自历史 `reports/` 和 `data/demo_trajectories/` 文件，**不是本次文档改造重新训练出来的结果**。

需要区分：一次实验内更换 episode seed、独立训练多次以及测试新的场景分布，是三种不同强度的验证。现有多数成功结果只覆盖第一种。

<a id="final-results"></a>
## 3. Go2W adapter 的归档实验结果

| 实验 | 记录结果 | 解释范围 | 证据 |
| --- | --- | --- | --- |
| 早期单层 PPO | 未达标 | 站桩或短程移动（记录至多约 0.7 m） | [实验总览](docs/EXPERIMENTS.md) |
| 分层弯道 `seed00` | **0/20** | 策略选择速度下界 0.5，无法有效前进 | [失败报告](reports/traverse_curve_high_level/seed00/report.md) |
| 分层弯道 `seed01` | **20/20** | 平均报告进度 1.355 m，0 摔倒 | [弯道报告](reports/traverse_curve_high_level/seed01/report.md) |
| 多段 A→B 初版 | **0%** | 过冲目标点，完整任务未达标 | [初版报告](reports/traverse_curve_multi_segment/seed00/report.md) |
| 多段 A→B `v2` | **20/20** | A 点停靠 100%，平均 15.2 秒，0 摔倒 | [v2 报告](reports/traverse_curve_multi_segment/seed00_v2/report.md) |
| 岔路 A/B | **40/40** | 按规则选路正确率 100%，平均 7.0 秒，0 摔倒 | [岔路报告](reports/traverse_curve_junction/seed00/report.md) |
| B+ 随机目标 | **18/20** | 2 次摔倒；只覆盖该随机目标协议 | [B+ 报告](reports/traverse_curve_high_level_bplus/seed00/report.md) |

**报告口径提醒：** 部分“平均距离”来自代码中的进度指标，并非统一定义的物理路程；历史弯道弧长公式还有待版本化修正。因此不能把不同报告的米数直接做物理路程比较，详见[完整性说明](docs/INTEGRITY_UPGRADE.md)。

仿真演示作为补充材料，而非独立证据：[弯道录像](media/rl_traverse_curve_high_level_seed01.mp4) · [多段录像](media/rl_traverse_curve_multi_segment_seed00_v2.mp4) · [岔路录像](media/rl_traverse_curve_junction_seed00.mp4)。它们全部是 **MuJoCo 视频，不是真机录像**。

## 4. 负结果、对照与讨论

### 4.1 BC 与 DAgger 是否改善了训练？

[BC 初始化与随机初始化对照](data/demo_trajectories/bc_ablation_report.md)在归档的 0.5M 步实验中都只有 **0% 成功率**。两组学习率和训练种子不同，因此这只能叫探索性比较，**不能证明 BC 的独立作用，更不能证明 BC 普遍无效**。

[DAgger 纠正实验](data/demo_trajectories/dagger_training_report.md)完成五轮纠正，最好一轮移动指标约 0.915 m，但成功率仍为 **0%**。教师在接近摔倒的状态下可能不能提供有效恢复动作，或纠正数据改变了训练分布；这些是需要再检验的解释，不是已证实的因果结论。

### 4.2 “分层后成功”能证明什么？

早期单层 PPO 训练失败，后来分层弯道 `seed01` 成功，说明**这一版本的完整方案能够完成被测任务**。但架构、动作约束和学习设置一起改变，并未做完全统一条件的多种子对照，因此不能据此量化“分层”这个单独变量的收益。

多段初版完成率为 0%，v2 为 20/20；v2 同时调整了停靠奖励、过冲惩罚、终止逻辑和阶段 BC。结论同样是**整体改造后的结果成立，单项增益尚无独立证据**。

### 4.3 一个必须保留的强基线

[多段导航脚本控制器验证](data/demo_trajectories/multi_segment_validation_report.md)本身已记录 **20/20 成功**。这说明最终系统的导航能力很大程度上依赖已有路径跟踪和停靠机制；不能把最终 20/20 都归功于 PPO。另一方面，脚本基线与学习型方案并非严格同协议、同种子的成对比较，也不能草率得出“PPO 完全没有价值”的结论。

### 4.4 历史 HER 文案不能直接引用为结论

B+ 历史报告的配置显示 HER 关闭，但文本结论出现过 HER 有效的表述。代码中的 future-goal relabeling 加 actor 辅助 MSE 也**不是标准 off-policy HER**。本研究不将这些报告作为标准 HER 收益的证据，详见[实验记录](docs/EXPERIMENTS.md)和[完整性报告](docs/INTEGRITY_UPGRADE.md)。

## 5. 有效性威胁与局限

<a id="limitations"></a>

1. **内部有效性：** 奖励、动作接口、初始化与课程阶段可能同时变化，缺少完备的单因素消融。
2. **统计有效性：** 20/20、40/40 只描述各自归档批次；回合种子不是独立训练种子，历史阶段评价与最终验收种子曾有重叠。
3. **外部有效性：** 路径几何和仿真位姿已知；多段与岔路实验未启用域随机化，尚无未见地形、真实传感输入与 Sim-to-Real 测试。
4. **测量有效性：** 进度指标与实际物理路程并不总等价；A 点停靠和 B 点到达使用不同判据。
5. **结果可复现性：** 成功 PPO 策略的最终 checkpoint、匹配归一化统计和完整训练日志未包含在当前公开快照中；现有代码、精选数据、录像和报告**不等于无需重新训练即可复现成功策略**。
6. **能力边界：** 规则模块、教师转向和部分制动是脚本逻辑；没有学习型 System 2、开放世界视觉导航或真实机器人的安全认证。
7. **系统层边界：** 本仓库是 MoRA-Train 的第一个 adapter；多粒度记忆、进度闭环、跨任务 adapter 尚未实现，因此这里的成功/失败不能作为整个训练系统的结论。

## 6. 复现与实验完整性

推荐 Python 3.10 或 3.12，在 **Linux / WSL2 的 Linux 文件系统**中运行。仓库第三方 Go2W 资产存在只在大小写上不同的文件名，常见 macOS/Windows 文件系统可能无法完整检出。具体要求见[使用指南](docs/USAGE.md)。

```bash
git clone https://github.com/Anhao1314/Go2w-MoRA-navigation.git
cd Go2w-MoRA-navigation
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python - <<'PY'
from rl.go2w_env import Go2wEnv
env = Go2wEnv(task="traverse_curve", domain_randomize=False)
obs, _ = env.reset(seed=0)
print(obs.shape, env.action_space.shape)  # (61,), (6,)
env.close()
PY
```

**无需训练或下载策略的现行控制器基线：**

```bash
python scripts/evaluate_navigation.py --task multi-segment --method controller --episodes 20
python scripts/evaluate_navigation.py --task junction --method controller --episodes 40
```

各次运行生成独立目录，包含 `run_config.json`、`episodes.json` 和 `summary.json`。配置记录源版本、安装版本、输入哈希和评估种子；`complete` 仅表示评估过程完成，不代表任务成功。PPO 回放必须有相匹配、可信的 PPO 权重与 `VecNormalize` 文件，不能用 BC 权重冒充。

```bash
python -m unittest discover -s tests -p 'test_*.py'
pyright rl scripts mujoco_demos
```

[2026-09-12 的封存验证记录](docs/PORTFOLIO_VALIDATION.md)为 **196 passed / 1 skipped、Pyright 0 errors**，属于特定环境和提交的历史结果，**不是本次 README 修改重新执行 CI 的声明**。[2026-10-06 的 `navigation-integrity-v1`](docs/INTEGRITY_UPGRADE.md)修复了种子初始化、课程阶段重复归一化、按回合计数和历史报告覆盖风险，**没有重新训练或追认历史 PPO 成功率**。

旧训练脚本可能覆盖同名演示和报告；需要重新训练时，应使用独立工作区。更多入口、配置、存储位置和兼容限制见[复现指南](docs/USAGE.md)。

## 7. 结论与后续工作

本仓库当前交付的是 MoRA-Train 的**第一个 adapter**：一个可审查的 Go2W 分层导航实验床，以及可独立验证的 C++ S0 控制核心。其价值是**层次接口、控制与学习分工、实验诊断和可复核记录**，不是证明一种新 RL 算法，也不是证明整个 MoRA-Train 系统。

下一步按系统层推进：

1. 在 Go2W 之外实现 Task Adapter API，抽象 `reset / step / goal / progress / success`；
2. 为 S1 加入短期/情节记忆与 progress head，做记忆/进度消融；
3. 把训练事件流接入 RL-Sentinel，量化跨任务、跨 seed 的早停可靠性；
4. 补多 seed、域随机化、单因素消融和匹配的 checkpoint/归一化；
5. 完成 ROS2 P1 垂直切片，再讨论实机边界。

## 8. S0 组件：C++ / ROS2 栈（P0）

在 MoRA-Train 的定位中，本节是 **S0 组件 + 第一个 adapter 的控制层验证**，不是整个训练系统的全部。在现有 Python/MuJoCo 仓库之上新增了一个 **ROS-agnostic C++17 控制核心**，作为可独立验证的 System 0。P0 已完成 C++ core + Python 适配器；ROS2 垂直切片（P1）尚未实现。

**包含：**

- 浮动基座 FK/IK、轮心 Jacobian（对 MuJoCo 夹具 1e-9 / 1e-6）
- 质心质量/COM/惯性（1e-9 / 1e-8）
- SRBD 线性 MPC（OSQP，N=20，50 Hz）
- 接触力 → 轮力矩映射、安全状态机
- 脚本教师 + 可选 A/B 预减速
- pybind11 注入现有 `HighLevelEnvWrapper`

**P0 证据（固定 seeds 1000–1009）：**

| 配置 | 成功率 | mean_x | mean_time |
| --- | ---: | ---: | ---: |
| scripted，无预减速（历史基线） | 0/10 | 4.7811 | 6.292 s |
| scripted，预减速 | 10/10 | 10.6571 | 13.990 s |
| **C++ S0，预减速** | **10/10** | **10.6571** | **13.990 s** |

C++ 与 scripted 预减速逐项相同；seed 1000 的逐步轨迹也完全一致。MPC benchmark：200/200 求解，mean 3.16 ms，p99 4.07 ms，hover 垂直力误差 2.15%。演示视频：[cpp_stack_p0_demo.mp4](media/cpp_stack_p0_demo.mp4)。原生 Windows（MSVC 2026）验证：Python 套件 **225 tests OK（4 skipped）**、绑定 3/3、MPC 200/200（mean 3.88 ms，p99 4.59 ms）、multi-segment **10/10**（mean_x 10.6563，14.000 s）。详见 [Windows 验证](docs/WINDOWS_NATIVE.md)。

**快速开始：**

```bash
cmake -S cpp -B build/core -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGO2W_BUILD_TESTS=ON -DGO2W_BUILD_BENCHMARKS=ON
cmake --build build/core
.venv/bin/ctest --test-dir build/core --output-on-failure
./build/core/go2w_mpc_bench
.venv/bin/python scripts/evaluate_navigation.py \
  --task multi-segment --method controller --episodes 10 \
  --s0 cpp --pre-brake --output reports/cpp_stack/p0_cpp_prebrake
```

**边界：** 仿真-only；无 ROS2 节点、EKF、完整 WBC/RNEA、视觉/触觉、实机或 sim-to-real；MPC 尚未闭环接入 `Go2wEnv`；历史报告成功率不被追认。详见 [C++ 架构](docs/CPP_ROS2_ARCHITECTURE.md) · [控制设计](docs/CONTROL_DESIGN.md) · [标定](docs/CALIBRATION.md) · [验证](docs/VALIDATION.md) · [P0 协议](docs/CPP_STACK_PROTOCOL.md) · [cpp/README](cpp/README.md)。

## 参考文献

[1] Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. [arXiv:1707.06347](https://arxiv.org/abs/1707.06347).

[2] Ross, S., Gordon, G., & Bagnell, J. A. (2011). *A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning*. AISTATS / PMLR 15, 627–635. [论文](https://proceedings.mlr.press/v15/ross11a.html).

[3] Todorov, E., Erez, T., & Tassa, Y. (2012). *MuJoCo: A physics engine for model-based control*. IROS. [DOI](https://doi.org/10.1109/IROS.2012.6386109).

[4] Raffin, A., et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR 22(268). [论文](https://www.jmlr.org/papers/v22/20-1364.html).

**本研究的一手材料：** [架构契约](docs/ARCHITECTURE.md) · [实验与负结果](docs/EXPERIMENTS.md) · [复现协议](docs/USAGE.md) · [实验完整性与遗留问题](docs/INTEGRITY_UPGRADE.md)

## 贡献与许可

**Anhao1314：** 项目负责人，原始系统、机器人/RL 实现、实验与归档证据。**ChatGPT（OpenAI）：** AI 协作参与仓库审查、[实验完整性重构](https://github.com/Anhao1314/Go2w-MoRA-navigation/pull/1)、验证设计与文档维护；该署名不代表独立人类作者或同行评议。自研代码使用 [MIT](LICENSE)，Go2W 资产与第三方内容见[模型许可](models/go2w/LICENSE)及[第三方声明](docs/THIRD_PARTY_NOTICES.md)。
