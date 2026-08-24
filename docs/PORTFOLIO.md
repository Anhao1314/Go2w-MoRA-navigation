# 作品集说明：从失败实验到可验证的分层导航系统

这份文档面向招聘方、研究合作者和第一次阅读仓库的工程师。它不重复 README 的使用说明，而是回答四个问题：**项目解决了什么、我具体做了什么、结果如何验证、哪些事情还没有完成。**

## 一分钟介绍

这是一个基于 MuJoCo 的 Unitree Go2w 分层导航强化学习项目。初始单层 PPO 直接从 61 维状态预测 6 维底层动作，在弯道和长程任务上反复学到站桩或短视行为。我把系统拆为三个可分别验证的层级：

1. System 0 用前视点与差速控制稳定执行运动并生成演示；
2. System 1 用 PPO 学习 2 维高层导航修正，并维护目标、进度和停靠状态；
3. System 2 在岔路口做可审计的离散分支决策。

项目价值不只是“最后跑通”，而是保留了 PPO、BC、DAgger、课程学习和奖励设计的失败证据，再通过消融实验把问题定位到动作空间、状态分布和终止条件。最终，本仓库验收集上的单段弯道、10 m 两段导航和岔路口任务均达到 100% 成功率。

## 我的工作范围

| 工作面 | 具体内容 | 仓库入口 |
| --- | --- | --- |
| 问题建模 | 将弯道、多子目标、精确停止和岔路口决策拆成逐级任务 | [`rl/go2w_env.py`](../rl/go2w_env.py) |
| 分层架构 | 实现低层控制器、高层动作包装、目标/进度/决策观测 | [`rl/low_level_controller.py`](../rl/low_level_controller.py)、[`rl/high_level_env_wrapper.py`](../rl/high_level_env_wrapper.py) |
| 学习流程 | PPO、BC 预热、DAgger、课程学习与阶段模型衔接 | [`rl/train.py`](../rl/train.py)、[`scripts/`](../scripts/) |
| 实验设计 | 动作下界、Goal-Conditioned、课程起点、停靠奖励等消融 | [`docs/EXPERIMENTS.md`](EXPERIMENTS.md) |
| 评估证据 | 固定协议验收、失败记录、报告、曲线和演示视频 | [`reports/`](../reports/)、[`media/`](../media/) |
| 工程质量 | 单元测试、静态检查、无头渲染、训练监控和报告生成 | [`tests/`](../tests/)、[`rl/webpanel.py`](../rl/webpanel.py) |

## 最能体现能力的三个决策

### 1. 不再把“训练更久”当作默认答案

纯 PPO 多轮失败后，我没有继续无边界地增加步数，而是构造单变量对照：只改变高层速度动作的最小值。`speed_scale` 从允许 0.5 调整为至少 0.9 后，策略从站桩 0% 变为 100%。这说明问题首先是控制接口设计，而不是算力不足。

体现的能力：实验设计、局部最优诊断、动作空间建模、成本意识。

### 2. 用自然状态分布审视课程学习

岔路口课程的人工中间态看起来更简单，却在多个评估点持续失败；从正常起点训练的完整任务反而成功。根因是人工设置的 yaw 和速度分布与策略自然到达岔路口时不同。由此把“课程由易到难”改为“课程状态分布必须真实”。

体现的能力：分布偏移分析、反直觉结果解释、训练流程修正。

### 3. 把精确停止变成可观测、可奖励、可终止的闭环

多段导航初版能到 B 点附近但持续过冲。修复不是简单加大位置奖励，而是同时加入停留时间、速度阈值、过冲惩罚和达标终止，使“停好”成为完整状态机而非单帧距离判断。

体现的能力：奖励工程、状态机设计、验收指标设计、端到端调试。

## 证据链

项目结论按“代码—配置—报告—媒体”四类证据组织：

| 结论 | 报告 | 可视化/补充证据 |
| --- | --- | --- |
| 分层弯道 20/20 | [`reports/traverse_curve_high_level/seed01/report.md`](../reports/traverse_curve_high_level/seed01/report.md) | [`curve_high_level_ablation.png`](images/curve_high_level_ablation.png) |
| Goal 输入在简单任务中无独立增益 | [`B+`](../reports/traverse_curve_high_level_bplus/seed00/report.md) / [`no-goal`](../reports/traverse_curve_high_level_bplus_no_goal/seed00/report.md) | [`bplus_evaluation_report.md`](../data/demo_trajectories/bplus_evaluation_report.md) |
| 10 m A→B + 精确停止 | [`multi-segment v2`](../reports/traverse_curve_multi_segment/seed00_v2/report.md) | [`multi_segment_curriculum.png`](images/multi_segment_curriculum.png) |
| 岔路口分支 40/40 | [`junction`](../reports/traverse_curve_junction/seed00/report.md) | [`junction_curriculum.png`](images/junction_curriculum.png) |
| 早期方法失败与原因 | [`EXPERIMENTS.md`](EXPERIMENTS.md) | [`data/demo_trajectories/`](../data/demo_trajectories/) |

“100%”只对应表中指定协议和 episode 数，不外推到未测试地形、真实机器人或开放世界。

## 技术能力映射

| 能力 | 项目中的体现 |
| --- | --- |
| Python 工程 | 模块化环境/包装器/训练脚本、CLI、报告与测试 |
| 强化学习 | PPO、奖励塑形、动作空间、观测设计、VecNormalize、课程学习 |
| 模仿学习 | BC 预热、DAgger、专用阶段数据集、negative transfer 分析 |
| 机器人仿真 | MuJoCo 模型、传感观测、差速控制、精确停靠、EGL 无头渲染 |
| 实验方法 | 对照组、消融、固定验收协议、失败归档、证据可追溯 |
| 工程运维 | 训练守护、Web 监控、资源监控、自动报告/视频、CPU smoke 验证 |

## 诚实边界

- **仅仿真**：尚未在真实 Go2w 上完成 sim-to-real；摩擦、时延、传感噪声和执行器误差仍需验证。
- **System 2 是规则模块**：它验证了分层接口，不等于已经实现通用 Agent 或 VLM 决策。
- **样本规模有限**：主要结论来自仓库明确列出的 seed 与 episode，不能等价为统计上的普适结论。
- **公开仓库是精选快照**：完整历史训练目录和运行服务不入库，报告中若引用 `rl/runs/`，需重新训练才能生成对应原始日志。
- **不是交易策略**：机器人反馈闭环为后续量化系统提供方法论和数据工程原型，但这里没有市场数据、回测、风控或实盘交易。

## 与量化系统的关系

机器人与交易的动作对象不同，但可以复用相同的研究骨架：

| 机器人导航 | 量化研究中的对应概念 |
| --- | --- |
| observation | 市场状态与特征 |
| target / branch | 策略目标与市场状态分类 |
| policy action | 仓位或交易指令 |
| reward / success | 收益、风险与约束后的目标函数 |
| episode evaluation | 时间切片回测与样本外评估 |
| run configuration | 数据版本、参数、代码版本与实验血缘 |

可复用的是**可追溯的决策反馈闭环**，不是把机器人 reward 直接转换成交易信号。后续量化模块应独立完成数据契约、时间一致性、交易成本、风险约束和样本外检验。

## 面试讲解建议（5 分钟）

1. **30 秒**：说明单层策略为什么在长程任务失败；
2. **60 秒**：画出 System 2→1→0 三层接口；
3. **90 秒**：讲动作下界 0.5→0.9 的单变量消融；
4. **60 秒**：讲人工中间态分布偏移这一反直觉结果；
5. **60 秒**：展示 multi-segment 或 junction GIF 与验收报告；
6. **40 秒**：主动说明仿真、规则 System 2 和样本规模限制。

推荐先展示“失败如何被定位”，再展示 100% 结果。这个顺序更能体现工程判断，而不只是训练出一个模型。

## 下一阶段

1. 为仿真到实机补充域随机化、延迟/噪声建模与安全约束；
2. 将逐 run 与逐 episode 结果输出为稳定数据契约；
3. 建立跨设备只读观测、增量导出和实验血缘；
4. 在独立量化模块中完成市场数据接入、特征、回测、交易成本与风险控制；
5. 只有样本外和纸面交易门禁通过后，才讨论真实交易接口。
