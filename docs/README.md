# 文档索引

[返回首页](../README.md)

本页是 `docs/` 的唯一导航入口，按「系统层 / Phase-1（已冻结）/ Phase-2（进行中）/ 跨阶段」分组。
每篇文档标注状态：🟢 活跃 · 🧊 冻结（随 Phase-1 归档，仅为复现维护）。
各代码区域的归属与冻结状态见[仓库布局契约](ARCHITECTURE.md#仓库布局契约)。

## 跨阶段（🟢 活跃）

| 文档 | 内容 |
| --- | --- |
| [ROADMAP.md](ROADMAP.md) | **Q4 项目目标与四周路线图**：北极星目标、四项能力证据、P0–P4 优先级、验收标准、风险预案、讲述要点 |
| [EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) | **M1/M2 预注册实验协议**（v1.0，2026-10-22 冻结）：研究问题、冻结种子与指标、验收阈值、场景阶梯 S1–S4、基线。Phase-2 的执行契约，changelog 记录一切冻结后变更 |
| [EXPERIMENT_DATA.md](EXPERIMENT_DATA.md) | 实验数据记录与论文证据协议：一次 run 保存什么、每类数据支持什么结论；Phase-2 沿用同一证据束约定 |
| [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) | 第三方许可登记（含 C1 基线仓库的待审查条目） |
| [LIVE_SYNC.md](LIVE_SYNC.md) | 本地与 GitHub 的实时同步：提交后自动推送 + 每 2 分钟自动快进 `main`，安装/卸载与边界说明 |
| [MORA_TRAIN.md](MORA_TRAIN.md) | MoRA-Train 系统层（v0.1）：`mora/` 包接口、当前能力与边界，跨 adapter 复用 |
| [MORA_RESEARCH.md](MORA_RESEARCH.md) | MoRA 架构深度研究：公开资料还原、仓库差距分析与 v2 路线（研究记录，非复现声明） |

## Phase-2：Go2 关节级 locomotion（🟢 活跃，代码在 [`locomotion/`](../locomotion/README.md)）

Phase-2 尚在起步，除协议文档外暂无独立设计文档；`locomotion/sim2sim/README.md`
（S4 设计占位）与 `locomotion/baselines/convex_mpc/README.md`（C1 接入位）在包内。
后续设计文档在此登记。

## Phase-1：Go2W 轮足导航 adapter（🧊 已冻结归档）

以下文档正文保持原样，不随 Phase-2 回写重释；其描述的架构、命令与脚本仍可复现 Phase-1 结果。

### 研究与结果

| 文档 | 内容 |
| --- | --- |
| [ARCHITECTURE.md](ARCHITECTURE.md) | 分层控制架构：S2 规则 / S1 PPO / S0 脚本-PD 的机制、观测演进与实现入口（含仓库布局契约段） |
| [EXPERIMENTS.md](EXPERIMENTS.md) | 实验总览与结论：成功与失败实验结果表（证据口径：只认 `reports/` 与 `data/` 归档） |
| [INTEGRITY_UPGRADE.md](INTEGRITY_UPGRADE.md) | 2026-10-06 `navigation-integrity-v1` 完整性升级：种子、归一化、计数与报告覆盖修复（未追认历史成功率） |
| [PORTFOLIO_VALIDATION.md](PORTFOLIO_VALIDATION.md) | 2026-09-12 封存验证记录（196 passed / Pyright 0 errors，特定环境历史结果） |
| [VALIDATION.md](VALIDATION.md) | P0 验证报告：C++ 控制核心、Python 适配器、多段集成、MPC 核心的数值验证 |

### C++ S0 控制栈

| 文档 | 内容 |
| --- | --- |
| [CONTROL_DESIGN.md](CONTROL_DESIGN.md) | C++ 控制核心设计：模型、公式与实现约定 |
| [CPP_STACK_PROTOCOL.md](CPP_STACK_PROTOCOL.md) | C++ Stack P0 评估协议（版本 `cpp-stack-p0-v1`，含模型 SHA-256） |
| [CPP_ROS2_ARCHITECTURE.md](CPP_ROS2_ARCHITECTURE.md) | C++/ROS2 System-0 架构：P0 已实现，P1（ROS2 垂直切片）尚未开始 |
| [CALIBRATION.md](CALIBRATION.md) | C++ 控制栈标定记录（符号/顺序约定须有探针或夹具支撑） |
| [WINDOWS_NATIVE.md](WINDOWS_NATIVE.md) | Native Windows (MSVC) 环境验证记录 |

### 运行与复现

| 文档 | 内容 |
| --- | --- |
| [USAGE.md](USAGE.md) | 使用与复现指南（Phase-1 训练/评估/检查命令） |
| [WINDOWS_TRAINING.md](WINDOWS_TRAINING.md) | Windows 训练流程（Phase-1 Go2W） |

## 阅读路径建议

- 复现 Phase-1 结果：首页研究报告 → [USAGE.md](USAGE.md) → 对应 `reports/` 目录
- 参与 Phase-2：[EXPERIMENT_DESIGN.md](EXPERIMENT_DESIGN.md) → [仓库布局契约](ARCHITECTURE.md#仓库布局契约) → [locomotion/README](../locomotion/README.md) → [MORA_TRAIN.md](MORA_TRAIN.md)（若接入 MoRA runtime）
- 审查证据链：[EXPERIMENTS.md](EXPERIMENTS.md)（Phase-1 结果表）→ [EXPERIMENT_DATA.md](EXPERIMENT_DATA.md)（证据束约定）→ run 目录与 `evidence_manifest.json`
