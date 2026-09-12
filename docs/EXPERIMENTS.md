# 实验总览与结论

[返回首页](../README.md) · [架构](ARCHITECTURE.md) · [实验](EXPERIMENTS.md) · [使用指南](USAGE.md)

所有数字来自 `reports/` 与 `data/demo_trajectories/` 下的验收文件。本页同时保留
成功与失败实验，避免只按最佳 run 叙述结果。

## 证据口径

- **成功率只在对应报告的任务、环境与 episode 数内成立**，不外推到实机或开放地形；
- `100%` 表示该验收批次无失败，不表示已经证明策略在总体分布上的失败率为零；
- 表内距离、时间和摔倒次数直接采用仓库报告，不从 GIF 主观估计；
- `rl/runs/` 属于本地训练产物，公开快照不持续提交；可复核入口以报告、精选数据与脚本为主；
- 失败实验是根因分析证据，用于保留未达标现象与后续修改的依据。

## 结果表

| 任务 | 方法 | 成功率 | 关键指标 | 状态 |
| --- | --- | --- | --- | --- |
| 双轮自平衡 | LQR 型反馈 | - | 保持 10.5s，偏差 0.002 rad | [balance](../reports/balance/seed00/report.md) |
| traverse_curve_high_level seed01 | MoRA-inspired 分层（S0 控制器+S1 PPO） | 100%（20/20） | 1.355m，0 摔倒 | [seed01](../reports/traverse_curve_high_level/seed01/report.md) |
| traverse_curve_high_level_bplus | 随机目标（报告配置 HER 关闭） | 90% | 0.834m，2 摔倒 | [B+](../reports/traverse_curve_high_level_bplus/seed00/report.md) |
| traverse_curve_high_level_bplus_no_goal | 无目标条件消融 | 100% | 1.352m | [no-goal](../reports/traverse_curve_high_level_bplus_no_goal/seed00/report.md) |
| traverse_curve_multi_segment v2 | 2 子目标 + 精确停止 + 三阶段课程 | 100%（20/20） | 10.73m，15.2s，A 停止 100% | [multi-segment v2](../reports/traverse_curve_multi_segment/seed00_v2/report.md) |
| traverse_curve_junction | System 2 规则 + System 1 导航 | 100%（40/40） | 5.03m，7.0s，分支正确 100% | [junction](../reports/traverse_curve_junction/seed00/report.md) |

## 失败实验

| 实验 | 现象 | 可能解释与后续修改 |
| --- | --- | --- |
| 单层 PPO（v1~v5） | 全部 ≤0.7m，站桩/乱动 | 控制接口、探索与任务状态可能共同影响；后续采用分层与 2 维动作 |
| 高层 seed00（允许停车） | speed_scale=0.5 站桩 0m | 动作下界 0.5→0.9，禁止停车 |
| 多段 seed00（初版） | 完整 0%，冲过 B 点 | 停靠奖励 +30→+100、过冲惩罚、B 点到达即终止、docking 连续化 |
| stage2 从岔路口人工起点 | 0%（人工起点 yaw=0、静止） | 完整任务成功；状态分布偏移是待进一步对照验证的解释 |

## 研究文档索引

- [bc_ablation_report.md](../data/demo_trajectories/bc_ablation_report.md) — BC 消融
- [bplus_evaluation_report.md](../data/demo_trajectories/bplus_evaluation_report.md) — B+ 随机目标评估
- [multi_segment_validation_report.md](../data/demo_trajectories/multi_segment_validation_report.md) — 多段 P0 验证
- [multi_segment_bc_report.md](../data/demo_trajectories/multi_segment_bc_report.md) / [multi_segment_stage2_bc_report.md](../data/demo_trajectories/multi_segment_stage2_bc_report.md) — BC 预热
- [junction_bc_report.md](../data/demo_trajectories/junction_bc_report.md) — 岔路口 BC
- [dagger_training_report.md](../data/demo_trajectories/dagger_training_report.md) — DAgger 训练
- [curve_curriculum_report.md](../data/demo_trajectories/curve_curriculum_report.md) — 课程学习
- [curve_v5_quick_validation.md](../data/demo_trajectories/curve_v5_quick_validation.md) — v5 快速验证
- [THIRD_PARTY_NOTICES.md](../docs/THIRD_PARTY_NOTICES.md) — 第三方许可

## 实验观察与解释

1. 调整高层速度下界后，归档弯道实验从站桩转为成功。结果提示动作约束影响探索，但不足以证明该修改在任意任务上都有效。
2. 岔路口人工中间态阶段失败，完整起点训练成功。人工与自然到达状态的速度、航向差异可能影响结果，尚需匹配状态分布的对照。
3. 多段 v2 同时调整停靠奖励、过冲惩罚、终止逻辑和阶段 BC。整体结果改善，单项贡献尚不能分别归因。
4. 简单弯道有、无目标条件均可成功。不同评估目标与条件需要区分，不能据此证明复杂任务中目标输入的必要性。

### 证据说明

[B+ 报告](../reports/traverse_curve_high_level_bplus/seed00/report.md)配置写明 HER 关闭，末尾结论却提到 HER；本页按配置记录，不将结果归因于 HER。随机目标验收有 2 次摔倒，固定目标验收为 0 次。

[弯道 seed01 报告](../reports/traverse_curve_high_level/seed01/report.md)已经记录固定脚本指令基线（0.915 m、0%），但后续多段与岔路口仍需在统一环境、预算和评估条件下补充对照。System 0 跟踪、脚本停靠及规则分支的贡献不能直接算作 PPO 的独立能力。

当前验收通过遍历回合种子进行重复评估，不等价于独立场景或多个训练种子。多段与岔路口环境关闭域随机化，阶段评估与最终验收的种子有重叠；未来应区分模型选择与独立测试。

## 尚未验证

- 真实 Go2w 上的控制安全、通信时延和 sim-to-real；
- 未见地形、不同摩擦/载荷和更大规模随机种子的泛化；
- 学习型 System 2、自然语言目标或视觉输入；
- 与强基线在统一算力、统一调参预算下的统计显著性比较。

这些项目应作为下一阶段实验，不应由当前 100% 验收结果直接推断为已经完成。
