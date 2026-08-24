# 实验总览与结论

所有数字来自 `reports/` 与 `data/demo_trajectories/` 下的验收文件。本页同时保留
成功与失败实验，避免只按最佳 run 叙述结果。

## 证据口径

- **成功率只在对应报告的任务、环境与 episode 数内成立**，不外推到实机或开放地形；
- `100%` 表示该验收批次无失败，不表示已经证明策略在总体分布上的失败率为零；
- 表内距离、时间和摔倒次数直接采用仓库报告，不从 GIF 主观估计；
- `rl/runs/` 属于本地训练产物，公开快照不持续提交；可复核入口以报告、精选数据与脚本为主；
- 失败实验是根因分析证据，不视为需要从作品集中删除的“无效结果”。

## 成果表

| 任务 | 方法 | 成功率 | 关键指标 | 状态 |
| --- | --- | --- | --- | --- |
| 双轮自平衡 | LQR 型反馈 | - | 保持 10.5s，偏差 0.002 rad | [balance](../reports/balance/seed00/report.md) |
| traverse_curve_high_level seed01 | MoRA-inspired 分层（S0 控制器+S1 PPO） | 100%（20/20） | 1.355m，0 摔倒 | [seed01](../reports/traverse_curve_high_level/seed01/report.md) |
| traverse_curve_high_level_bplus | 随机目标 + 轻量 HER | 90% | 0.834m | [B+](../reports/traverse_curve_high_level_bplus/seed00/report.md) |
| traverse_curve_high_level_bplus_no_goal | 无目标条件消融 | 100% | 1.352m | [no-goal](../reports/traverse_curve_high_level_bplus_no_goal/seed00/report.md) |
| traverse_curve_multi_segment v2 | 2 子目标 + 精确停止 + 三阶段课程 | 100% | 10.73m，15.2s，A 停止 100% | [multi-segment v2](../reports/traverse_curve_multi_segment/seed00_v2/report.md) |
| traverse_curve_junction | System 2 规则 + System 1 导航 | 100%（40/40） | 5.03m，7.0s，分支正确 100% | [junction](../reports/traverse_curve_junction/seed00/report.md) |

## 失败实验（同样重要）

| 实验 | 现象 | 根因与修复 |
| --- | --- | --- |
| 单层 PPO（v1~v5） | 全部 ≤0.7m，站桩/乱动 | 无目标概念 → 分层 + 2 维动作 |
| 高层 seed00（允许停车） | speed_scale=0.5 站桩 0m | 动作下界 0.5→0.9，禁止停车 |
| 多段 seed00（初版） | 完整 0%，冲过 B 点 | 停靠奖励 +30→+100、过冲惩罚、B 点到达即终止、docking 连续化 |
| stage2 从岔路口人工起点 | 0%（yaw=0 起始分布不匹配） | 完整任务从自然到达状态仍 100%，人工中间态课程无效 |

## 研究文档索引

- `data/demo_trajectories/bc_ablation_report.md` — BC 消融
- `data/demo_trajectories/bplus_evaluation_report.md` — B+ 随机目标评估
- `data/demo_trajectories/multi_segment_validation_report.md` — 多段 P0 验证
- `data/demo_trajectories/multi_segment_bc_report.md` / `multi_segment_stage2_bc_report.md` — BC 预热
- `data/demo_trajectories/junction_bc_report.md` — 岔路口 BC
- `data/demo_trajectories/dagger_training_report.md` — DAgger 训练
- `data/demo_trajectories/curve_curriculum_report.md` — 课程学习
- `data/demo_trajectories/curve_v5_quick_validation.md` — v5 快速验证
- `docs/THIRD_PARTY_NOTICES.md` — 第三方许可

## 教训总结

1. 任务几何必须先用脚本控制器验证可走通，再训练 RL（岔路口终点曾因几何过陡全部翻车）。
2. 动作空间里"允许停车"等于给策略留了合法出口；禁止停车是打破站桩局部最优的最小改动。
3. BC 预热 + norm 冻结 + 学习率 warmup 可避免 BC→PPO negative transfer。
4. 人工构造的中间态课程（岔路口 yaw=0 起步）与真实分布不匹配时无效；
   从完整任务自然状态出发反而能学到正确行为。

## 尚未验证

- 真实 Go2w 上的控制安全、通信时延和 sim-to-real；
- 未见地形、不同摩擦/载荷和更大规模随机种子的泛化；
- 学习型 System 2、自然语言目标或视觉输入；
- 与强基线在统一算力、统一调参预算下的统计显著性比较。

这些项目应作为下一阶段实验，不应由当前 100% 验收结果直接推断为已经完成。
