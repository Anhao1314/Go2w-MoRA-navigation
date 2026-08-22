# 实验总览与结论

所有数字来自 `reports/` 与 `data/demo_trajectories/` 下的真实验收文件。

## 成果表

| 任务 | 方法 | 成功率 | 关键指标 | 状态 |
|---|---|---|---|---|
| 双轮自平衡 | LQR 型反馈 | - | 保持 10.5s，偏差 0.002 rad | ✅ |
| traverse_curve_high_level seed01/02 | MoRA 分层（S0 控制器+S1 PPO） | 100% | 1.355m，0 摔倒 | ✅ |
| traverse_curve_high_level_bplus | 随机目标 + 轻量 HER | 90% | 0.834m | ✅ |
| traverse_curve_high_level_bplus_no_goal | 无目标条件消融 | 100% | 对照组 | ✅ |
| traverse_curve_multi_segment v2 | 2 子目标 + 精确停止 + 三阶段课程 | 100% | 10.73m，15.2s，A 停止 100% | ✅ |
| traverse_curve_junction | System 2 规则 + System 1 导航 | 100% | 5.03m，7.0s，分支正确 100% | ✅ |

## 失败实验（同样重要）

| 实验 | 现象 | 根因与修复 |
|---|---|---|
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
