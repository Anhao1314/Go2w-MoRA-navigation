# traverse_curve 分层控制（MoRA 第一步）训练报告

生成时间：2026-08-21 17:16:51

## 架构

- System 0：`LowLevelController`（前视跟踪 + 差速转向，forward=0.12）
- System 1：PPO 高层策略，动作 2 维 [speed_scale, turn_adjust]
- 观测：61 维不变；奖励：v5（摔倒 -10、到达 +200）

## 训练配置

- 总步数目标：2M；实际提前停止：success
- envs=4，lr=3e-4，ent_coef=0.01，best=距离稳定窗口（ep_len>500）
- run 目录：`rl/runs/traverse_curve_high_level/seed01`

## 训练曲线（每 50k 评估）

| timesteps | reward | ep_len | mean_x | success |
|---|---|---|---|---|
| 50000 | 284.24 | 651 | 1.355 | 100% |
| 100000 | 282.79 | 641 | 1.351 | 100% |
| 150000 | 282.74 | 643 | 1.355 | 100% |

## 正式验收（20 episodes，全新环境）

- 成功率：100%（20/20）
- 平均距离：1.355 m（最大 1.355）
- 平均 ep_len：643
- 摔倒次数：0
- 平均总奖励：282.74
- 最大俯仰偏差：0.0103 rad
- 高层动作：speed=0.90±0.00，turn=0.09±0.41

## 基线对比

| 方法 | 平均距离 | 成功率 |
|---|---|---|
| 脚本控制器（固定[1.0,0.0]） | 0.915m | 0% |
| 纯 PPO v5（6 维动作） | 0.000m | 0% |
| DAgger iter1 BC | 0.915m | 0% |
| 高层 RL（本 run） | 1.355m | 100% |

## 结论

✅ 分层控制有效（距离>1.5m 或成功率>10%），可进入第二步（记忆/目标条件）

## 演示视频（全局俯视）

- MP4：`media/rl_traverse_curve_high_level_seed01.mp4`
- GIF：`media/rl_traverse_curve_high_level_seed01.gif`
