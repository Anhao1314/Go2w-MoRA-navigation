# traverse_curve DAgger→PPO 微调报告（提前停止）

生成时间：2026-08-21 14:45

## 配置

- 初始化：`data/demo_trajectories/bc_iter1.zip`（DAgger iter1，BC 平均距离 0.915m）
- VecNormalize：`curve_vecnorm.pkl` 初始化，前 500k 步冻结（实际未到解冻点）
- 奖励：v5，摔倒/早停 -10，到达 +200，前进 +1.0×dx，势能 shaping，航向 1.5
- 学习率：预热调度 100k 步 5e-5 → 之后 1.5e-4
- ent_coef：0.005；探索增强：启用（双模式检测）
- best 选择：`--best-metric distance`（ep_len>500 才保存）
- 步数目标：2M；实际执行：**200k 后按 negative transfer 规则提前停止**

## 训练曲线（eval_log.csv）

| timesteps | mean_reward | mean_ep_len |
|---|---|---|
| 50k | -27.38 | 34 |
| 100k | -29.67 | 23 |
| 150k | -27.84 | 33 |
| 200k | -25.36 | 45 |

## 提前停止原因

- BC iter1 的 ep_len 为 1500（满时长），PPO 微调 50k 后崩到 23~45（约 0.2~0.45 秒即倒）；
- 200k 时仍在“快速摔倒”区间，判定为 negative transfer（BC 行为被破坏）；
- 因 distance-best 要求 ep_len>500 才保存，**本 run 未产出 best/final 模型**；
- 探索增强在 200k 前未触发（需连续 5 个 eval 点，实际只观察到 4 个点即停止）。

## 结论

❌ 失败（negative transfer，无模型产出）。BC 的稳定行为在 PPO 更新下迅速丢失，
即使使用 5e-5 预热学习率、norm 冻结和低熵系数也无法保护。

## 后续建议

1. 微调学习率再降一个量级（1e-5~3e-5）并加 BC 行为 KL 约束 / 行为克隆损失混合；
2. 先只微调 actor 最后一层或冻结前几层，再逐步放开；
3. 或放弃“BC 初始化 + PPO 在线微调”，改用离线 RL（如 IQL/CQL）或直接部署教师控制器；
4. 若继续 PPO，可在回调中增加“保存当前模型”的提前停止钩子，避免失败 run 无 checkpoint。
