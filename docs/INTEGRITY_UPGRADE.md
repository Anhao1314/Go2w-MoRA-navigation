# 实验完整性升级：navigation-integrity-v1

[返回首页](../README.md) · [历史实验](EXPERIMENTS.md) · [架构](ARCHITECTURE.md)

本轮基于 `3d9655f4a3884421fed9620050bfaf8c62ba8e69` 改进可复现性和实验记录，
不改变机器人几何、奖励函数、脚本停靠或岔路口规则转向，不新增实机能力。
历史报告、GIF、教师数据和 BC 模型保持原样；历史成功率不是本轮重新训练的结果。

## 本轮修复

| 问题 | 修复 | 验证方向 |
| --- | --- | --- |
| 包装器采样目标前没有初始化自己的 RNG | `reset(seed=...)` 先初始化 RNG，再调用子类选项目标钩子 | 同种子重复、跨实例一致、无种子连续流、岔路口目标可重复 |
| 课程后续阶段把已归一化环境再次传给 `VecNormalize.load` | `normalize_once` 只接受未归一化的 VecEnv | 直接/间接嵌套拒绝、恢复后的数值与冻结统计 |
| 加载 BC/PPO 时沿用保存模型的种子 | 各阶段加载显式传递本次 `--seed` | 实验配置记录实际请求种子；多训练种子效果仍需实跑 |
| 持续为真的成功/过 A 点标志可能按 step 累加 | 先记录 episode 标志，再每回合累加一次 | 持续标志测试的比例仍为 1.0 |
| 最终验收与模型选择重复使用 0..4 | 新验收默认从 1000 开始，并拒绝重叠区间 | 逐回合保存种子；这不构成独立场景泛化 |
| 任意运行目录仍覆盖固定 `reports/.../seed00` 与演示 | 报告、配置、可选录像全部跟随新 run 目录 | 已有目录拒绝写入，缺少输入时不创建 run |
| 非成功终止全部被叫作摔倒 | 新报告用 `failed_terminations` | 明确其还可能包含出界等原因，不伪造细粒度原因 |

多段任务仍是 stage1、stage2 分别加载对应 BC 模型，stage3 接续 stage2。
本轮修正了旧注释中“每阶段都接续上一阶段”的错误描述，但没有改写这套训练方案。

## 不训练即可执行的基线

在 Linux / WSL2 Linux 文件系统安装 `requirements-dev.txt` 后：

```bash
python scripts/evaluate_navigation.py --task multi-segment --method controller --episodes 20
python scripts/evaluate_navigation.py --task junction --method controller --episodes 40
```

此基线固定高层命令为 `[1.0, 0.0]`，保留已有路径跟踪、停靠与规则分支。
它回答的是“同一套环境与底层控制在没有学习策略时表现怎样”，而不是“无控制器机器人能否成功”。
岔路口默认转向仍不使用 `turn_adjust`；把它接通必须作为新的动作协议与独立实验，不能套用旧 40/40 结果。

回放已有的、可信的 PPO checkpoint：

```bash
python scripts/evaluate_navigation.py \
  --task multi-segment --method ppo \
  --checkpoint /path/to/best_model.zip \
  --normalization /path/to/best_vec_normalize.pkl \
  --episodes 20 --seed-start 1000
```

路径是需要替换的本地文件路径。命令不下载模型，不把 BC 权重当作 PPO 结果。
归一化文件必须是相匹配的 `VecNormalize` 保存文件，而不是 BC 的裸统计字典。
加载 checkpoint 会使用 pickle / 模型反序列化，只使用可信来源。

`--output` 可指定一个**不存在的新目录**。不指定时自动在
`rl/runs/navigation_evaluation/` 创建唯一目录。输出包括：

- `run_config.json`：任务、方法、种子、源代码提交、工作树状态、安装版本、输入文件 SHA-256；
- `episodes.json`：每回合种子、成功状态、进度、步数、时间、终止/截断标志及任务状态；
- `summary.json`：由实际回合汇总的指标与适用边界。`complete` 只表示评估执行完成，不表示策略成功率达标。

两种方法必须使用一致任务、回合种子、环境版本和协议才能比较。
`max_progress` 沿用环境指标，不代表积分得到的实际行驶路径长度。
不输出未经测量的提升百分比，也不把两回合 smoke 的结果写成正式 benchmark。

## 新课程运行方式

```bash
python scripts/train_multi_segment_curriculum.py --smoke --seed 1
python scripts/train_junction_curriculum.py --smoke --seed 1
```

正式训练去掉 `--smoke`；需要录像时增加 `--record`。
`--run-root` 现在必须指向一个**不存在的新目录**，否则直接报错。
默认使用唯一目录，不继续写历史 seed00 路径。本入口不实现已有目录续训。

每轮输出包含 `run_config.json`、各阶段模型/归一化文件与日志。
正式验收写在该 run 的 `evaluation/`；smoke 写在 `smoke_evaluation/`，
结论标记为 `smoke_only`，不会录制或覆盖历史演示。
开启录像时输出该 run 的 `evaluation/demo.*`。
`docs/USAGE.md` 中的旧报告路径属于历史协议；本节是两个课程脚本的新输出约定。
其他教师生成与弯道训练入口未改造，仍应使用隔离副本，避免覆盖同名历史产物。

修复随机种子与归一化会改变训练分布与结果。旧 checkpoint 不会被自动修复，
旧归一化统计也不能直接证明新流程有效；必须把新结果作为新 run 记录。

## 测试与证据边界

```bash
python -m unittest discover -s tests -p 'test_*.py'
pyright rl scripts mujoco_demos
```

新增 `test_experiment_integrity.py` 与 `test_navigation_integrity.py`。
后者同时包含明确标记的合成单元测试夹具，以及真实 MuJoCo 控制器 CLI 的
两任务、各两回合 smoke。合成夹具不作为仿真结果；真实 smoke 也不要求策略成功，
只检查能够完成评估并写出实际回合证据。测试数量与 CI 成败以对应提交的 Actions 为准。

## 仍需单独完成的工作

成功 PPO checkpoint、完整训练日志和可下载 Release 仍缺失，本轮没有补造。
多训练种子、独立场景/域随机化测试、受控消融与实机评估也没有由本轮自动完成。

下一轮应先审计 BC 数据的观测掩码、训练/推理归一化一致性和仅由训练集估计统计量，
再重新生成 BC 与 PPO 产物。当前多段 BC 脚本的原始观测训练与归一化推理路径不能
因为课程归一化修复而视为已解决。

弯道 `_curve_arc_grid` 的累计长度公式需单独版本化修正：水平项应对应单位斜率，
即 `sqrt(1 + (dy/dx)^2) * dx`，不能把 `dx` 再放入 `hypot` 的水平项。
这会改变进度、目标与奖励口径，因此本轮未悄悄重定义历史弯道任务；旧“距离”不能
直接作为物理路径长度进行跨版本比较。

弯道 `HerAuxCallback` 仍是 future-goal relabeling 加 actor 辅助 MSE，
不是标准 off-policy HER；旧 CLI 与报告生成器的术语迁移不属于本轮修复范围。
不得据此新增“PPO + 标准 HER”或“HER 带来提升”的成果声明。
