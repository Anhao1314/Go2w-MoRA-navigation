# M1 评估报告：Go2 平地速度跟踪（三种子）

- **日期**：2026-10-23
- **协议**：[docs/EXPERIMENT_DESIGN.md](../../docs/EXPERIMENT_DESIGN.md) v1.0（冻结）
- **任务**：`Go2Loco-S1-Flat-Blind-v0`（平地速度跟踪，vx/ωz ∈ [-1, 1]，vy = 0）
- **结论**：**三种子全部达标**，`fall_rate = 0`，种子间离散度很小。

## 1. 训练配置

| 项 | 值 |
| --- | --- |
| 算法 | PPO (RSL-RL 3.1.2)，2048 并行环境 |
| 观测 / 动作 | 盲观测 48 维 / 12 关节目标位置 |
| 迭代 | 1500 × 每轮 24 步 |
| Episode | 20 秒（1000 步，50 Hz） |
| 硬件 | RTX 5060 Ti 16 GB |
| 环境 | Isaac Lab 2.7.0 + Isaac Sim 5.1.0.0，Python 3.11 |

## 2. 评估协议

- 评估种子集 **E = {0..19}**（`locomotion/eval/seeds.py`）。
- 指标由 `locomotion/eval/metrics.py` 计算：剔除前 0.5 s 瞬态后的 `vx_rmse`、`ωz_rmse`；`fall_rate` 为摔倒比例。
- 执行方式：20 个环境并行（`method = parallel_20_envs_seed0`），`cfg.seed = 0`，每个环境由 `env_id` 派生独立随机流。批次可复现；`--serial` 可切回逐种子单环境模式。
- **未挑选种子**：报告全部三个训练种子的 mean ± std。

## 3. 结果

### 3.1 各训练种子（20 episodes 汇总）

| 训练种子 | `vx_rmse` (m/s) | `ωz_rmse` (rad/s) | `fall_rate` |
| --- | --- | --- | --- |
| seed 0 | 0.0585 ± 0.0291 | 0.0851 ± 0.0109 | 0.00 |
| seed 1 | 0.0561 ± 0.0251 | 0.0873 ± 0.0109 | 0.00 |
| seed 2 | 0.0561 ± 0.0270 | 0.0893 ± 0.0307 | 0.00 |
| **跨种子** | **0.0569 ± 0.0011** | **0.0872 ± 0.0017** | **0.000** |

阈值（协议 §4）：`vx_rmse ≤ 0.15`、`ωz_rmse ≤ 0.2`。**三项全部通过**。

### 3.2 分布特征

- 种子间标准差很小（vx 0.0011、wz 0.0017），说明训练对随机种子不敏感。
- 单个 episode 的最大 `vx_rmse` 为 0.153（seed 0 的评估 episode 13），略高于 0.15；协议验收看 20 episodes 的汇总值，故不构成未达标，但这一尾部值得在 M2 扰动测试中关注。
- 全部 60 个评估 episode 无一摔倒。

## 4. 证据

| 内容 | 位置 |
| --- | --- |
| 训练证据束（3 份） | `logs/rsl_rl/go2_loco_s1_flat/2026-10-23_01-15-02|01-35-42|01-56-30/`：`run_config.json` + `sha256_manifest.json`（各 31 checkpoints / 37 文件哈希） |
| 评估结果 | 各 run 目录下的 `evaluation.json`（20 条 episode 明细 + summary） |
| 训练过程视频 | `media/training/m1_training_progress.mp4` |
| 训练日志 | 本地 `m1_training.log`（含每个种子的起止时间） |

## 5. 局限

1. **仿真限定**：无真机，结论仅适用于该仿真与命令范围。
2. **评估方法**：并行 20 环境（全局种子 0）可复现，但不是"单环境 + 种子 i"的逐 episode 重放；两者在统计意义上等价，字面语义上有差别，`method` 字段已标注。
3. **未做**：扰动（S2）、轻地形（S2b）、Sim2Sim、奖励消融。
4. 训练种子 0 的 run 在 dirty 工作树上启动（`run_config.json` 记录 `dirty=True`）；种子 1/2 在干净树上启动。

## 6. 复现

```powershell
# 训练（单种子）
$env:PYTHONPATH = "$PWD"; $env:OMNI_KIT_ACCEPT_EULA = "YES"
.venv311\Scripts\python.exe scripts\train_locomotion.py `
    --task Go2Loco-S1-Flat-Blind-v0 --headless --num_envs 2048 --max_iterations 1500 --seed 0

# 评估
.venv311\Scripts\python.exe scripts\eval_policy.py `
    --run-dir logs/rsl_rl/go2_loco_s1_flat/<stamp> --episodes 20

# 训练过程视频
.venv311\Scripts\python.exe scripts\render_training_video.py --log <training.log>
```
