# A/B 对比报告：单层 vs 分层（L1 任务）

- **日期**：2026-10-23
- **任务**：`Go2Loco-Sequence-v0`（L1：前进 1.2 m → 停止并保持 0.5 s）
- **评估**：冻结种子 E={0..19}，20 episodes，同一停止判据
- **对比组**
  - **A 单层**：53 维观测 → 12 维关节，PPO
  - **B 分层**：53 维观测 → 2 维速度指令 → 冻结 S0 locomotion 策略（M1 seed 0）→ 12 维关节

两组使用**相同的环境、奖励函数、观测、PPO 超参、迭代数（1500）和种子**。

## 1. 最终性能（1500 迭代的 checkpoint）

| 指标 | A 单层 | B 分层 |
| --- | --- | --- |
| 成功率 | **1.00（20/20）** | **1.00（20/20）** |
| 平均完成步数 | 563.95（11.3 s） | **97.8（2.0 s）** |
| 摔倒率 | 0.00 | 0.00 |
| 训练末 reward | 42.95 | 50.95 |

**结论**：L1 上两者都完全解决（符合 H2：分层不是银弹）。完成效率上分层快 **5.8 倍**，
但这主要反映"2 维指令直接给出全速前进/停止"的动作抽象优势，不是成功率差异。

## 2. 样本效率（关键结果）

对训练中间 checkpoint 逐个评估同 20 个种子：

| 迭代 | A 单层成功率 | B 分层成功率 |
| --- | --- | --- |
| 100 | **0.00** | **1.00** |
| 200 | 1.00 | 1.00 |
| 300 | 1.00 | 1.00 |
| 500 | 1.00 | 1.00 |
| 700 | 1.00 | 1.00 |
| 1000 | 1.00 | 1.00 |
| 1499 | 1.00 | 1.00 |

曲线：`media/training/sweep_ab_comparison.png`

**结论**：分层在第 100 迭代内达到 100%，单层需要 200 迭代（100 迭代时为 0%）。
按已采样的点，**分层达到同等成功率所需训练量约为单层的一半**。

局限必须写明：

1. 采样粒度粗（100 迭代一档），两组的真实收敛点分别在 (0,100] 与 (100,200] 区间内，
   "2 倍"是上界估计，精确值需要 25/50/75 与 125/150/175 的细扫。
2. 每个训练条件只有 **1 个训练种子**，没有跨种子方差；结论是单次观测。
3. 分层 B 的 S0 来自 M1 的预训练策略，其训练成本（约 1200 秒）没有计入 B 的训练预算。
   公平的比较是"B 的总成本 = S0 训练 + S1 训练"，而 A 只有一次训练。

## 3. 对假设的支持程度

| 假设 | 证据 | 判定 |
| --- | --- | --- |
| H1 分层样本效率更高 | 100 迭代内 100% vs 需要 200 | **部分支持**（单种子、粗粒度） |
| H2 简单任务上无差异 | 最终成功率都是 1.00 | **支持** |
| H3 分层失败可归因 | 尚未在 L2/L3 触发失败 | 未检验 |
| H4 接口损失可量化 | 未测量 S0 跟踪误差对整体上限的影响 | 未检验 |

## 4. 复现

```powershell
$env:PYTHONPATH = "$PWD"; $env:OMNI_KIT_ACCEPT_EULA = "YES"

# A 组
.venv311\Scripts\python.exe scripts\train_locomotion.py --task Go2Loco-Sequence-v0 `
    --headless --num_envs 2048 --max_iterations 1500 --seed 0

# B 组
.venv311\Scripts\python.exe scripts\train_layered.py --task Go2Loco-Sequence-v0 `
    --s0-checkpoint logs/rsl_rl/go2_loco_s1_flat/2026-10-23_01-15-02/model_1499.pt `
    --num_envs 2048 --max_iterations 1500 --seed 0

# 样本效率扫描
.venv311\Scripts\python.exe scripts\sweep_checkpoints.py --run-dir <run> --out <json> --plot <png>
```

## 5. 证据

| 内容 | 位置 |
| --- | --- |
| A 组训练 + 报告 | `logs/rsl_rl/go2_loco_sequence/2026-10-23_15-26-53/`、`reports/training/2026-10-23_15-26-53.md` |
| B 组训练 + 报告 | `logs/rsl_rl_layered/2026-10-23_16-55-02/`、`reports/training/2026-10-23_16-55-02.md` |
| A 组扫描 | `reports/training/sweep_single_a.json` |
| B 组扫描 | `reports/training/sweep_layered.json` |
| 对比曲线 | `media/training/sweep_ab_comparison.png` |
| 运动视频 | `media/training/robot_motion_l1_success.mp4` |

## 6. 下一步

1. **细扫收敛点**（25/50/75 与 125/150/175），把"2 倍"从估计变成测量。
2. **补训练种子**（A/B 各 2 个额外种子），给出跨种子方差。
3. **升到 L2/L3**（加转向、加长距离），检验 H3/H4——那里才可能出现真正的失败归因差异。
