# 使用与复现指南

[返回首页](../README.md) · [架构](ARCHITECTURE.md) · [实验](EXPERIMENTS.md) · [使用指南](USAGE.md)

> 本文档说明环境搭建、演示运行、实验复现、模型评估和自定义任务的完整步骤。
> 所有命令、参数和预期结果均来自项目实际文件（`--help` 输出、`rl/go2w_env.py`、
> `rl/train.py`、各课程脚本与 `reports/` 验收数据），未做任何虚构。
>
> **公开快照说明**：仓库提交精选源码、模型、演示数据和报告，不持续提交完整
> `rl/runs/`。报告中引用的原始训练曲线若不在仓库中，需要按对应脚本重新训练生成。
> 项目定位与结果边界见 [README](../README.md#limitations)。

## 目录

1. [环境搭建](#1-环境搭建)
2. [快速演示](#2-快速演示)
3. [实验复现指南](#3-实验复现指南)
4. [模型评估](#4-模型评估)
5. [自定义任务](#5-自定义任务)
6. [常见问题](#6-常见问题)

---

## 1. 环境搭建

### 1.1 系统要求

- 操作系统：Linux（推荐；已在 Ubuntu 24.04 CPU 环境完成全量测试、MuJoCo/EGL、
  PPO smoke 与评估链路验证）/ macOS / Windows（推荐 WSL2）
- Python：3.10 或 3.12（项目测试环境）
- 硬件：CPU 可运行；在 i7-1185G7 CPU-only 环境，4096 步 smoke 约 29 秒，
  长训练通常需要数小时。实际吞吐受任务、并行环境数和评估频率影响；GPU 可选。
- 磁盘：仓库约 100MB；训练产物（rl/runs）另计，建议预留 500MB 以上
- 无头服务器需要 EGL 渲染：`export MUJOCO_GL=egl`（`scripts/make_demo.sh` 已内置）

macOS 和原生 Windows 的大小写不敏感文件系统可能无法完整表达第三方模型资产中的同名大小写
文件，建议在 WSL2 的 Linux 文件系统内克隆和运行，而不是直接使用 NTFS 工作区。

### 1.2 安装 MuJoCo

本项目通过 pip 安装 MuJoCo（`requirements.txt` 固定 `mujoco==3.11.0`）：

```bash
pip install mujoco
python -c "import mujoco; print(mujoco.__version__)"
```

### 1.3 克隆项目并安装依赖

```bash
git clone https://github.com/Anhao1314/go2w-MoRA-navigation.git
cd go2w-MoRA-navigation
pip install -r requirements.txt
```

依赖清单（来自 `requirements.txt`）：

- `mujoco==3.11.0`
- `numpy>=2.2.6,<3`
- `Pillow>=10.0`、`imageio>=2.34`、`imageio-ffmpeg>=0.6`（视频/GIF 渲染）
- `stable-baselines3==2.9.0`、`gymnasium>=0.29`、`torch>=2.0`
- `tensorboard>=2.18`（训练曲线）、`psutil>=7.0`（监控）

### 1.4 验证安装

```bash
python -c "
from rl.go2w_env import Go2wEnv
env = Go2wEnv(task='traverse_curve', domain_randomize=False)
obs, info = env.reset(seed=0)
print(f'obs_dim={obs.shape}, action_dim={env.action_space.shape}')
"
```

预期输出：`obs_dim=(61,), action_dim=(6,)`（traverse 任务；balance 任务为 44/2）。

---

## 2. 快速演示

### 2.1 双轮自平衡演示

```bash
python scripts/demo_go2w.py
```

预期：机器狗收腿进入双轮姿态并保持平衡约 10.5s，俯仰偏差约 0.002 rad
（对应 `media/go2w_two_wheel_balance.gif` 与 reports/balance 实测）。

### 2.2 生成弯道演示轨迹（System 0 脚本控制器）

```bash
python scripts/gen_demo_trajectory.py --seed 0 --noise 0.03
```

真实 CLI（`--help` 输出）：

- `--seed`：随机种子，默认 `0`
- `--noise`：动作高斯噪声标准差（制造轨迹多样性），默认 `0.03`
- `--attempts`：失败自动重试次数，默认 `10`

输出：`data/demo_trajectories/curve_demo_seed00.npz`（61 维决策前 obs + 6 维低层动作）。
注意：`bc_pretrain.py` 用这套轨迹训练的纯 BC 基线成功率只有 0%（0.276m），
本项目把它作为"单层方法失败"的对照，不是成功控制器。

### 2.3 生成多段导航演示轨迹

```bash
python scripts/gen_demo_multi_segment.py --seeds 20 --tag main
```

真实 CLI（`--help` 输出）：`--seeds`（默认 20）、`--noise`（默认 0.0）、
`--skip-stop`（测试A：A点不停留）、`--start-at-a`（测试B：从A点出发段2）、
`--stage2`（生成段2 A→B 演示）、`--attempts`（默认 1）、`--tag`（输出标签）、
`--plot/--plot-file/--plot-out`（俯视轨迹图）。

输出：`data/demo_trajectories/curve_multi_segment_demo_main_seedXX.npz`（20 条）。
P0 验证结论：20/20 全部走通（`multi_segment_validation_report.md`）。

### 2.4 生成岔路口演示轨迹

```bash
python scripts/gen_demo_junction.py
```

该脚本**没有 CLI 参数**（无 argparse），直接生成 seed00~19 共 20 条：
seed00~09 目标 A（左分支）、seed10~19 目标 B（右分支）。

输出：`data/demo_trajectories/junction_demo_seedXX.npz`，实测 20/20 成功。

---

## 3. 实验复现指南

### 3.1 弯道分层控制（A 步骤，核心成功实验）

**实验目的**：验证 System 0 + System 1 分层控制在弯道导航上的效果；
`speed_scale` 下界 0.9（禁止停车）是关键设计（消融见 `docs/images/curve_high_level_ablation.png`）。

**训练命令**（真实 CLI）：

```bash
python scripts/train_high_level_curve.py \
  --total-steps 2000000 --envs 4 --seed 01 \
  --learning-rate 3e-4 --ent-coef 0.01 \
  --run-dir rl/runs/traverse_curve_high_level/seed01
```

默认值：`--total-steps 2000000`、`--envs 4`、`--seed 0`、`--learning-rate 3e-4`、
`--ent-coef 0.01`、`--eval-freq 50000`、`--eval-episodes 5`、`--run-dir` 指向
`rl/runs/traverse_curve_high_level/seed<seed>`。注意：**没有 `--init-from` 参数**，
BC 预热模型不是通过该脚本加载的（多段/岔路口课程脚本才有 `--bc-model`）。

**预期训练时间**：4 envs 实测约 600 步/s；本实验实际 150k 步即收敛
（eval 点为 50k/100k/150k，全程 100%），通常 10 分钟内完成。

**预期结果**（`reports/traverse_curve_high_level/seed01/` 实测）：

- 成功率：100%
- 平均距离：1.3552m（目标 1.4m）
- 平均 ep_len：643（150k eval 点）
- 摔倒次数：0

### 3.2 10m 长程多段导航（multi_segment v2）

**实验目的**：验证 A→B 两段导航 + 精确停止。P0 修复（docking 奖励 +100、
过冲惩罚、B 点到达即终止）+ P1（stage2 单独 BC 预热）是关键。

#### 前置：演示轨迹 + 两阶段 BC 预热

```bash
# 主任务（段1）20 条演示
python scripts/gen_demo_multi_segment.py --seeds 20 --tag main
# 段2（A→B）20 条演示
python scripts/gen_demo_multi_segment.py --seeds 20 --stage2
# 主任务 BC 预热
python scripts/bc_pretrain_multi_segment.py --scope 0
# 段2 BC 预热
python scripts/bc_pretrain_multi_segment.py --scope 1
```

`bc_pretrain_multi_segment.py` 真实参数：`--epochs`（默认 300）、`--lr`（默认 1e-3）、
`--batch-size`（默认 512）、`--scope`（0=主任务段1，1=段2 A→B）、`--patience`（默认 10）。
输出：`multi_segment_bc_pretrain.zip` / `multi_segment_stage2_bc_pretrain.zip` 及对应 norm。

**训练命令**（真实 CLI；阶段步数在脚本内硬编码为 0.5M/0.5M/1M，**没有**
`--stage*-steps` 参数）：

```bash
python scripts/train_multi_segment_curriculum.py \
  --envs 4 --seed 0 \
  --run-root rl/runs/traverse_curve_multi_segment/seed00_v2
```

可选参数：`--bc-model`、`--bc-norm`、`--stage2-bc-model`、`--stage2-bc-norm`（默认指向
`data/demo_trajectories/` 下对应文件）、`--smoke`（10k 步快速验证）。

**预期训练时间**：约 30 分钟（实际 1M 步：stage1 250k + stage2 250k + stage3 500k，均提前达标）。

**预期结果**（`reports/traverse_curve_multi_segment/seed00_v2/` 实测）：

- 完整成功率：100%（20 episodes）
- A 点精确停止达标率：100%
- 平均距离：10.732m（B 点 10.8m）
- 平均 ep_len：1516 步（15.16s）
- 摔倒次数：0
- stage1/stage2/stage3 均 100%（v2 修复后；首版 v1 为 0%，见 EXPERIMENTS.md）

### 3.3 岔路口决策（junction，System 2 首次验证）

**实验目的**：验证 System 2 规则决策（A→左 / B→右）+ System 1 导航。

#### 前置：演示轨迹 + BC 预热

```bash
python scripts/gen_demo_junction.py        # 20 条（A 10 + B 10）
python scripts/bc_pretrain_junction.py     # 无参数，训练并 20/20 验证
```

输出：`data/demo_trajectories/junction_bc_pretrain.zip` + `junction_bc_vecnorm.pkl`。

**训练命令**（真实 CLI；阶段步数硬编码 0.3M/0.5M/1M）：

```bash
python scripts/train_junction_curriculum.py \
  --envs 4 --seed 0 \
  --run-root rl/runs/traverse_curve_junction/seed00
```

可选参数：`--bc-model`、`--bc-norm`（默认指向 junction BC 文件）、`--smoke`。

**预期训练时间**：约 30 分钟（实际 stage1 250k + stage2 500k + stage3 约 500k）。

**预期结果**（`reports/traverse_curve_junction/seed00/` 实测）：

- 完整成功率：100%（40 episodes，A 20 + B 20）
- 分支决策正确率：100%
- 平均距离：5.0325m（终点 5.5m，容差 0.5m）
- 平均 ep_len：702.5 步（7.02s）
- 摔倒次数：0
- stage1 250k 达标；**stage2 10 个 eval 点全 0%**（人工岔路口起点 yaw=0、v=0，
  分布不匹配）；stage3 完整任务 100%（约 1.008M 提前停止）

**重要发现**：stage2 失败但 stage3 成功，提示课程学习的中间态起始分布可能需要与
策略自然到达该状态时的分布匹配。详见 `docs/EXPERIMENTS.md`。

### 3.4 B+ 随机目标泛化实验（可选）

```bash
python scripts/train_high_level_curve.py \
  --goal-min 0.5 --goal-max 1.4 --use-her \
  --total-steps 2000000 --seed 0 \
  --run-dir rl/runs/traverse_curve_high_level_bplus/seed00
```

真实参数：`--goal-min/--goal-max`（随机目标弧长范围，默认 0.5/1.4）、
`--use-her`（轻量 HER 回调）、`--no-goal-condition`（消融：观测回到 61 维）、
`--eval-fixed-goal`（默认 1.4）、`--her-update-freq/--her-buffer-size/--her-batch-size`
（默认 10000/20000/256）。

预期结果（`reports/traverse_curve_high_level_bplus/seed00/` 实测）：
成功率 90%、平均距离 0.8339m、2 摔倒；无目标条件对照组为 100%/1.3522m。
该历史报告配置记录 HER 为关闭，但末尾结论提到 HER；本节带 `--use-her` 的命令是功能入口，不保证重现该历史配置。

### 3.5 失败实验复现（可选，用于对比）

以下实验在归档弯道任务上未达标，可作为不同控制接口与学习流程的对照：

- **纯 PPO**：`python rl/train.py --task traverse_curve --total-steps 4000000`
  （建议先 `--smoke` 验证管线）→ 站桩 301 步不动或 ≤0.7m
- **BC+PPO 微调**：`python scripts/bc_pretrain.py`（无参数）→ BC 基线 0%；
  微调阶段出现 negative transfer
- **DAgger 5 轮**：`python scripts/dagger_train.py --max-iter 5`
  （默认 `--episodes 20 --epochs-init 50 --epochs-iter 30`）→ 最佳 0.915m 未达标
- **固定课程 4 阶段**：`python scripts/train_curve_curriculum.py` → 不动最优复归

---

## 4. 模型评估

### 当前模型资产

当前快照包含教师轨迹及 `data/demo_trajectories/` 中的 BC `.zip` 模型与对应归一化 `.pkl` 文件；它们用于预热或基线，不是首页成功演示对应的最终 PPO 模型。

首页弯道、多段和岔路口策略的最终训练目录 `rl/runs/` 未提交。查看已有视频不需要模型；重新评估或录像需要先按第 3 节训练，并取得同一 run 的模型、归一化参数和任务配置。不要混用不同观测维度或不同阶段的资产。

`make_demo.sh` 面向单任务 PPO，要求对应 run 中的 `final_model.zip`、`.completed`、`best_model.zip` 与 `best_vec_normalize.pkl`；缺失时会跳过，不会自动训练，也不用于复现首页分层演示。


### 4.1 用已有模型生成验收报告

弯道分层脚本提供 `--report-only`（跳过训练，用 run-dir 已有模型出报告）：

```bash
python scripts/train_high_level_curve.py \
  --report-only \
  --run-dir rl/runs/traverse_curve_high_level/seed01
```

注意：run-dir 需要存在 `best_model.zip` 与 `best_vec_normalize.pkl`
（`rl/runs/` 默认被 gitignore，需先本地训练生成）。
多段/岔路口课程脚本没有 `--eval-only`；训练结束后会自动执行
20/40-episode 正式验收并写入 `reports/` 与 run-dir 的 `eval_log.csv`。

### 4.2 生成演示视频

```bash
# 只用已有模型录制（默认相机 overview=全局俯视）
python scripts/train_high_level_curve.py \
  --record-only \
  --run-dir rl/runs/traverse_curve_high_level/seed01 \
  --record media/rl_traverse_curve_high_level_seed01 \
  --camera overview
```

多段/岔路口课程脚本训练完成后也会自动生成
`media/rl_traverse_curve_multi_segment_seed00_v2.{mp4,gif}` 与
`media/rl_traverse_curve_junction_seed00.{mp4,gif}`。

### 4.3 查看训练曲线

```bash
tensorboard --logdir rl/runs/traverse_curve_high_level/seed01/
```

每个 run 的 `eval_log.csv` 包含各评估点的 `timesteps / mean_reward / mean_ep_len /
success_rate`（多段/岔路口还有 `passed_rate` 或 `correct_rate`）。

---

## 5. 自定义任务

### 5.1 Go2wEnv 参数（`rl/go2w_env.py` 构造函数真实默认值）

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `task` | `"balance"` | `balance`/`full_chain`/`full_chain_simple`/`traverse_flat_slope`/`traverse_slope`/`traverse_curve` |
| `domain_randomize` | `True` | 质量/摩擦/惯性域随机化 |
| `terrain` | `"hfield"` | `hfield` 运行时高度场 / `boxes` 旧盒状斜坡 |
| `heading_penalty` | `1.5` | traverse 航向误差惩罚系数 |
| `corridor_width` | `0.40` | curve 走廊半宽（m） |
| `curve_amplitude` | `0.35` | curve 弯道振幅（m） |
| `reward_version` | `"v4"` | `v4`/`simple`/`v5`/`mseg`/`junction` |
| `fall_penalty` | `-5.0` | 摔倒/早停惩罚 |
| `goal_bonus` | `100.0` | 到达奖励 |
| `goal_arc` | `None` | 目标弧长（`None` 用任务默认终点） |
| `multi_segment` | `False` | 多段导航开关 |
| `segments` | `None` | 多段定义列表 |
| `subgoal_switch` | `False` | 子目标切换开关 |
| `scope` | `2` | `0`=只到A、`1`=从A到B、`2`=完整两段 |
| `junction` | `False` | 岔路口任务开关 |
| `junction_config` | `None` | 岔路口几何配置 |
| `target_goal` | `"A"` | 岔路口目标（A 左 / B 右） |
| `branch_selected` | `None` | 已锁定分支 |

观测/动作维度：traverse 底层 61 维/6 维动作；balance 44 维/2 维；
高层包装器追加目标/进度观测后为 63（B+）、67（多段）、68（岔路口）维，
动作恒为 2 维 `[speed_scale, turn_adjust]`。

### 5.2 奖励版本说明

- `v4`：默认，前进 shaping + 航向/横向惩罚 + 到达 +100
- `simple`：课程学习简洁奖励
- `v5`：可配置摔倒惩罚与到达奖励（`--fall-penalty`/`--goal-bonus`）
- `mseg`：多段专用（A 到达 +50、精确停止 +100、切换 +10、B 到达 +100、
  前进 0.1×dx、航向/横向 shaping）
- `junction`：岔路口专用（到达岔路口 +20、决策正确 +30/错误 -30、终点 +100）

### 5.3 训练新任务示例

```python
from rl.go2w_env import Go2wEnv
from rl.high_level_env_wrapper import HighLevelEnvWrapper

base_env = Go2wEnv(
    task="traverse_curve",
    domain_randomize=False,
    corridor_width=0.50,      # 加宽走廊
    curve_amplitude=0.20,     # 减小弯道
    reward_version="v5",       # 使用 v5 奖励
)
env = HighLevelEnvWrapper(base_env, use_goal_condition=True)
obs, info = env.reset(seed=0)
print(obs.shape, env.action_space.shape)
```

新增奖励版本：在 `rl/go2w_env.py` 的 `_get_reward` 中按 `reward_version`
分支添加，并在 `__init__` 的合法值列表中加入版本名。

---

## 6. 常见问题

### Q1: MuJoCo 安装失败 / 找不到 GLFW

确保系统有 OpenGL 依赖；无头服务器使用 `export MUJOCO_GL=egl`。
`scripts/make_demo.sh` 已内置该变量，训练本身不需要显示环境。

### Q2: 策略学会站桩不动

这是"不动最优"局部最优。确保高层动作 `speed_scale` 下界 ≥0.9（禁止停车）；
`speed_scale≥0.5` 的版本实测 0m 站桩（见 `docs/images/curve_high_level_ablation.png`）。

### Q3: 加载 BC/训练模型后行为异常

检查 norm 与 obs 维度是否匹配：弯道 61 维、多段 67 维、岔路口 68 维；
BC 模型必须搭配对应的 `*_vecnorm.pkl`（`--bc-norm`/`--init-norm`）。

### Q4: 训练速度慢

训练吞吐取决于 CPU、任务、`n_steps`、环境数和评估开销，不能用 README 中的单个
速度数字作为预算。可逐步调高 `--envs`（内存相应增加），但应先记录一次本机 smoke
的墙钟时间和峰值内存，再估算完整训练；先跑 `--smoke` 验证配置。

### Q5: 演示轨迹生成失败

脚本控制器对噪声/几何敏感。`gen_demo_trajectory.py` 默认 `--noise 0.03`、
`--attempts 10` 自动重试；多段/岔路口演示已实测 20/20 走通，失败多为改了
环境几何（先恢复默认 `JUNCTION_DEFAULT`/`segments` 再跑）。

### Q6: 如何查看训练中的实时状态

```bash
python rl/webpanel.py --host 127.0.0.1 --port 8787
```

浏览器打开 <http://127.0.0.1:8787> 查看训练进度/资源/报告；训练曲线另可用
`tensorboard --logdir rl/runs/`。

---

## 开发检查

在安装依赖后，从仓库根目录运行：

```bash
python -m unittest discover -s tests -p 'test_*.py'
python -m pip install pyright
pyright rl scripts mujoco_demos
```

静态检查工具单独安装；测试与检查结果取决于当前版本和环境，应以实际输出为准。轨迹生成会写入样例目录，重复运行前请备份需要保留的数据。

## 参考文档

- [架构设计](ARCHITECTURE.md) — MoRA 三层架构、观测空间演进、奖励设计
- [实验总览](EXPERIMENTS.md) — 所有实验结果、失败教训、研究方法论
- [第三方许可](THIRD_PARTY_NOTICES.md) — 宇树模型与地形算法许可声明
