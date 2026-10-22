# locomotion/ — Phase-2：Unitree Go2 关节级 RL locomotion（M1/M2）

本目录是仓库**当前活跃阶段**的唯一代码归宿，执行
[`docs/EXPERIMENT_DESIGN.md`](../docs/EXPERIMENT_DESIGN.md)（v1.0 预注册协议，
2026-10-22 冻结）。Phase-1（Go2W 轮足导航，`rl/`、`scripts/` 等）已完成归档、
原地冻结，见[仓库布局契约](../docs/ARCHITECTURE.md)。

## 协议 → 代码对应表

| 协议条目（§） | 归宿 |
| --- | --- |
| 评估种子集 E = {0..19}（§3.1） | [`eval/seeds.py`](eval/seeds.py)（单一事实来源） |
| 指标 vx_rmse / ωz_rmse / fall_rate / recovery_time / success_rate / energy-per-meter（§3.2） | [`eval/metrics.py`](eval/metrics.py)（纯 Python，两侧评估共用） |
| 验收阈值常量 + 自检 CLI（§4） | [`scripts/eval_protocol.py`](../scripts/eval_protocol.py)（协议指名路径） |
| S1 平地速度跟踪（§5） | [`isaac/flat_env_cfg.py`](isaac/flat_env_cfg.py)，任务 ID `Go2Loco-S1-Flat-Blind-v0` |
| S2 扰动：100 N/0.5 s 推、±3 kg、摩擦（§5） | [`isaac/disturbance_env_cfg.py`](isaac/disturbance_env_cfg.py)，`Go2Loco-S2-Disturbance-v0` |
| S2b 轻地形（§5） | 同上文件，`Go2Loco-S2b-LightTerrain-v0` |
| S3 楼梯课程（§5，预算门控） | 未建（协议标注 budget-gated，开跑前再建） |
| S4 Sim2Sim（§5，RQ4） | [`sim2sim/`](sim2sim/README.md)（设计占位） |
| C0 scripted PD 基线（§6） | [`baselines/scripted_pd.py`](baselines/scripted_pd.py) |
| C1 convex-MPC 基线（§6） | [`baselines/convex_mpc/`](baselines/convex_mpc/README.md)（待许可审查，无代码） |
| run 证据束：run_config.json / events.jsonl / SHA-256（§7） | 沿用仓库既有约定（见 [`docs/EXPERIMENT_DATA.md`](../docs/EXPERIMENT_DATA.md)） |

## 双环境说明

本仓库同时存在两套 Python 环境，职责不同、不要混用：

| 环境 | 用途 | 关键依赖 |
| --- | --- | --- |
| 根 `requirements.txt`（Python 3.10/3.12） | Phase-1 MuJoCo/SB3 代码 + CI + 本包的 `eval/`、`baselines/` 单测 | mujoco、stable-baselines3、numpy |
| `.venv311`（Python 3.11，仓库内已 gitignore） | **Isaac Lab 训练**：`locomotion/isaac/` 只能在此环境导入 | Isaac Lab 2.7.0（isaaclab 0.54.2 可编辑安装指向 `D:\work\IsaacLab`）、Isaac Sim |

CI（GitHub Actions）只有 pip 环境、没有 Isaac Sim，因此 `locomotion/isaac/`
的测试必须用环境守卫跳过；`locomotion/eval/` 与 `baselines/scripted_pd.py`
保持纯 Python 可测。

## 快速开始（M1 开跑时）

```powershell
# 训练（S1，.venv311 环境，仓库根目录在 PYTHONPATH 上）
$env:PYTHONPATH = "$PWD"
.venv311\Scripts\python.exe D:\work\IsaacLab\scripts\reinforcement_learning\rsl_rl\train.py `
    --task Go2Loco-S1-Flat-Blind-v0 --headless --num_envs 2048

# 协议自检（任意环境）
python scripts/eval_protocol.py --verify
```

## runs/ 产物

训练产物写入 `locomotion/runs/`（已 gitignore），入库存档走既有 evidence
collector 流程（`scripts/collect_run_evidence.py`），不在本目录直推。
