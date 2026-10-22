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
| `.venv311`（Python 3.11，仓库内已 gitignore） | **Isaac Lab 训练**：`locomotion/isaac/` 只能在此环境导入 | Isaac Lab 2.7.0（isaaclab 0.54.2 可编辑安装指向 `D:\work\IsaacLab`）、Isaac Sim 5.1.0.0 |

`.venv311` 的版本约束（已由 2026-10-22 的 M1 冒烟训练验证）：

| 包 | 锁定版本 | 原因 |
| --- | --- | --- |
| `rsl-rl-lib` | **==3.1.2** | Isaac Lab 2.7.0 `source/isaaclab_rl/setup.py` 的硬要求；装 5.x 会在 `OnPolicyRunner` 初始化时抛 `KeyError: 'actor'` |
| `packaging` | **==23.0** | `isaacsim-core 5.1.0.0` 硬要求（`isaaclab-rl` 要求 `<24`）；装 25.x 会破坏 isaacsim |
| `torch` | 2.7.0+cu128 | 与 Isaac Sim 5.1 驱动栈匹配 |
| `gymnasium` | 1.2.1 | Isaac Lab 2.7.0 官方要求 |

安装或升级依赖后请重跑 `python scripts/eval_protocol.py --verify` 与一次 2-iteration
冒烟训练（见下），确认链路仍可用。

CI（GitHub Actions）只有 pip 环境、没有 Isaac Sim，因此 `locomotion/isaac/`
的测试必须用环境守卫跳过；`locomotion/eval/` 与 `baselines/scripted_pd.py`
保持纯 Python 可测。

## 快速开始（M1 开跑时）

```powershell
# 训练（S1）：wrapper 先注册 Go2Loco-*，再交给 Isaac Lab 的 RSL-RL 启动器
$env:PYTHONPATH = "$PWD"
$env:OMNI_KIT_ACCEPT_EULA = "YES"    # 首次运行已由你确认过 NVIDIA Omniverse EULA 时
.venv311\Scripts\python.exe scripts\train_locomotion.py `
    --task Go2Loco-S1-Flat-Blind-v0 --headless --num_envs 2048 --seed 0

# 协议自检（任意环境）
python scripts/eval_protocol.py --verify
```

**导入顺序契约（重要）**：`locomotion.isaac` 只做字符串形式的 Gymnasium 注册，
在 Kit 启动前导入是安全的；但它指向的 env cfg 依赖 `pxr`（USD），而 `pxr` 由
Isaac Sim 的 Kit 运行时提供——只有 `AppLauncher` 启动后才可导入。因此**不要**在
未启动 Kit 的进程里直接 import `locomotion.isaac.flat_env_cfg`；请通过
`scripts/train_locomotion.py` 或先启动 `AppLauncher` 再导入。
`ISAACLAB_ROOT` 可指定 Isaac Lab 检出位置（默认探测 `../IsaacLab`、`D:\work\IsaacLab`）。

## 训练产物与证据束

Isaac Lab 把每次训练写到 `logs/rsl_rl/<experiment_name>/<timestamp>/`（已 gitignore）。
`scripts/train_locomotion.py` 按协议 §7 自动生成证据束：

- **训练前**：`logs/rsl_rl/_pending/<run-id>.json` —— git SHA、dirty 标记、依赖版本
  （isaaclab / isaacsim / rsl-rl-lib / torch / gymnasium）、完整命令行、task、seed
- **训练后**：该 run 目录下的 `run_config.json`（由 pending 记录定稿）+ 
  `sha256_manifest.json`（目录内每个文件的 SHA-256）

训练被中断时 pending 记录保留，可手动补定稿：

```powershell
.venv311\Scripts\python.exe scripts\run_evidence.py finalize --run-id <run-id>
```

入库存档时把 `run_config.json`、评估报告和 manifest 一起提交到 `reports/`；
`logs/` 本身不入库。
