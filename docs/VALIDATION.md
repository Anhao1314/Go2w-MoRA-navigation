# P0 验证报告

**范围：** C++ 控制核心、Python 适配器、多段任务集成、MPC 核心。
**不包含：** ROS2、实机、视觉/触觉、完整 WBC、学习型 S1。

## 1. 数值夹具

| 夹具 | 样本数 | 来源 |
| --- | ---: | --- |
| `fk.csv` | 100 | MuJoCo `mj_kinematics` 轮心 `xpos` |
| `jac.csv` | 100 | MuJoCo `mj_jac` 轮心 `jacp` |
| `centroidal.csv` | 50 | MuJoCo `mj_comPos` + body inertials |
| `parity.csv` | 1920 | Python `MultiSegmentLowLevelController` 动作 |

夹具生成：`scripts/gen_cpp_fixtures.py`，固定 MuJoCo 3.11.0、seed 20261019。

## 2. C++ 测试

命令：

```bash
ctest --test-dir build/core --output-on-failure
```

结果：**13 个 test cases、43 个 assertions，全部通过**。

| 测试 | 容差/结果 |
| --- | --- |
| FK 轮心 | max error <1e-9 |
| 轮心 Jacobian | max error <1e-6 |
| IK 往返 | ≥190/200 目标成功，误差 <1e-5 |
| 质心质量/COM | <1e-9 |
| 质心惯性 | <1e-8 |
| 教师 parity | max error ~3e-8（float32 舍入），阈值 1e-6 |
| MPC hover | 总垂直力误差 ~2.15%，摩擦/力限满足 |
| MPC 前向参考 | 净前向力 >0 |
| MPC benchmark | 200/200 求解，mean 3.16 ms，p99 4.07 ms（归档运行） |
| 安全层 | 低高度 ESTOP、QP 连续失败 DEGRADED、动作过滤 |

## 3. MPC benchmark

命令：

```bash
./build/core/go2w_mpc_bench
```

结果（macOS arm64，Apple Silicon，Release）：

```json
{
  "iterations": 200,
  "solved": 200,
  "mean_ms": 3.161866,
  "p50_ms": 3.117875,
  "p95_ms": 3.303791,
  "p99_ms": 4.073791,
  "max_ms": 5.037166,
  "horizontal_hover_vertical_force_n": 183.587062,
  "weight_n": 187.630062,
  "vertical_force_error_pct": 2.154772
}
```

完整归档：`reports/cpp_stack/mpc_bench.json`。不同运行之间 mean/p99
会有约 10% 波动。

## 4. Python 集成

命令：

```bash
python scripts/evaluate_navigation.py \
  --task multi-segment --method controller --episodes 10 \
  --s0 cpp --pre-brake --output reports/cpp_stack/p0_cpp_prebrake
```

| 配置 | 成功率 | A 停靠 | mean_x | mean_ep_len | mean_time | failed terminations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scripted，无预减速（历史基线） | 0/10 | 10/10 | 4.7811 | 630.0 | 6.292 s | 10 |
| scripted，预减速 | 10/10 | 10/10 | 10.6571 | 1399.0 | 13.990 s | 0 |
| **C++ S0，预减速** | **10/10** | 10/10 | **10.6571** | **1399.0** | **13.990 s** | **0** |

C++ 与 scripted 预减速的指标完全相同；seed 1000 的逐步轨迹也完全一致。
演示视频：`media/cpp_stack_p0_demo.mp4`（1399 步，13.99 s，goal=true）。

**注意：** 无预减速的 0/10 是现有 `ControllerPolicy` 基线的行为，不是 C++
引入的回归。预减速是归档多段脚本验证中已有的行为，P0 将其显式化为
`pre_brake` 开关。

## 5. Python 测试

```bash
python -m unittest discover -s tests -p 'test_*.py'
pyright --pythonpath .venv/bin/python rl scripts mujoco_demos
```

- 全量 unittest 通过（含新增 `test_cpp_binding.py`、`test_model_constants_sync.py`）。
- Pyright：0 errors。

## 6. 限制

1. 仿真-only；无实机或 sim-to-real。
2. P0 的 MPC 未闭环接入 `Go2wEnv`；它验证了求解、约束和 hover 力分配。
3. 现有 6D 接口只执行轮力矩；侧向/垂直力未执行。
4. 无 EKF、视觉、触觉、语言；使用 MuJoCo 特权状态。
5. 10 个 episode 是固定场景的重复评估，不是独立训练种子或场景泛化。
6. C++ 与 scripted 的一致性只覆盖多段任务和现有控制器接口。
7. 历史报告的 20/20 不被追认为新栈结果；P0 结果单独归档。
