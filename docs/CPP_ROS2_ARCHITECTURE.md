# C++ / ROS2 System-0 架构

**状态：** P0 已实现（C++ 控制核心 + Python 适配器）；P1（ROS2 垂直切片）尚未实现。

本文说明 P0 的边界、模块职责和后续 ROS2 接入点。所有数字与限制见
[VALIDATION.md](VALIDATION.md) 和 [CPP_STACK_PROTOCOL.md](CPP_STACK_PROTOCOL.md)。

## 1. 分层与频率

```text
S2  Mission Planner（现有 Python 规则/任务协议）
        │  2D high-level action / mission command
        ▼
S1  High-level policy（现有 PPO 或脚本 ControllerPolicy，100 Hz）
        │  [speed_scale, turn_adjust]
        ▼
S0  C++ controller（P0：100 Hz 调用；P1：ROS2 节点 500 Hz）
        │  6D normalized action（LEGACY_6D）
        ▼
MuJoCo Go2wEnv（500 Hz physics，5 substeps per policy step）
```

| 层 | P0 实现 | P1 目标 |
| --- | --- | --- |
| S2 | 现有规则/任务脚本 | `go2w_mission` 节点，GoalSpec/MissionStatus |
| S1 | 现有 PPO / `ControllerPolicy` | `go2w_s1` 节点（脚本优先，后续 ONNX） |
| S0 | C++ core 通过 pybind11 注入 `Go2wEnv` | `go2w_control` + `go2w_sim` ROS2 节点，500 Hz |
| State estimation | 直接读 MuJoCo 状态 | `go2w_state_estimation`（EKF/odometry） |
| Physics | Python `Go2wEnv` | C++ MuJoCo C API 节点 |

## 2. C++ core 的边界

`cpp/` 是 **ROS-agnostic、MuJoCo-runtime-agnostic** 的库：

- 不包含 ROS2 头文件或节点代码。
- 不链接 MuJoCo；MuJoCo 只用于 Python 夹具生成和 `Go2wEnv`。
- 不读取环境私有状态；所有输入通过 `RobotState` / `HighLevelCommand`。
- 不依赖 Python；pybind11 绑定是单独的薄层。

```text
cpp/
  include/go2w/
    types.hpp              # 状态/命令/输出类型
    model.hpp              # 模型常量与名称映射
    kinematics.hpp         # FK / IK / Jacobian
    dynamics.hpp           # 质心动力学
    mpc.hpp                # SRBD 线性 MPC
    force_mapper.hpp       # 接触力 → 轮力矩
    path_tracker.hpp       # 脚本教师 + 预减速
    safety.hpp             # 安全状态机
    controller.hpp         # 顶层 S0
    generated/
      model_constants.hpp  # 由 MJCF 生成，禁止手改
  src/                     # 对应实现
  bindings/pybind_module.cpp
  tests/                   # Catch2 + MuJoCo 数值夹具
  apps/                    # replay / MPC benchmark
```

## 3. 模型常量与顺序

MJCF 中的顺序不一致，必须由名称解析：

| 实体 | MuJoCo 顺序 | canonical 顺序 |
| --- | --- | --- |
| Actuator | FR, FL, RR, RL legs, then wheels FR, FL, RR, RL | 同 actuator |
| qpos | FL, FR, RL, RR body order | FR, FL, RR, RL |
| qvel | FL, FR, RL, RR | FR, FL, RR, RL |

- `scripts/gen_model_constants.py` 生成 C++ 头文件和 `rl/go2w_constants.py`。
- `tests/test_model_constants_sync.py` 防止生成文件漂移。
- C++ 所有索引来自 `constants::kLegs`，不假设数组顺序。

## 4. P0 Python 集成

```python
from rl.cpp_s0_controller import CppS0Controller
controller = CppS0Controller(env, pre_brake=True)
wrapper = HighLevelEnvWrapper(base_env, controller=controller, ...)
```

- `CppS0Controller` 实现 wrapper 的 duck-typed 接口：
  - `get_state_from_env(env) -> (x, y, yaw)`
  - `compute_action(x, y, yaw, cmd) -> np.ndarray[6]`
- `HighLevelEnvWrapper.reset()` 调用 `controller.reset(base_env)`（若存在）。
- `scripts/evaluate_navigation.py --s0 cpp --pre-brake` 可直接运行 C++ S0。
- `Go2wEnv.step` 的观测/奖励/终止逻辑未改动。

### 关键状态提取约定

| 量 | 来源 | 说明 |
| --- | --- | --- |
| base position | `data.body("base_link").xpos` | 世界系 |
| base quaternion | `data.xquat[base_body_id]` | **归一化 wxyz**；转 xyzw 给 C++ |
| leg qpos | `LEG_QPOS[leg][0:3]` | FR/FL/RR/RL canonical |
| wheel qpos | `WHEEL_QPOS[leg]` | FR/FL/RR/RL canonical |
| speed | `Go2wEnv._body_vel()[0]` | 车体系 x 速度 |

不要用 `qpos[3:7]` 直接算 yaw：数值积分后的四元数可能未归一化，
与 `Go2wEnv._yaw_of()` 会出现 ~1e-5 的差异，足以让轨迹漂移。

## 5. 控制模式

| 模式 | 输出 | 状态 |
| --- | --- | --- |
| `LEGACY_6D` + scripted | 6D normalized action，教师 + 预减速 | P0 默认，已验证 |
| `LEGACY_6D` + MPC | 6D action，MPC 接触力 → 轮力矩 | MPC core 已验证；闭环集成留 P1 |
| `PHYSICAL_VELOCITY` | 16 维力矩 | P1，ROS2 sim 节点 |

P0 的 scripted 模式逐元素复现 Python `MultiSegmentLowLevelController`；
`pre_brake=True` 时复现归档脚本的 A/B 点预减速行为。

## 6. P1 ROS2 垂直切片（未实现）

```text
ros2_ws/src/
  go2w_msgs/            # GoalSpec, HighLevelCommand, JointTorqueCommand, ...
  go2w_sim/             # MuJoCo C API, 500 Hz physics
  go2w_control/         # C++ S0 node
  go2w_state_estimation/# EKF/odometry
  go2w_mission/         # S2 GoalSpec
  go2w_s1/              # Python rclpy scripted S1
  go2w_bringup/         # launch/config
  go2w_tests/           # launch_testing
```

关键接口：

| 话题 | 类型 | 频率 |
| --- | --- | --- |
| `/go2w/sim/state` | RobotState | 500 Hz |
| `/go2w/sim/imu` | sensor_msgs/Imu | 500 Hz |
| `/go2w/control/high_level_command` | HighLevelCommand | 100 Hz |
| `/go2w/control/torque` | JointTorqueCommand | 500 Hz |
| `/go2w/mission/goal_spec` | GoalSpec | event |
| `/go2w/mission/progress` | MissionStatus | 10 Hz |

P1 必须补：

- C++ MuJoCo sim 节点（500 Hz，确定性 `realtime_factor=0` 用于 CI）。
- EKF（不能偷看 `framequat` 真值）。
- 16 维力矩接口与接触力 → 腿关节前馈。
- ROS2 launch testing / Docker（macOS 用 OrbStack 或 CI Ubuntu）。

## 7. 与现有仓库的关系

- 保留全部 Python 任务、奖励、评估和证据协议。
- Python S0 仍可用于训练/对照；C++ S0 是可选的 `--s0 cpp`。
- `pre_brake` 是显式开关，默认关闭，避免改变历史协议。
- P0 的 C++ 结果与 scripted 结果分开记录，不追认历史报告。
