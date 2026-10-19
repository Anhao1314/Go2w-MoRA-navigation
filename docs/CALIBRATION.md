# C++ 控制栈标定记录

所有符号/顺序约定必须有探针或夹具支撑。本文记录 P0 已执行的标定。

## 1. 顺序映射

| 探针 | 方法 | 结论 |
| --- | --- | --- |
| actuator vs qpos | `mj_name2id` + 生成常量 vs MJCF | actuator 顺序 FR/FL/RR/RL，qpos 顺序 FL/FR/RL/RR；禁止按索引假设对应 |
| canonical 映射 | 逐 leg 打印 qpos/dof/actuator | FR=(11,12,13,14)/(10..13)/(0,1,2,12)，其余见 `model_constants.hpp` |
| 生成同步 | `tests/test_model_constants_sync.py` | 生成文件必须与当前 MJCF 一致 |

**规则：** C++ 和 Python 都只使用 `LEG_QPOS` / `LEG_DOF` / `LEG_ACT` 表。

## 2. 四元数与 yaw

| 探针 | 观察 | 处理 |
| --- | --- | --- |
| `qpos[3:7]` vs `xquat` | 数值积分后 `qpos` 四元数可能未归一化；一步后 yaw 差 ~1.56e-5 | 适配器使用归一化的 `data.xquat[base_body_id]` |
| 四元数顺序 | MuJoCo 为 wxyz；Eigen 为 xyzw | 适配器显式重排 |
| yaw 公式 | `atan2(2(wz+xy), 1-2(y²+z²))` 与 `atan2(R10,R00)` 等价 | 使用 `xquat` 后与 `Go2wEnv._yaw_of()` 一致 |

修正后，C++ 与 Python 控制器的动作逐元素一致；多段 episode 的
步数、时间、位置完全相同（见 [VALIDATION.md](VALIDATION.md)）。

## 3. 自由关节 Jacobian 约定

| 探针 | 方法 | 结论 |
| --- | --- | --- |
| base 角速度列 | 用 yaw=90° 状态比较 `mj_jac` 与候选公式 | MuJoCo 自由关节角速度在 **body frame** |
| 正确公式 | `J_ang = -skew(p - p_base) * R_base` | 与 `mj_jac` 的 base 列一致 |
| leg 列 | 解析 `axis × (p - anchor)` | 与 `mj_jac` 误差 <1e-6 |
| 每腿 Jacobian | base 块依赖轮心，不能四腿共用 | API 为 `contactJacobianWorld(state, leg)` |

## 4. 轮力矩符号

| 探针 | 方法 | 结果 |
| --- | --- | --- |
| 四轮 +1 Nm | `Go2wEnv` 执行 6D action `[+1,+1,+1,+1,0,0]`，100 步内 | `vx=+0.0434 m/s`，x 增加 |
| 四轮 -1 Nm | 同上，符号取反 | `vx=-0.0374 m/s`，x 减少 |

**结论：** 正轮力矩产生 +x 方向运动；`ForceMapper` 的
`tau = +r_w * f_x` 方向正确。

## 5. 教师 parity

| 探针 | 方法 | 结果 |
| --- | --- | --- |
| scripted teacher | 1920 组随机 x/y/yaw/命令，C++ vs Python | max error 2.97e-8（float32 舍入），阈值 1e-6 |
| 预减速 | Python/C++ 同一 `pre_brake` 开关 | 动作逐元素一致；episode 指标完全相同 |
| 端到端 | seed 1000 完整多段 episode | 630 步、6.292 s、x=4.781105（无预减速）完全一致 |

## 6. 未完成的标定

- MPC 接触力 → 真实轮地力的闭环验证（P0 只验证 hover 总力与摩擦约束）。
- 侧向/垂直接触力到腿关节力矩的映射（P1）。
- IMU 噪声、轮速噪声、接触检测（P1 EKF）。
- 实机传感器/执行器时延（不在当前范围）。

## 7. 复现

```bash
python scripts/gen_model_constants.py
python scripts/gen_cpp_fixtures.py
cmake -S cpp -B build/core -G Ninja -DCMAKE_BUILD_TYPE=Release -DGO2W_BUILD_TESTS=ON
cmake --build build/core
ctest --test-dir build/core --output-on-failure
```
