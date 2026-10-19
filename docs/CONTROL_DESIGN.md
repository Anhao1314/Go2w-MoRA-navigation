# C++ 控制核心设计

本文记录 P0 C++ S0 的模型、公式和实现约定。所有数值验证见
[VALIDATION.md](VALIDATION.md)。

## 1. 模型与符号

- 机器人：Unitree Go2W，浮动基座 + 4 条 3-DoF 腿 + 4 个轮关节。
- 自由度：`nq=23`（free 7 + 16 hinge），`nv=22`（free 6 + 16 hinge）。
- 执行器：16 个力矩电机（12 腿 + 4 轮）。
- 总质量：19.126408 kg。
- 轮半径：0.086 m。
- 物理步长：0.002 s（500 Hz）；P0 策略步 0.01 s（100 Hz，5 substeps）。

### 1.1 canonical 顺序

| Leg | 枚举 | qpos（hip/thigh/calf/wheel） | dof | actuator |
| --- | --- | --- | --- | --- |
| FR | 0 | 11,12,13,14 | 10,11,12,13 | 0,1,2,12 |
| FL | 1 | 7,8,9,10 | 6,7,8,9 | 3,4,5,13 |
| RR | 2 | 19,20,21,22 | 18,19,20,21 | 6,7,8,14 |
| RL | 3 | 15,16,17,18 | 14,15,16,17 | 9,10,11,15 |

这些表由 `scripts/gen_model_constants.py` 生成，测试锁定。

### 1.2 关节轴与几何

- hip：绕 base 的 x 轴，范围 ±1.0472 rad。
- thigh/calf/wheel：绕父 body 的 y 轴。
- 每条腿的平移链：
  - hip：`(±0.1934, ±0.0465, 0)`
  - thigh：`(0, ±0.0955, 0)`
  - calf：`(0, 0, -0.213)`
  - wheel：`(0, 0, -0.2264)`
- 左右腿的 y 偏移符号不同；body 轴向不镜像。

## 2. 运动学

### 2.1 浮动基座 FK

对每个 body：

```text
T_base = Translation(p) * R(q)
T_child = T_parent * Translation(body_pos) * R_axis(q_joint)
```

对于 wheel body，轮关节旋转不改变轮心位置，但影响轮 body 的惯性姿态。

### 2.2 轮心 Jacobian

对每条腿的三个关节 `k=0..2`：

```text
T_pre = T_parent * Translation(body_pos_k)
axis_k = R(T_pre) * axis_local
anchor_k = t(T_pre)
J_k = axis_k × (p_wheel - anchor_k)
```

`wheelJacobianBase` 返回 3×3。`contactJacobianWorld(state, leg)` 返回该轮的 3×22：

- 基座线性列：`I`。
- 基座角速度列：`-skew(p - p_base) * R_base`。
  - MuJoCo 自由关节的角速度在 **body frame**；该约定由 `jac.csv` 夹具验证。
- 该腿三关节列：上式；轮关节列为 0（轮心在轮轴上）。

`contactJacobianWorldAll` 堆叠四条腿为 12×22，供后续 WBC 使用。

### 2.3 IK

P0 使用阻尼最小二乘（DLS）：

```text
dq = (J^T J + λI)^{-1} J^T (p_target - p_current)
q += dq,  q clipped to joint limits
```

- 初值：stand pose `[0, 0.67, -1.3]`。
- 最多 60 次迭代，位置容差 1e-8。
- 测试覆盖 200 个随机目标，≥190 个成功，最终误差 <1e-5。
- 解析 2-link IK 可在 P1 加入；当前 DLS 足够验证 FK/Jacobian。

## 3. 质心动力学

对每个 body：

```text
p_i = T_i * ipos_i
R_i = R(T_i) * R(iquat_i)
I_i_world = R_i * diag(inertia_i) * R_i^T
com = Σ m_i p_i / M
I_com = Σ [I_i_world + m_i (|r_i|² I - r_i r_i^T)],  r_i = p_i - com
```

- `com_vel` 由 base 速度和各关节 `axis × (p_i - anchor) * qd` 累加。
- 对 MuJoCo `mj_comPos` / body inertials 做数值对照：
  - 质量/COM 误差 <1e-9
  - 惯性张量误差 <1e-8
- P0 **不实现完整 RNEA/浮基 WBC**；不对外声称完整逆动力学。

## 4. SRBD 线性 MPC

### 4.1 状态与输入

```text
x = [p_com(3), v_com(3), theta(3), omega(3)] ∈ R^12
u = [f_1(3), f_2(3), f_3(3), f_4(3)] ∈ R^12
```

`theta = [roll, pitch, yaw]`，`f_i` 是地面作用在机器人上的世界系接触力。

### 4.2 连续模型

```text
p_dot = v
v_dot = (1/m) Σ f_i + g
theta_dot = omega
omega_dot = I_c^{-1} Σ (r_i - p_com) × f_i
```

### 4.3 离散线性化

```text
x_{k+1} = A x_k + B u_k + c
A = I + dt * A_c
B = dt * B_c
c = [0, 0, 0, 0, 0, -g*dt, 0, ...]^T
```

其中：

- `A_c[0:3, 3:6] = I`，`A_c[6:9, 9:12] = I`。
- `B_c[3:6, 3i:3i+3] = (1/m) I`。
- `B_c[9:12, 3i:3i+3] = I_c^{-1} skew(r_i - p_com)`。
- 忽略 `omega × I_c omega` 二阶项和姿态-角速度耦合。
- 每次 solve 用当前 `com`、`r_i`、`I_c` 重新线性化。

### 4.4 QP

```text
min Σ ||x_k - x_ref||_Q² + ||u_k||_R² + ρ||s_k||² + ||x_N - x_N_ref||_QN²
s.t. x_0 = x_meas
     x_{k+1} - A x_k - B u_k = c
     |f_i,x| <= (μ/√2) f_i,z
     |f_i,y| <= (μ/√2) f_i,z
     0 <= f_i,z <= f_z,max
     x_min - s <= x_k <= x_max + s,  s >= 0
```

- 状态约束只作用于 `p_z`, `roll`, `pitch`；slack 同样只有 3 维。
- 摩擦系数 `μ=0.6`（模型为 0.8，保守）。
- `f_z,max=150 N`；摩擦锥与 `f_z,max` 隐含 `||f_i||` 约 <175 N，
  低于配置中的 `f_max=200 N`，因此 P0 未单独加入范数约束。
- `N=20`，`dt=0.02 s`（50 Hz，0.4 s horizon）。
- 权重：位置 `[1,1,10]`，速度 `[10,10,10]`，姿态 `[100,100,50]`，
  角速度 `[10,10,10]`；输入 `1e-3`；slack `1e4`。
- 求解器：OSQP 0.6.3，`eps_abs=eps_rel=1e-4`，`max_iter=2000`。

### 4.5 求解器封装

- `mpc.cpp` 每次 solve 组装稀疏 `P/A`（Eigen triplets → CSC）。
- CSC 数组由 `CscOwner` 管理，`osqp_cleanup` 后释放。
- 输出第一拍接触力，供 `ForceMapper` 映射。
- 失败（非 `OSQP_SOLVED`）时返回 `success=false`，调用方应降级到 scripted 模式。
- P0 每次 solve 重新 setup；实测 mean ~3.2–3.6 ms、p99 ~4.0 ms
  （归档运行见 `reports/cpp_stack/mpc_bench.json`）。
  P1 可改为复用 workspace + `osqp_update_*` 以降低分配。

## 5. 接触力映射

MPC 输出世界系接触力。对每个轮：

```text
f_base = R_base^T f_world
tau_wheel = +r_w * f_base.x
tau_wheel = clip(tau_wheel, ±15 Nm)
action6[i] = tau_wheel / 15
```

- 符号由独立探针确认：四轮 +1 Nm → base `vx > 0`；-1 Nm → `vx < 0`。
- `LEGACY_6D` 只使用纵向力；侧向/垂直力不进入现有 6D 接口，腿姿态由
  现有 PD 保持。
- P1 的 16 维力矩模式将加入 `J_leg^T f + PD` 前馈。

## 6. 脚本教师与预减速

`PathTracker` 逐元素复现 `MultiSegmentLowLevelController`：

```text
center_y(seg, x) = amp * sin(2π (x - x0) / (x1 - x0))
xt = min(x + lookahead, x1)
bias = clip(-k_ang * wrap(atan2(center_y(xt) - y, xt - x) - yaw)
            + k_dev * (y - center_y(x)), ±bias_clip)
action = [fwd - bias, fwd + bias, fwd - bias, fwd + bias, 0, 0]
fwd = 0.12 * speed_scale
```

- `lookahead=0.7`，`k_ang=2.5`，`k_dev=0.5`，`bias_clip=1.0`。
- 子目标容差内：速度低于阈值 → 零力矩；否则 `-0.02|v|` 制动。
- `pre_brake=True` 时，在 `dist<0.8` 预减速：
  `dist<0.3 → speed_scale=0.4`；否则 `0.3 + (dist-0.3)/0.5*0.7`。
- 该行为来自归档多段脚本验证；默认关闭以保持历史协议。

## 7. 安全层

| 状态 | 触发 | 行为 |
| --- | --- | --- |
| INIT | 启动 | 零输出，等待 reset |
| STAND | 姿态稳定 | 零输出 |
| TRACK | 正常 | 正常动作 |
| DEGRADED | QP 连续失败 >0.1 s / 命令超时 | 动作 ×0.5 |
| SAFE_STOP | roll/pitch > 0.5 rad | 零输出 |
| ESTOP | base z < 0.15 m / NaN | 零输出，需 reset |

P0 的 `LEGACY_6D` 在 `TRACK` 下不修改动作；`DEGRADED` 缩放 0.5。
安全层测试覆盖低高度、QP 连续失败和动作过滤。

## 8. 已知简化与边界

- P0 的 MPC 未闭环接入 `Go2wEnv`；它是独立验证的控制模块。
- 完整 WBC / RNEA / 浮基动力学约束未实现。
- 侧向/垂直接触力未通过现有 6D 接口执行。
- EKF 未实现；P0 直接使用 MuJoCo 状态。
- 无视觉/触觉/语言；无实机/sim-to-real。
- P1 必须补齐 16 维力矩接口、EKF、ROS2 节点和闭环 MPC 评估。
