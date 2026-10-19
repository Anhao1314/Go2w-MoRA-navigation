# C++ Stack P0 评估协议

**协议版本：** `cpp-stack-p0-v1`
**模型 SHA-256：** `c8feaef4afdf360335727c80a826d1611950c562a3daaa5b5bfcf8b57f6859a6`
**MuJoCo：** 3.11.0
**环境：** macOS arm64（Apple Silicon），Release build

## 1. 任务与固定条件

| 项 | 值 |
| --- | --- |
| task | `traverse_curve` + `multi_segment=True` |
| scope | 2（完整 A→B） |
| domain_randomize | `False` |
| 高层命令 | `[speed_scale=1.0, turn_adjust=0.0]` |
| S0 | `scripted` 或 `cpp` |
| pre_brake | `True`（归档多段脚本行为） |
| episodes | 10 |
| 评估种子 | 1000–1009 |
| 输出目录 | `reports/cpp_stack/p0_cpp_prebrake`、`p0_scripted_prebrake`、`p0_scripted_noprebrake` |

## 2. 指标

- **success_rate：** `info["goal"]` 为真的 episode 比例。
- **a_stop_rate：** `info["passed_subgoal"]` 为真的比例。
- **mean_x：** episode 内最大 `progress`（环境指标，不等于物理路程）。
- **mean_ep_len：** 平均策略步数。
- **mean_time：** 平均仿真时间。
- **failed_terminations：** 非目标终止且非截断的次数。

## 3. 结果

| 配置 | success | a_stop | mean_x | mean_ep_len | mean_time | failed term |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scripted，无预减速 | 0/10 | 10/10 | 4.7811 | 630.0 | 6.292 s | 10 |
| scripted，预减速 | 10/10 | 10/10 | 10.6571 | 1399.0 | 13.990 s | 0 |
| C++ S0，预减速 | 10/10 | 10/10 | 10.6571 | 1399.0 | 13.990 s | 0 |

C++ 与 scripted 预减速逐项相同；seed 1000 的完整轨迹也相同。

## 4. 可以主张什么

1. C++ S0 在现有多段任务上逐元素复现了 Python 教师/控制器。
2. 显式 `pre_brake` 开关把现有 `ControllerPolicy` 基线从 0/10 提升到 10/10。
3. FK、Jacobian、质心动力学对 MuJoCo 夹具达到 1e-9 / 1e-6 / 1e-8 级一致。
4. SRBD MPC 在 hover 工况 200/200 求解，p99 4.07 ms，总垂直力误差 2.15%
   （归档运行 `reports/cpp_stack/mpc_bench.json`）。

## 5. 不能主张什么

- 不能主张 MPC 闭环导航优于脚本控制；P0 未做该实验。
- 不能主张 C++ 栈已接入 ROS2、实机或 EKF。
- 不能主张 10/10 是总体成功率；它只是 10 个固定种子/场景的结果。
- 不能把历史报告的成功率追认为新栈结果。
- 不能把 `mean_x` 当作物理路程。
- 不能声称完整 WBC、RNEA 或 sim-to-real。

## 6. 复现

```bash
# C++ core / tests / MPC benchmark
cmake -S cpp -B build/core -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGO2W_BUILD_TESTS=ON -DGO2W_BUILD_BENCHMARKS=ON
cmake --build build/core
ctest --test-dir build/core --output-on-failure
./build/core/go2w_mpc_bench

# Python bindings
cmake -S cpp -B build/py -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGO2W_BUILD_PYTHON=ON -DGO2W_BUILD_TESTS=OFF \
  -DGO2W_BUILD_BENCHMARKS=OFF \
  -Dpybind11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake --build build/py

# Acceptance
.venv/bin/python scripts/evaluate_navigation.py \
  --task multi-segment --method controller --episodes 10 \
  --s0 cpp --pre-brake --output reports/cpp_stack/p0_cpp_prebrake

.venv/bin/python scripts/evaluate_navigation.py \
  --task multi-segment --method controller --episodes 10 \
  --s0 scripted --pre-brake --output reports/cpp_stack/p0_scripted_prebrake
```

## 7. 残留风险

- macOS 大小写不敏感导致 `models/go2w/assets/terrain.STL` 幻影修改；
  不要提交该文件。
- OSQP 0.6.3 在 CMake 4 下需要 `CMAKE_POLICY_VERSION_MINIMUM=3.5`
  和 Git 子模块；CMakeLists 已处理。
- MPC 每次 solve 重新 setup；P1 应复用 workspace 以降低延迟/分配。
