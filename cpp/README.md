# Go2W C++ Control Core (P0)

ROS-agnostic C++17 control core for the Unitree Go2W, with:

- floating-base FK, damped-least-squares IK, per-leg Jacobians;
- centroidal mass/COM/inertia;
- SRBD linear MPC (OSQP, N=20, 50 Hz);
- contact-force → wheel-torque mapping;
- scripted path tracker with optional subgoal pre-braking;
- safety state machine;
- pybind11 bindings for the existing Python `Go2wEnv`.

The core does **not** link ROS2 or the MuJoCo runtime. MuJoCo is used only by
the Python fixture generator and the existing environment.

## Layout

```text
include/go2w/         # public headers
src/                  # implementations
bindings/             # pybind11 module
tests/                # Catch2 tests + MuJoCo CSV fixtures
apps/                 # replay and MPC benchmark
```

## Dependencies

- C++17 compiler (Apple Clang or GCC)
- CMake ≥ 3.20
- Eigen 3.4 (FetchContent)
- OSQP 0.6.3 (FetchContent, Git + qdldl submodule)
- Catch2 3.7.1 (FetchContent, tests only)
- pybind11 (pip, bindings only)

## Build and test (macOS / Linux)

```bash
cmake -S cpp -B build/core -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGO2W_BUILD_TESTS=ON \
  -DGO2W_BUILD_BENCHMARKS=ON
cmake --build build/core
ctest --test-dir build/core --output-on-failure
./build/core/go2w_mpc_bench
```

On CMake 4.x, OSQP 0.6.3 needs `CMAKE_POLICY_VERSION_MINIMUM=3.5`; the
CMakeLists sets it for the OSQP subproject.

## Python bindings

```bash
cmake -S cpp -B build/py -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DGO2W_BUILD_PYTHON=ON \
  -DGO2W_BUILD_TESTS=OFF \
  -DGO2W_BUILD_BENCHMARKS=OFF \
  -Dpybind11_DIR="$(python -m pybind11 --cmakedir)"
cmake --build build/py
```

`rl/cpp_s0_controller.py` imports `go2w_py` from the installed package or falls
back to `build/py`.

## Use from Python

```python
from rl.cpp_s0_controller import CppS0Controller

controller = CppS0Controller(env, pre_brake=True)
wrapper = HighLevelEnvWrapper(env, controller=controller, append_mora=True, ...)
```

Or from the CLI:

```bash
python scripts/evaluate_navigation.py \
  --task multi-segment --method controller --episodes 10 \
  --s0 cpp --pre-brake --output /tmp/cpp_eval
```

## Documentation

- [Architecture](../docs/CPP_ROS2_ARCHITECTURE.md)
- [Control design](../docs/CONTROL_DESIGN.md)
- [Calibration](../docs/CALIBRATION.md)
- [Validation](../docs/VALIDATION.md)
- [P0 protocol](../docs/CPP_STACK_PROTOCOL.md)

## Status and limits

P0 only. No ROS2 nodes, EKF, full WBC/RNEA, vision/tactile, real robot, or
sim-to-real. The MPC module is numerically validated but not yet closed-loop
integrated with `Go2wEnv`; the scripted C++ S0 is the P0 task baseline.
