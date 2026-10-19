# C++ Stack P0 Evidence

**Protocol:** `cpp-stack-p0-v1` (see [docs/CPP_STACK_PROTOCOL.md](../../docs/CPP_STACK_PROTOCOL.md))
**Model SHA-256:** `c8feaef4afdf360335727c80a826d1611950c562a3daaa5b5bfcf8b57f6859a6`
**MuJoCo:** 3.11.0

## Contents

| Path | Description |
| --- | --- |
| `p0_cpp_prebrake/` | C++ S0, 10 episodes, seeds 1000–1009, pre-brake (macOS) |
| `p0_cpp_prebrake_win/` | C++ S0, same protocol, native Windows/MSVC |
| `p0_scripted_prebrake/` | Scripted S0, same protocol |
| `p0_scripted_noprebrake/` | Scripted S0 without pre-brake (historical baseline) |
| `mpc_bench.json` | 200 MPC solves, hover force/timing |
| `ctest.txt` | C++ test run (13 test cases / 43 assertions) |
| `../../media/cpp_stack_p0_demo.mp4` | C++ S0 seed-1000 demo video |

Each acceptance directory contains `run_config.json`, `episodes.json`,
and `summary.json`, written by `scripts/evaluate_navigation.py`.

## Results

| Configuration | Success | A-stop | mean_x | mean_ep_len | mean_time | failed terminations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| scripted, no pre-brake (historical baseline) | 0/10 | 10/10 | 4.7811 | 630.0 | 6.292 s | 10 |
| scripted, pre-brake | 10/10 | 10/10 | 10.6571 | 1399.0 | 13.990 s | 0 |
| **C++ S0, pre-brake** | **10/10** | 10/10 | **10.6571** | **1399.0** | **13.990 s** | **0** |

C++ and scripted pre-brake match item by item; the seed-1000 trajectory is
identical step by step. The 0/10 no-pre-brake result is the existing
`ControllerPolicy` baseline, not a C++ regression.

MPC benchmark (archived run): 200/200 solved, mean 3.16 ms, p99 4.07 ms,
hover vertical-force error 2.15%.

## Native Windows replication (MSVC 2026)

`p0_cpp_prebrake_win/` is the same protocol on native Windows: **10/10**
success, A-stop 10/10, mean_x 10.6563, mean_ep_len 1400.0, mean_time 14.000 s,
0 failed terminations. The small difference from macOS (10.6571 / 13.990 s) is
platform floating-point/step-count variation, not a controller change; the
Windows C++/scripted binding parity test passes on a full episode.

Windows validation also covered 225 Python tests (4 skipped), the pybind11
module, and a 200/200 MPC benchmark (mean 3.88 ms, p99 4.59 ms). See
[docs/WINDOWS_NATIVE.md](../../docs/WINDOWS_NATIVE.md).

## Reproduction

```bash
cmake -S cpp -B build/core -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGO2W_BUILD_TESTS=ON -DGO2W_BUILD_BENCHMARKS=ON
cmake --build build/core
.venv/bin/ctest --test-dir build/core --output-on-failure
./build/core/go2w_mpc_bench

.venv/bin/python scripts/evaluate_navigation.py \
  --task multi-segment --method controller --episodes 10 \
  --s0 cpp --pre-brake --output reports/cpp_stack/p0_cpp_prebrake
```

## Limits

Simulation only; no ROS2 nodes, EKF, full WBC/RNEA, vision/tactile, real
robot, or sim-to-real. MPC is not yet closed-loop integrated with `Go2wEnv`.
Ten fixed episodes are not a population success rate or a scene-generalization
claim. Historical reports are not retroactively attributed to this stack.
