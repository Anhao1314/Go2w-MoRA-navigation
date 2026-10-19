# Native Windows (MSVC) Validation

This document records the native Windows setup used to validate the C++ core,
the Python environment, and the Python bindings without WSL.

## Environment

| Item | Value |
| --- | --- |
| OS | Windows 11 Pro (25H2) |
| Compiler | Visual Studio 2026 / MSVC 14.51.36231 |
| CMake | 3.30.0-rc4 |
| Ninja | VS-bundled `Common7/IDE/CommonExtensions/Microsoft/CMake/Ninja/ninja.exe` |
| Python | 3.12.10 |
| GPU | NVIDIA GeForce RTX 5060 Ti |
| RAM | 31.8 GB |

Workspace:

```text
D:\work\go2w
  cpp\                 # C++ core
  deps\                # offline Eigen/OSQP/Catch2 sources (not committed)
  build\windows\       # MSVC build
  build\py\            # pybind11 module
  .venv\               # Python 3.12 venv
```

## Why offline dependencies

CMake `FetchContent` for OSQP (`github.com/osqp/osqp`) stalled in
`git index-pack` on this network. The build now uses local source trees:

```text
-D FETCHCONTENT_FULLY_DISCONNECTED=ON
-D FETCHCONTENT_SOURCE_DIR_EIGEN=...\deps\eigen-src
-D FETCHCONTENT_SOURCE_DIR_OSQP=...\deps\osqp-src
-D FETCHCONTENT_SOURCE_DIR_CATCH2=...\deps\catch2-src
```

`D:\work\go2w\build_windows.bat` performs the C++ build; `build_py_windows.bat`
builds the pybind11 module. Both use the VS developer environment and the
VS-bundled Ninja.

## Windows-specific fixes

Two POSIX-only calls in the Python tooling were fixed and covered by the
existing tests:

| File | Problem | Fix |
| --- | --- | --- |
| `rl/monitor.py` | `os.getloadavg()` does not exist on Windows | fall back to `psutil.getloadavg()`, then `(0, 0, 0)` |
| `rl/viewer_launcher.py` | `os.killpg()` does not exist on Windows | use `os.kill(pid, SIGTERM)` and wait for the process to exit |
| `cpp/CMakeLists.txt` | OSQP/qdldl static+shared targets both emit `*.lib` on MSVC | rename static/shared outputs on MSVC |
| `cpp/src/mpc.cpp` | Windows `max`/`min` macros break `std::max` | define `NOMINMAX`, `#undef max/min` |

## Results

| Check | Result |
| --- | --- |
| C++ build | 176 targets, all linked |
| `ctest` | `go2w_tests` passed (13 test cases / 43 assertions) |
| MPC benchmark | 200/200 solved; mean 3.88 ms, p99 4.59 ms, max 5.09 ms; hover vertical-force error 2.15% |
| Python suite | **225 tests OK (4 skipped)** with `MUJOCO_GL=wgl` |
| Python bindings | `go2w_py.cp312-win_amd64.pyd` built; 3 binding/integration tests passed |
| C++/scripted parity | full multi-segment episode identical on Windows |
| 10-episode acceptance | **10/10** success; mean_x 10.6563, mean_time 14.000 s (archived as `reports/cpp_stack/p0_cpp_prebrake_win/`) |

The four skipped Python tests are platform-specific (the same ones skipped on
macOS); they are not failures.

## Commands

```bat
D:\work\go2w\build_windows.bat
D:\work\go2w\build\windows\go2w_mpc_bench.exe
D:\work\go2w\build_py_windows.bat
cd /d D:\work\go2w
.venv\Scripts\python.exe -m unittest discover -s tests -p test_*.py
.venv\Scripts\python.exe -m unittest discover -s tests -p test_cpp_binding.py -v
```

## Limits

- Native Windows, no WSL/ROS2. P1 ROS2 remains on Linux/WSL2/Docker.
- The 4 skipped tests are viewer/display-specific; the real GUI viewer was not
  evaluated interactively.
- The offline `deps\` tree is a local build aid and is ignored by git; it is not
  a substitute for the pinned FetchContent tags.
- No real Go2W hardware was used.
