"""Python adapter that injects the C++ System-0 controller into Go2wEnv.

The adapter implements the duck-typed interface expected by
``HighLevelEnvWrapper``:

    get_state_from_env(env) -> (x, y, yaw)
    compute_action(x, y, yaw, command) -> np.ndarray[6]

The C++ controller reproduces the scripted multi-segment teacher exactly (see
``cpp/tests/test_kinematics.cpp``); future MPC modes will use the same adapter.
"""

from __future__ import annotations

import pathlib
import sys
from typing import Any

import numpy as np

from rl.go2w_constants import LEG_QPOS, WHEEL_QPOS


def _import_go2w_py():
    try:
        import go2w_py  # type: ignore[import-not-found]

        return go2w_py
    except ImportError:
        # Local CMake build fallback (see docs/CPP_ROS2_ARCHITECTURE.md).
        build_dir = pathlib.Path(__file__).resolve().parents[1] / "build" / "py"
        if build_dir.is_dir():
            sys.path.insert(0, str(build_dir))
        import go2w_py  # type: ignore[import-not-found]

        return go2w_py


class CppS0Controller:
    """Adapter around the pybind11 ``go2w_py.S0Controller``."""

    def __init__(self, env: Any | None = None, pre_brake: bool = True) -> None:
        try:
            go2w_py = _import_go2w_py()
        except ImportError as exc:  # pragma: no cover - exercised by users
            raise ImportError(
                "go2w_py is not built. Run: cmake -S cpp -B build/py "
                "-DGO2W_BUILD_PYTHON=ON && cmake --build build/py"
            ) from exc
        self._controller = go2w_py.S0Controller(pre_brake=pre_brake)
        self._env: Any | None = env
        self.pre_brake = bool(pre_brake)

    def reset(self, env: Any | None = None) -> None:
        self._env = env if env is not None else self._env
        self._controller.reset()

    def get_state_from_env(self, env: Any) -> tuple[float, float, float]:
        self._env = env
        x = float(env.data.body("base_link").xpos[0])
        y = float(env.data.body("base_link").xpos[1])
        yaw = float(env._yaw_of())
        return x, y, yaw

    def compute_action(
        self, x: float, y: float, yaw: float, command: dict | None = None
    ) -> np.ndarray:
        if self._env is None:
            raise RuntimeError("get_state_from_env() must be called before compute_action()")
        env = self._env
        cmd = command or {}

        base_p = np.array(
            [x, y, float(env.data.body("base_link").xpos[2])], dtype=np.float64
        )
        # Use the normalized xquat (same source as Go2wEnv._yaw_of), not the
        # raw qpos quaternion, which can drift from unit norm during stepping.
        quat = env.data.xquat[env.base_body_id]  # wxyz
        base_q = np.array([quat[1], quat[2], quat[3], quat[0]], dtype=np.float64)

        q_leg = np.zeros(12, dtype=np.float64)
        for leg in range(4):
            for k in range(3):
                q_leg[3 * leg + k] = float(env.data.qpos[LEG_QPOS[leg][k]])
        q_wheel = np.array(
            [float(env.data.qpos[WHEEL_QPOS[leg]]) for leg in range(4)],
            dtype=np.float64,
        )

        action = self._controller.step_legacy(
            base_p,
            base_q,
            q_leg,
            q_wheel,
            float(cmd.get("speed_scale", 1.0)),
            float(cmd.get("turn_adjust", 0.0)),
            int(cmd.get("segment_idx", 0)),
            float(cmd.get("dist_to_subgoal", 1.0e9)),
            float(cmd.get("speed", 0.0)),
        )
        return np.asarray(action, dtype=np.float32)

    @property
    def total_mass(self) -> float:
        return float(self._controller.total_mass())

    @property
    def last_safety(self) -> int:
        return int(self._controller.last_safety())

    @property
    def last_valid(self) -> bool:
        return bool(self._controller.last_valid())
