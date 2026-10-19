#!/usr/bin/env python3
"""Generate MuJoCo reference fixtures for the C++ core tests.

Outputs (whitespace-separated numeric CSV, one row per sample):
  cpp/tests/fixtures/fk.csv
  cpp/tests/fixtures/jac.csv
  cpp/tests/fixtures/centroidal.csv
  cpp/tests/fixtures/parity.csv

The C++ tests do not link MuJoCo; these files are the only numerical coupling.

Run from the repository root:

  .venv/bin/python scripts/gen_cpp_fixtures.py
"""

from __future__ import annotations

import pathlib
import sys

import mujoco
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.go2w_constants import (  # noqa: E402
    LEG_QPOS,
    LEG_RANGES,
    NUM_DOF,
    NUM_QPOS,
    STAND_LEG,
)
from rl.low_level_controller import MultiSegmentLowLevelController  # noqa: E402
from rl.go2w_env import MULTI_SEGMENT_DEFAULT  # noqa: E402

MODEL_PATH = PROJECT_ROOT / "models" / "go2w" / "go2w.xml"
FIXTURE_DIR = PROJECT_ROOT / "cpp" / "tests" / "fixtures"
LEGS = ("FR", "FL", "RR", "RL")


def quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.array([q[3], q[0], q[1], q[2]], dtype=np.float64)


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    mat = np.zeros(9, dtype=np.float64)
    mujoco.mju_quat2Mat(mat, q)
    return mat.reshape(3, 3)


def set_state(model: mujoco.MjModel, data: mujoco.MjData,
              base_p: np.ndarray, base_q_xyzw: np.ndarray,
              q_leg: np.ndarray, q_wheel: np.ndarray) -> None:
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.qpos[0:3] = base_p
    data.qpos[3:7] = quat_xyzw_to_wxyz(base_q_xyzw)
    for leg in range(4):
        for k in range(3):
            data.qpos[LEG_QPOS[leg][k]] = q_leg[3 * leg + k]
        data.qpos[LEG_QPOS[leg][3]] = q_wheel[leg]
    mujoco.mj_forward(model, data)


def random_state(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    base_p = np.array([rng.uniform(-0.5, 3.0), rng.uniform(-0.5, 0.5), rng.uniform(0.30, 0.60)])
    roll = rng.uniform(-0.12, 0.12)
    pitch = rng.uniform(-0.12, 0.12)
    yaw = rng.uniform(-np.pi, np.pi)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
    q = np.array([
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
        cr * cp * cy + sr * sp * sy,
    ])
    q_leg = np.zeros(12)
    for leg in range(4):
        for k in range(3):
            lo, hi = LEG_RANGES[leg][k]
            stand = STAND_LEG[k]
            q_leg[3 * leg + k] = rng.uniform(max(lo, stand - 0.35), min(hi, stand + 0.35))
    q_wheel = rng.uniform(-2.0, 2.0, size=4)
    return base_p, q, q_leg, q_wheel


def write_rows(path: pathlib.Path, rows: list[list[float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(" ".join(f"{v:.17g}" for v in row) + "\n")


def main() -> int:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    rng = np.random.default_rng(20261019)

    wheel_bodies = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_wheel_link")
        for leg in LEGS
    ]
    base_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    fk_rows: list[list[float]] = []
    jac_rows: list[list[float]] = []
    centroidal_rows: list[list[float]] = []
    parity_rows: list[list[float]] = []
    controller = MultiSegmentLowLevelController(MULTI_SEGMENT_DEFAULT)

    for sample in range(100):
        base_p, base_q, q_leg, q_wheel = random_state(rng)
        set_state(model, data, base_p, base_q, q_leg, q_wheel)

        expected_centers = []
        expected_jacs = []
        for body_id in wheel_bodies:
            center = data.xpos[body_id].copy()
            expected_centers.extend(center.tolist())
            jacp = np.zeros((3, NUM_DOF))
            jacr = np.zeros((3, NUM_DOF))
            mujoco.mj_jac(model, data, jacp, jacr, center, body_id)
            expected_jacs.extend(jacp.reshape(-1).tolist())

        state_row = list(base_p) + list(base_q) + list(q_leg) + list(q_wheel)
        if sample < 100:
            fk_rows.append(state_row + expected_centers)
            jac_rows.append(state_row + expected_jacs)

        if sample < 50:
            mujoco.mj_comPos(model, data)
            com = data.subtree_com[0].copy()
            inertia = np.zeros((3, 3))
            for body_id in range(1, model.nbody):
                mass = float(model.body_mass[body_id])
                R_body = data.xmat[body_id].reshape(3, 3)
                R_inertial = R_body @ quat_wxyz_to_mat(model.body_iquat[body_id])
                I_world = R_inertial @ np.diag(model.body_inertia[body_id]) @ R_inertial.T
                body_com = data.xpos[body_id] + R_body @ model.body_ipos[body_id]
                r = body_com - com
                inertia += I_world + mass * (r.dot(r) * np.eye(3) - np.outer(r, r))
            mass = float(model.body_mass[1:].sum())
            centroidal_rows.append(
                state_row + [mass] + com.tolist() + inertia.reshape(-1).tolist()
            )

        if sample < 80:
            x = float(base_p[0])
            y = float(base_p[1])
            yaw = float(np.arctan2(
                2.0 * (base_q[3] * base_q[2] + base_q[0] * base_q[1]),
                1.0 - 2.0 * (base_q[1] ** 2 + base_q[2] ** 2),
            ))
            for segment_idx in (0, 1):
                seg = MULTI_SEGMENT_DEFAULT[segment_idx]
                goal_x = float(seg["x1"])
                goal_y = float(
                    seg["amp"]
                    * np.sin(2.0 * np.pi * (goal_x - seg["x0"]) / (seg["x1"] - seg["x0"]))
                )
                dist = float(np.hypot(x - goal_x, y - goal_y))
                for speed_scale in (0.9, 1.0, 1.2, 1.5):
                    for turn_adjust in (-0.3, 0.0, 0.2):
                        speed = float(rng.uniform(0.0, 1.0))
                        action = controller.compute_action(
                            x, y, yaw,
                            {
                                "speed_scale": speed_scale,
                                "turn_adjust": turn_adjust,
                                "segment_idx": segment_idx,
                                "dist_to_subgoal": dist,
                                "speed": speed,
                            },
                        )
                        parity_rows.append(
                            [x, y, yaw, speed_scale, turn_adjust, float(segment_idx),
                             dist, speed] + action.tolist()
                        )

    write_rows(FIXTURE_DIR / "fk.csv", fk_rows)
    write_rows(FIXTURE_DIR / "jac.csv", jac_rows)
    write_rows(FIXTURE_DIR / "centroidal.csv", centroidal_rows)
    write_rows(FIXTURE_DIR / "parity.csv", parity_rows)
    print(f"wrote {len(fk_rows)} FK, {len(jac_rows)} Jacobian, "
          f"{len(centroidal_rows)} centroidal, {len(parity_rows)} parity rows")
    print(f"fixture dir: {FIXTURE_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
