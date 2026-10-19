#!/usr/bin/env python3
"""Generate canonical Go2W constants for the C++ core and Python adapter.

The MuJoCo model orders joints/bodies/actuators differently from the canonical
order used by the C++ controller (FR, FL, RR, RL). This script resolves every
entity by name and emits:

  cpp/include/go2w/generated/model_constants.hpp
  rl/go2w_constants.py

Run from the repository root:

  .venv/bin/python scripts/gen_model_constants.py
"""

from __future__ import annotations

import hashlib
import pathlib
import sys

import mujoco
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "go2w" / "go2w.xml"
CPP_HEADER = PROJECT_ROOT / "cpp" / "include" / "go2w" / "generated" / "model_constants.hpp"
PY_MODULE = PROJECT_ROOT / "rl" / "go2w_constants.py"

LEGS = ("FR", "FL", "RR", "RL")
JOINT_KINDS = (
    ("hip", "hip_joint", 0),
    ("thigh", "thigh_joint", 1),
    ("calf", "calf_joint", 2),
    ("wheel", "wheel_joint", 3),
)


def _name2id(model: mujoco.MjModel, obj: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, obj, name)
    if idx < 0:
        raise ValueError(f"model entity not found: {name}")
    return int(idx)


def _quat_wxyz_to_xyzw(q: np.ndarray) -> list[float]:
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def _fmt(value: float) -> str:
    if np.isinf(value):
        return "1.0e30" if value > 0 else "-1.0e30"
    return f"{value:.17g}"


def _fmt_vec(values) -> str:
    return "{" + ", ".join(_fmt(float(v)) for v in values) + "}"


def _fmt_quat(values) -> str:
    return "{" + ", ".join(_fmt(float(v)) for v in values) + "}"


def build() -> dict:
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    raw = MODEL_PATH.read_bytes()
    model_hash = hashlib.sha256(raw).hexdigest()

    base_bid = _name2id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    # The free joint is unnamed in the MJCF; resolve it by scanning joints
    # attached to base_link.
    free_jid = next(
        j for j in range(model.njnt) if model.jnt_bodyid[j] == base_bid
    )

    bodies: list[dict] = []

    def add_body(
        *,
        name: str,
        parent: int,
        pos,
        axis,
        leg: int,
        joint_kind: int,
        bid: int,
    ) -> int:
        bodies.append(
            {
                "name": name,
                "parent": parent,
                "pos": [float(v) for v in pos],
                "axis": [float(v) for v in axis],
                "leg": leg,
                "joint_kind": joint_kind,
                "mass": float(model.body_mass[bid]),
                "ipos": [float(v) for v in model.body_ipos[bid]],
                "iquat": _quat_wxyz_to_xyzw(model.body_iquat[bid]),
                "diag_inertia": [float(v) for v in model.body_inertia[bid]],
            }
        )
        return len(bodies) - 1

    add_body(
        name="base_link",
        parent=-1,
        pos=(0.0, 0.0, 0.0),
        axis=(0.0, 0.0, 0.0),
        leg=-1,
        joint_kind=-1,
        bid=base_bid,
    )

    legs: list[dict] = []
    standard = {"hip": 0, "thigh": 1, "calf": 2, "wheel": 3}
    for leg_idx, leg in enumerate(LEGS):
        leg_info: dict = {
            "name": leg,
            "bodies": [0, 0, 0, 0],
            "qpos": [0, 0, 0, 0],
            "dof": [0, 0, 0, 0],
            "act": [0, 0, 0, 0],
            "joints": [0, 0, 0, 0],
            "ranges": [],
            "limited": [],
            "tau_max": [],
            "axes": [],
            "body_pos": [],
        }
        parent_body_index = 0  # base
        for kind_name, joint_suffix, kind_idx in JOINT_KINDS:
            body_name = f"{leg}_{kind_name}" if kind_name != "wheel" else f"{leg}_wheel_link"
            joint_name = f"{leg}_{joint_suffix}"
            act_name = f"{leg}_{kind_name}" if kind_name != "wheel" else f"{leg}_wheel"
            bid = _name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            jid = _name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            aid = _name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_name)
            limited = bool(model.jnt_limited[jid])
            rng = model.jnt_range[jid]
            if not limited:
                lo, hi = -1.0e30, 1.0e30
            else:
                lo, hi = float(rng[0]), float(rng[1])
            axis = [float(v) for v in model.jnt_axis[jid]]
            body_index = add_body(
                name=body_name,
                parent=parent_body_index,
                pos=model.body_pos[bid],
                axis=axis,
                leg=leg_idx,
                joint_kind=kind_idx,
                bid=bid,
            )
            leg_info["bodies"][kind_idx] = body_index
            leg_info["qpos"][kind_idx] = int(model.jnt_qposadr[jid])
            leg_info["dof"][kind_idx] = int(model.jnt_dofadr[jid])
            leg_info["act"][kind_idx] = aid
            leg_info["joints"][kind_idx] = jid
            leg_info["ranges"].append((lo, hi))
            leg_info["limited"].append(limited)
            leg_info["tau_max"].append(float(model.actuator_ctrlrange[aid][1]))
            leg_info["axes"].append(axis)
            leg_info["body_pos"].append([float(v) for v in model.body_pos[bid]])
            parent_body_index = body_index
        legs.append(leg_info)

    # Wheel radius and stand pose are project constants, not MuJoCo fields.
    wheel_radius = 0.086
    stand_leg = (0.0, 0.67, -1.3)
    stand_base_height = 0.436
    torque_limit_wheel = 15.0

    # Sensor ids for the generated table (used by fixtures / adapters).
    sensor_names = [model.sensor(i).name for i in range(model.nsensor)]
    sensor_ids = {
        name: _name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        for name in sensor_names
    }

    return {
        "model": model,
        "model_hash": model_hash,
        "base_body_id": base_bid,
        "free_joint_id": free_jid,
        "free_qpos": int(model.jnt_qposadr[free_jid]),
        "free_dof": int(model.jnt_dofadr[free_jid]),
        "nq": int(model.nq),
        "nv": int(model.nv),
        "nu": int(model.nu),
        "nbody": int(model.nbody),
        "bodies": bodies,
        "legs": legs,
        "wheel_radius": wheel_radius,
        "stand_leg": stand_leg,
        "stand_base_height": stand_base_height,
        "torque_limit_wheel": torque_limit_wheel,
        "sensor_ids": sensor_ids,
        "physics_dt": float(model.opt.timestep),
        "gravity": [0.0, 0.0, -9.81],
    }


def emit_cpp(data: dict) -> str:
    lines: list[str] = []
    lines.append("// Generated by scripts/gen_model_constants.py. Do not edit.")
    lines.append(f"// Model: models/go2w/go2w.xml")
    lines.append(f"// SHA-256: {data['model_hash']}")
    lines.append("#pragma once")
    lines.append("")
    lines.append("#include <array>")
    lines.append("")
    lines.append("namespace go2w {")
    lines.append("namespace constants {")
    lines.append("")
    lines.append(f"constexpr int kNumLegs = 4;")
    lines.append(f"constexpr int kNumLegJoints = 12;")
    lines.append(f"constexpr int kNumWheels = 4;")
    lines.append(f"constexpr int kNumActuators = {data['nu']};")
    lines.append(f"constexpr int kNumQpos = {data['nq']};")
    lines.append(f"constexpr int kNumDof = {data['nv']};")
    lines.append(f"constexpr int kNumBodies = {len(data['bodies'])};")
    lines.append(f"constexpr int kBaseBodyId = {data['base_body_id']};")
    lines.append(f"constexpr int kFreeJointId = {data['free_joint_id']};")
    lines.append(f"constexpr int kFreeQpos = {data['free_qpos']};")
    lines.append(f"constexpr int kFreeDof = {data['free_dof']};")
    lines.append(f"constexpr double kPhysicsDt = {_fmt(data['physics_dt'])};")
    lines.append(f"constexpr double kGravity = 9.81;")
    lines.append(f"constexpr double kWheelRadius = {_fmt(data['wheel_radius'])};")
    lines.append(f"constexpr double kTorqueLimitWheel = {_fmt(data['torque_limit_wheel'])};")
    lines.append(f"constexpr std::array<double, 3> kStandLeg = {_fmt_vec(data['stand_leg'])};")
    lines.append(f"constexpr double kStandBaseHeight = {_fmt(data['stand_base_height'])};")
    lines.append("")
    lines.append("struct BodyDef {")
    lines.append("  const char* name;")
    lines.append("  int parent;          // -1 for the floating base")
    lines.append("  int leg;             // -1 for the base, otherwise 0..3")
    lines.append("  int joint_kind;      // 0=hip, 1=thigh, 2=calf, 3=wheel")
    lines.append("  std::array<double, 3> pos;        // child frame origin in parent")
    lines.append("  std::array<double, 3> axis;       // hinge axis in parent frame")
    lines.append("  std::array<double, 3> ipos;       // inertial origin in body frame")
    lines.append("  std::array<double, 4> iquat;      // inertial orientation (xyzw)")
    lines.append("  std::array<double, 3> diag_inertia;")
    lines.append("  double mass;")
    lines.append("};")
    lines.append("")
    lines.append(f"inline constexpr std::array<BodyDef, {len(data['bodies'])}> kBodies = {{{{")
    for b in data["bodies"]:
        lines.append(
            '    {"%s", %d, %d, %d, %s, %s, %s, %s, %s, %s},'
            % (
                b["name"],
                b["parent"],
                b["leg"],
                b["joint_kind"],
                _fmt_vec(b["pos"]),
                _fmt_vec(b["axis"]),
                _fmt_vec(b["ipos"]),
                _fmt_quat(b["iquat"]),
                _fmt_vec(b["diag_inertia"]),
                _fmt(b["mass"]),
            )
        )
    lines.append("}};")
    lines.append("")
    lines.append("struct LegDef {")
    lines.append("  const char* name;")
    lines.append("  std::array<int, 4> body;")
    lines.append("  std::array<int, 4> qpos;")
    lines.append("  std::array<int, 4> dof;")
    lines.append("  std::array<int, 4> act;")
    lines.append("  std::array<double, 4> tau_max;")
    lines.append("  std::array<std::array<double, 2>, 4> range;")
    lines.append("  std::array<std::array<double, 3>, 4> axis;")
    lines.append("};")
    lines.append("")
    lines.append("inline constexpr std::array<LegDef, 4> kLegs = {{")
    for leg in data["legs"]:
        ranges = ", ".join(
            "{%s, %s}" % (_fmt(lo), _fmt(hi)) for lo, hi in leg["ranges"]
        )
        axes = ", ".join(_fmt_vec(a) for a in leg["axes"])
        lines.append(
            '    {"%s", %s, %s, %s, %s, %s, {{%s}}, {{%s}}},'
            % (
                leg["name"],
                _fmt_vec(leg["bodies"]),
                _fmt_vec(leg["qpos"]),
                _fmt_vec(leg["dof"]),
                _fmt_vec(leg["act"]),
                _fmt_vec(leg["tau_max"]),
                ranges,
                axes,
            )
        )
    lines.append("}};")
    lines.append("")
    lines.append("}  // namespace constants")
    lines.append("}  // namespace go2w")
    lines.append("")
    return "\n".join(lines)


def emit_python(data: dict) -> str:
    lines: list[str] = []
    lines.append('"""Generated by scripts/gen_model_constants.py. Do not edit."""')
    lines.append("")
    lines.append(f"MODEL_SHA256 = {data['model_hash']!r}")
    lines.append(f"NUM_LEGS = 4")
    lines.append(f"NUM_QPOS = {data['nq']}")
    lines.append(f"NUM_DOF = {data['nv']}")
    lines.append(f"NUM_ACTUATORS = {data['nu']}")
    lines.append(f"FREE_QPOS = {data['free_qpos']}")
    lines.append(f"FREE_DOF = {data['free_dof']}")
    lines.append(f"WHEEL_RADIUS = {data['wheel_radius']!r}")
    lines.append(f"TORQUE_LIMIT_WHEEL = {data['torque_limit_wheel']!r}")
    lines.append(f"STAND_LEG = {tuple(data['stand_leg'])!r}")
    lines.append(f"STAND_BASE_HEIGHT = {data['stand_base_height']!r}")
    lines.append("")
    lines.append("LEG_ORDER = ('FR', 'FL', 'RR', 'RL')")
    lines.append("")
    lines.append("LEG_QPOS = (")
    for leg in data["legs"]:
        lines.append(f"    {tuple(leg['qpos'])!r},")
    lines.append(")")
    lines.append("")
    lines.append("LEG_DOF = (")
    for leg in data["legs"]:
        lines.append(f"    {tuple(leg['dof'])!r},")
    lines.append(")")
    lines.append("")
    lines.append("LEG_ACT = (")
    for leg in data["legs"]:
        lines.append(f"    {tuple(leg['act'])!r},")
    lines.append(")")
    lines.append("")
    lines.append(f"WHEEL_QPOS = {tuple(leg['qpos'][3] for leg in data['legs'])!r}")
    lines.append(f"WHEEL_DOF = {tuple(leg['dof'][3] for leg in data['legs'])!r}")
    lines.append(f"WHEEL_ACT = {tuple(leg['act'][3] for leg in data['legs'])!r}")
    lines.append("")
    lines.append("LEG_RANGES = (")
    for leg in data["legs"]:
        lines.append(f"    {tuple(tuple(r) for r in leg['ranges'])!r},")
    lines.append(")")
    lines.append("")
    lines.append("LEG_TAU_MAX = (")
    for leg in data["legs"]:
        lines.append(f"    {tuple(leg['tau_max'])!r},")
    lines.append(")")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    data = build()
    cpp_text = emit_cpp(data)
    py_text = emit_python(data)
    CPP_HEADER.parent.mkdir(parents=True, exist_ok=True)
    CPP_HEADER.write_text(cpp_text)
    PY_MODULE.write_text(py_text)
    print(f"wrote {CPP_HEADER.relative_to(PROJECT_ROOT)}")
    print(f"wrote {PY_MODULE.relative_to(PROJECT_ROOT)}")
    print(f"model sha256: {data['model_hash']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
