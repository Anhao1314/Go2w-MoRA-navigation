#!/usr/bin/env python3
"""Export the Unitree Go2 rigid-body model used by the C++ guardrail core.

The C++ arbiter must use exactly the parameters the simulator uses, so they
are exported from the official Isaac Lab asset at runtime rather than typed
in by hand.  Output: ``locomotion/guardrail/go2_model.json`` plus a SHA-256
sidecar; a Python test compares the JSON against a fresh instance to catch
drift.

Sources per quantity:
- mass / inertia         : PhysX articulation view (get_masses / get_inertias)
- COM in body frame      : ArticulationData.body_com_pos_b
- joint position limits  : PhysX get_dof_limits
- joint velocity limits  : ArticulationData.joint_velocity_limits
- effort / stiffness     : actuator config on the task
- joint axis / frames    : USD stage (UsdPhysics.RevoluteJoint localPos/localRot)

Run with the repo's ``.venv311`` interpreter:

    PYTHONPATH=. OMNI_KIT_ACCEPT_EULA=YES \
      .venv311/Scripts/python.exe scripts/export_go2_model.py

Exit codes: 0 = exported, 1 = failed.

This script runs only inside the Isaac Lab environment (`.venv311`), which CI
does not install, so unresolved isaaclab/pxr imports are expected there.
"""

# pyright: reportMissingImports=false

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pathlib
import re
import sys
import time
from typing import Any, cast

os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT = PROJECT_ROOT / "locomotion" / "guardrail" / "go2_model.json"

_AXIS = {"X": [1.0, 0.0, 0.0], "Y": [0.0, 1.0, 0.0], "Z": [0.0, 0.0, 1.0]}


def _flatten(value) -> list:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
        if hasattr(value, "tolist"):
            value = value.tolist()
    if isinstance(value, (list, tuple)):
        out: list = []
        for item in value:
            out.extend(_flatten(item))
        return out
    return [value]


def _flat(value) -> list[float]:
    return [float(item) for item in _flatten(value)]


def _axis_from_usd(value) -> list[float] | None:
    if value is None:
        return None
    text = str(value)
    if text in _AXIS:
        return list(_AXIS[text])
    try:
        return [float(v) for v in value]
    except (TypeError, ValueError):
        return None


def _read_usd_joints() -> dict[str, dict]:
    """Read revolute-joint frames from the live USD stage, keyed by joint name."""
    try:
        import omni.usd
        from pxr import UsdPhysics
    except Exception as exc:  # pragma: no cover - Kit only
        print(f"[export] WARNING: cannot import USD APIs: {exc}")
        return {}

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[export] WARNING: no USD stage available")
        return {}

    joints: dict[str, dict] = {}
    for prim in stage.Traverse():
        try:
            if not prim.IsA(UsdPhysics.RevoluteJoint):
                continue
        except Exception:
            continue
        name = prim.GetName()
        record: dict = {"prim": str(prim.GetPath())}
        for attr, key in (
            ("physics:axis", "axis"),
            ("physics:localPos0", "local_pos0"),
            ("physics:localRot0", "local_rot0"),
            ("physics:localPos1", "local_pos1"),
            ("physics:localRot1", "local_rot1"),
        ):
            try:
                value = prim.GetAttribute(attr).Get()
            except Exception:
                value = None
            if value is None:
                continue
            if key == "axis":
                axis = _axis_from_usd(value)
                if axis is not None:
                    record["axis"] = axis
            elif "rot" in key:
                record[key] = [
                    float(value.GetReal()),
                    float(value.GetImaginary()[0]),
                    float(value.GetImaginary()[1]),
                    float(value.GetImaginary()[2]),
                ]
            else:
                record[key] = [float(c) for c in value]
        for rel, key in (("physics:body0", "body0"), ("physics:body1", "body1")):
            try:
                targets = prim.GetRelationship(rel).GetTargets()
            except Exception:
                targets = []
            if targets:
                record[key] = targets[0].name
        if "axis" in record:
            joints[name] = record
    return joints


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="Go2Loco-S1-Flat-Blind-v0")
    parser.add_argument("--output", type=pathlib.Path, default=OUTPUT)
    args = parser.parse_args(argv)

    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher(headless=True)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import locomotion.isaac  # noqa: F401
    from isaaclab_tasks.utils import parse_env_cfg

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1, use_fabric=True)
    env = gym.make(args.task, cfg=env_cfg)
    env.reset()

    # Isaac Lab's manager-based env exposes attributes gym.Env does not type
    inner = cast(Any, env.unwrapped)
    robot = inner.scene["robot"]
    view = robot.root_physx_view

    body_names = list(robot.body_names)
    joint_names = list(robot.joint_names)

    masses = view.get_masses()
    if masses.ndim > 1:
        masses = masses[0]
    inertias = view.get_inertias().reshape(-1, 9)
    com_b = robot.data.body_com_pos_b[0]  # (num_bodies, 3)

    dof_limits = view.get_dof_limits()
    if dof_limits.ndim == 3:
        dof_limits = dof_limits[0]
    vel_limits = robot.data.joint_velocity_limits[0]

    actuator_effort: dict[str, float] = {}
    actuator_stiffness: dict[str, float] = {}
    actuator_damping: dict[str, float] = {}
    for actuator in env_cfg.scene.robot.actuators.values():
        patterns = getattr(actuator, "joint_names_expr", [])
        # actuator classes differ: implicit actuators expose *_sim fields,
        # explicit DC-motor configs (Go2) expose the plain names
        effort = getattr(actuator, "effort_limit_sim", None)
        if effort is None:
            effort = getattr(actuator, "effort_limit", None)
        stiffness = getattr(actuator, "stiffness", None)
        damping = getattr(actuator, "damping", None)
        for joint in joint_names:
            if any(re.fullmatch(str(pattern), joint) for pattern in patterns):
                if effort is not None:
                    actuator_effort[joint] = float(effort)
                if stiffness is not None:
                    actuator_stiffness[joint] = float(stiffness)
                if damping is not None:
                    actuator_damping[joint] = float(damping)

    usd_joints = _read_usd_joints()

    bodies = [
        {
            "name": name,
            "mass": float(masses[index]),
            "com_b": _flat(com_b[index]),
            "inertia": _flat(inertias[index]),
        }
        for index, name in enumerate(body_names)
    ]

    joints = []
    for index, name in enumerate(joint_names):
        usd = usd_joints.get(name, {})
        joints.append(
            {
                "name": name,
                "limit": _flat(dof_limits[index]),
                "velocity_limit": float(vel_limits[index]),
                "effort_limit": actuator_effort.get(name),
                "stiffness": actuator_stiffness.get(name),
                "damping": actuator_damping.get(name),
                "axis": usd.get("axis"),
                "local_pos0": usd.get("local_pos0"),
                "local_rot0": usd.get("local_rot0"),
                "local_pos1": usd.get("local_pos1"),
                "local_rot1": usd.get("local_rot1"),
                "body0": usd.get("body0"),
                "body1": usd.get("body1"),
                "usd_prim": usd.get("prim"),
            }
        )

    payload = {
        "schema": "go2-model-v1",
        "robot": "Unitree Go2",
        "source": {
            "task": args.task,
            "usd": getattr(env_cfg.scene.robot.spawn, "usd_path", None),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "generator": "scripts/export_go2_model.py",
            "usd_joints_found": len(usd_joints),
        },
        "base_body": body_names[0] if body_names else None,
        "bodies": bodies,
        "joints": joints,
        "feet": [name for name in body_names if name.lower().endswith("foot")],
        "default_joint_pos": _flat(robot.data.default_joint_pos[0]),
    }

    env.close()

    # Write before closing Isaac Sim: close() terminates the process from C++,
    # so anything after it is not guaranteed to run.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    args.output.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    args.output.with_suffix(".json.sha256").write_text(
        f"{digest}  {args.output.name}\n", encoding="ascii"
    )

    print(
        f"[export] bodies={len(bodies)} joints={len(joints)} "
        f"usd_joints={len(usd_joints)} feet={payload['feet']}"
    )
    if len(usd_joints) < len(joints):
        missing = [j["name"] for j in joints if j["axis"] is None]
        print(f"[export] WARNING: USD frame missing for {missing}")
    print(f"[export] wrote {args.output}")
    print(f"[export] sha256 {digest}", flush=True)

    simulation_app.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
