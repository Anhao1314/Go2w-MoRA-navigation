"""C0 baseline: scripted PD controller (protocol lower bound).

Implements the C0 baseline of ``docs/EXPERIMENT_DESIGN.md`` §6: standing plus
proportional command tracking under the *same task definition* as the RL
policies -- same episodes, same frozen seed set ``E = {0..19}``, same metrics
from :mod:`locomotion.eval.metrics`.  It is the attribution lower bound for
RQ2; its numbers must be reported next to the RL results, sign unconstrained.

Skeleton status: the controller interface and the PD law are fixed here; the
nominal Go2 stance keyframe and the gait-phase schedule are M1 wiring work
(marked TODO(M1)).  Pure Python -- importable without Isaac Sim so the
interface is unit-testable in CI.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

NUM_GO2_JOINTS = 12
"""Go2 has 12 actuated joints (3 per leg: hip, thigh, calf)."""


@dataclass
class ScriptedPDBaseline:
    """Joint-space PD around a nominal stance, scaled by velocity commands.

    The controller is deliberately simple: it holds the nominal stance and
    leans the joint targets proportionally to the commanded forward/yaw
    velocity.  No gait generator -- if standing-plus-lean cannot move the
    robot, that negative result is reportable evidence about how much
    locomotion the RL policy actually contributes.
    """

    nominal_stance_q: np.ndarray = field(
        default_factory=lambda: np.zeros(NUM_GO2_JOINTS)
    )
    """Nominal joint positions [rad].  TODO(M1): fill from the official Go2 keyframe."""

    kp: float = 40.0
    """Proportional gain for the joint-space PD."""

    kd: float = 1.0
    """Derivative gain for the joint-space PD."""

    lean_gain: float = 0.15
    """Joint-target lean per unit commanded forward velocity [rad/(m/s)]."""

    yaw_gain: float = 0.05
    """Hip-target offset per unit commanded yaw rate [rad/(rad/s)]."""

    def compute_joint_targets(
        self,
        commanded_vx: float,
        commanded_wz: float,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(position_targets, feedforward_torques)`` for one control step.

        ``feedforward_torques`` is the pure PD output ``kp*(q*-q) - kd*qdot``;
        the caller decides whether to apply it as torque or track the
        position targets, and must record that choice in the run config.
        """
        joint_pos = np.asarray(joint_pos, dtype=float)
        joint_vel = np.asarray(joint_vel, dtype=float)
        if joint_pos.shape != (NUM_GO2_JOINTS,) or joint_vel.shape != (NUM_GO2_JOINTS,):
            raise ValueError(
                f"expected joint arrays of shape ({NUM_GO2_JOINTS},), got "
                f"{joint_pos.shape} and {joint_vel.shape}"
            )

        targets = self.nominal_stance_q.copy()
        # lean forward/back with commanded vx (all calf/thigh joints)
        targets = targets + self.lean_gain * commanded_vx
        # twist hips with commanded yaw rate (front/rear hip pairs, signs at M1 wiring)
        targets = targets + self.yaw_gain * commanded_wz

        torques = self.kp * (targets - joint_pos) - self.kd * joint_vel
        return targets, torques
