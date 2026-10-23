"""Command-sequence MDP terms: stage state, observations, reward, termination.

Four stages, executed in order:

1. walk forward to P1 = (start_x + 2 m, start_y)
2. turn in place by +90 degrees
3. walk to P2 = (start_x + 2 m, start_y + 2 m)
4. stop and hold for 1 s

State lives on the env instance (``_seq_state``) so the MDP terms remain
stateless functions of the environment, which is how Isaac Lab terms are
expected to behave.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedRLEnv

# --- Difficulty L1: forward 1.2 m, then stop and hold 0.5 s ---------------
# L3 (forward 2 m -> turn +90 -> forward 2 m -> stop 1 s) is kept in git
# history (commit ac369fa and before) for when L1 is solved by both
# architectures; the ladder is defined in docs/RESEARCH.md.
FORWARD_DISTANCE_M = 1.2
TURN_RADIANS = 1.5707963267948966  # unused at L1, kept for the L3 variant
POSITION_TOLERANCE_M = 0.30
HEADING_TOLERANCE_RAD = 0.175  # 10 degrees
STOP_SPEED_TOLERANCE_MPS = 0.10
STOP_HOLD_STEPS = 25  # 0.5 s at 50 Hz
NUM_STAGES = 2

# command each stage asks the low-level controller for: [vx, wz]
STAGE_COMMANDS = (
    (0.6, 0.0),  # forward
    (0.0, 0.0),  # stop
)
STAGE_NAMES = ("forward", "stop")


class _SeqState:
    """Per-env sequence state, created lazily on first use."""

    stage: torch.Tensor
    start: torch.Tensor  # (num_envs, 3): x, y, heading
    stop_steps: torch.Tensor
    completed: torch.Tensor


def _state(env: ManagerBasedRLEnv) -> _SeqState:
    state = getattr(env, "_seq_state", None)
    if state is None:
        state = _SeqState()
        device = env.device
        state.stage = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        state.start = torch.zeros(env.num_envs, 3, device=device)
        state.stop_steps = torch.zeros(env.num_envs, device=device)
        state.completed = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        env._seq_state = state  # type: ignore[attr-defined]
    return state


def _robot(env: ManagerBasedRLEnv):
    return env.scene["robot"]


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _targets(env: ManagerBasedRLEnv) -> torch.Tensor:
    """P1 in world coordinates, anchored on the initial heading.

    The first version used world +x, which was a real bug: the policy tracks
    the *body-frame* command, and `reset_base` randomises the initial yaw, so
    walking "forward" in body frame never approaches a world +x target.
    """
    state = _state(env)
    heading0 = state.start[:, 2]
    forward = torch.stack((torch.cos(heading0), torch.sin(heading0)), dim=1)
    return state.start[:, :2] + FORWARD_DISTANCE_M * forward


def _remaining(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Remaining distance for stage 0, speed for stage 1 (stop)."""
    state = _state(env)
    robot = _robot(env)
    position = robot.data.root_pos_w[:, :2]
    distance = torch.norm(_targets(env) - position, dim=1)
    speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)
    return torch.where(state.stage == 0, distance, speed)


def reset_stage(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None) -> None:
    """Event term (mode='reset'): put the sequence back to stage 0."""
    state = _state(env)
    ids = slice(None) if env_ids is None else env_ids
    robot = _robot(env)
    state.stage[ids] = 0
    state.stop_steps[ids] = 0.0
    state.completed[ids] = False
    if hasattr(state, "prev_remaining"):
        delattr(state, "prev_remaining")  # rebuilt on the next reward call
    position = robot.data.root_pos_w[ids]
    state.start[ids, 0] = position[:, 0]
    state.start[ids, 1] = position[:, 1]
    state.start[ids, 2] = robot.data.heading_w[ids]


def _advance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Advance stages whose success condition holds; return which envs advanced."""
    state = _state(env)
    robot = _robot(env)
    position = robot.data.root_pos_w[:, :2]
    distance = torch.norm(_targets(env) - position, dim=1)
    speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)

    at_p1 = distance <= POSITION_TOLERANCE_M

    # hold timer only accumulates during the stop stage
    holding = (state.stage == 1) & (speed <= STOP_SPEED_TOLERANCE_MPS)
    state.stop_steps = torch.where(holding, state.stop_steps + 1.0, torch.zeros_like(state.stop_steps))
    held = state.stop_steps >= STOP_HOLD_STEPS

    advanced = ((state.stage == 0) & at_p1) | ((state.stage == 1) & held)
    state.stage = torch.where(advanced, state.stage + 1, state.stage)
    state.completed = state.stage >= NUM_STAGES
    # The episode resets in the same step it completes, wiping `completed`, so
    # completion is also appended to a log that only the evaluator clears.
    log = getattr(env, "_seq_success_log", None)
    if log is None:
        log = set()
        env._seq_success_log = log  # type: ignore[attr-defined]
    for env_id in (state.completed & advanced).nonzero(as_tuple=True)[0].tolist():
        log.add(int(env_id))
    return advanced


def stage_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation: the [vx, wz] command the current stage asks for.

    Stage 0 is a *closed-loop* command: forward speed plus a proportional
    heading correction toward P1.  The first version emitted a constant
    ``[0.6, 0]``; the policy then walked straight, but any initial heading
    offset from the target direction accumulated sideways error (measured
    15.8 deg -> 0.33 m lateral at 1.2 m) and the stage never fired, so the
    robot kept walking away.  The correction is the rule-level "high-level
    instruction" that the low-level policy executes, which is exactly the
    layering the project studies.
    """
    state = _state(env)
    robot = _robot(env)

    delta = _targets(env) - robot.data.root_pos_w[:, :2]
    desired_heading = torch.atan2(delta[:, 1], delta[:, 0])
    heading_error = _wrap(desired_heading - robot.data.heading_w)

    # slow down while turning; full speed when roughly aligned
    speed = 0.6 * torch.clamp(1.0 - 0.8 * heading_error.abs(), 0.15, 1.0)
    turn = torch.clamp(2.0 * heading_error, -1.0, 1.0)

    stopping = state.stage >= 1
    speed = torch.where(stopping, torch.zeros_like(speed), speed)
    turn = torch.where(stopping, torch.zeros_like(turn), turn)
    return torch.stack((speed, turn), dim=1)


def stage_onehot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation: one-hot encoding of the current stage."""
    state = _state(env)
    index = torch.clamp(state.stage, 0, NUM_STAGES - 1)
    return torch.nn.functional.one_hot(index, num_classes=NUM_STAGES).float()


def target_delta(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation: remaining distance and heading error to the current target."""
    state = _state(env)
    heading_error = torch.abs(_wrap(state.start[:, 2] - _robot(env).data.heading_w))
    return torch.stack((_remaining(env), heading_error), dim=1)


def stage_velocity_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Auxiliary reward: track the current stage's target velocity.

    The progress reward alone gives a standing policy no gradient toward
    motion: standing is safe (no fall) and pays zero progress, while any
    random attempt to walk risks a fall.  Two runs of PPO converged to
    standing under progress-only reward.

    This term pays ``exp(-4 * (|vx - target_vx| + |wz - target_wz|))`` using
    the per-stage command from ``STAGE_COMMANDS``, which supplies the
    exploration gradient.  It does not change the task: success is still
    completing all four stages.
    """
    robot = _robot(env)
    target = stage_command(env)  # same closed-loop instruction the policy sees
    lin_error = (robot.data.root_lin_vel_b[:, 0] - target[:, 0]).abs()
    yaw_error = (robot.data.root_ang_vel_b[:, 2] - target[:, 1]).abs()
    return torch.exp(-4.0 * (lin_error + yaw_error))


def stage_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Shaped progress reward: pay per metre/radian of remaining error removed.

    A plain ``exp(-k * remaining)`` reward was tried first and failed: at a 2 m
    standoff it is ~0.018 per step and its gradient (~0.037 per metre) is
    smaller than the action/posture penalties, so standing still was the local
    optimum and the policy never advanced.  The difference form gives a
    constant signal per step of progress:

        r_t = 10 * clip(remaining_{t-1} - remaining_t, -0.05, 0.05)

    Stage transitions reset the baseline so the metric change between stages
    (distance -> heading error) is not paid as progress; a flat +10 bonus marks
    the transition instead.
    """
    advanced = _advance(env)
    state = _state(env)
    remaining = _remaining(env)

    previous = getattr(state, "prev_remaining", None)
    if previous is None:
        state.prev_remaining = remaining.clone()  # type: ignore[attr-defined]
        return 10.0 * advanced.float()

    delta = previous - remaining
    delta = torch.where(advanced, torch.zeros_like(delta), delta)
    state.prev_remaining = remaining.clone()  # type: ignore[attr-defined]
    return 10.0 * delta.clamp(min=-0.05, max=0.05) + 10.0 * advanced.float()


def sequence_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Termination: all four stages finished."""
    _advance(env)
    return _state(env).completed
