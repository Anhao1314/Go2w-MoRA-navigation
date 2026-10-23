"""Command-sequence MDP terms: stage state, observations, reward, termination.

L3 difficulty (docs/RESEARCH.md ladder): four stages executed in order,

1. walk forward 2 m along the initial heading to P1
2. turn in place by +90 degrees
3. walk 2 m more (along the turned heading) to P2
4. stop and hold 1 s

Design decisions carried over from the L1 iteration (each was a diagnosed
failure fix, see the journal):

- targets are anchored on the *initial heading*, not world axes: the policy
  tracks body-frame commands, and reset randomises yaw
- the stage command is *closed loop*: proportional heading correction toward
  the current target, speed reduction while turning.  A constant command let
  a 15.8 deg heading offset accumulate past the position tolerance and the
  stage never fired
- "stopped" is judged on positional drift (<= STOP_DRIFT_TOLERANCE_M from the
  stop anchor), not instantaneous speed: a settled quadruped still shows speed
  oscillation at standstill

State lives on the env instance (``_seq_state``) so the MDP terms remain
stateless functions of the environment.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:  # pragma: no cover
    from isaaclab.envs import ManagerBasedRLEnv

FORWARD_DISTANCE_M = 2.0
TURN_RADIANS = 1.5707963267948966  # +90 degrees
POSITION_TOLERANCE_M = 0.30
HEADING_TOLERANCE_RAD = 0.175  # 10 degrees
STOP_SPEED_TOLERANCE_MPS = 0.10
# 0.15 m covers the standing weight-shift of the quadruped: measured settled
# drift is 0.042 m (single layer) and 0.110 m (layered), both stable rather
# than accumulating.  The same criterion is applied to both groups.
STOP_DRIFT_TOLERANCE_M = 0.15
STOP_HOLD_STEPS = 50  # 1 s at 50 Hz
NUM_STAGES = 4


class _SeqState:
    """Per-env sequence state, created lazily on first use."""

    stage: torch.Tensor
    start: torch.Tensor  # (num_envs, 3): x, y, heading at reset
    stop_steps: torch.Tensor
    completed: torch.Tensor
    stop_anchor: torch.Tensor  # (num_envs, 2) position when the stop stage began


def _state(env: ManagerBasedRLEnv) -> _SeqState:
    state = getattr(env, "_seq_state", None)
    if state is None:
        state = _SeqState()
        device = env.device
        state.stage = torch.zeros(env.num_envs, dtype=torch.long, device=device)
        state.start = torch.zeros(env.num_envs, 3, device=device)
        state.stop_steps = torch.zeros(env.num_envs, device=device)
        state.completed = torch.zeros(env.num_envs, dtype=torch.bool, device=device)
        state.stop_anchor = torch.zeros(env.num_envs, 2, device=device)
        env._seq_state = state  # type: ignore[attr-defined]
    return state


def _robot(env: ManagerBasedRLEnv):
    return env.scene["robot"]


def _wrap(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _targets(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """(P1, P2) in world coordinates, anchored on the initial heading."""
    state = _state(env)
    heading0 = state.start[:, 2]
    heading1 = heading0 + TURN_RADIANS
    forward1 = torch.stack((torch.cos(heading0), torch.sin(heading0)), dim=1)
    forward2 = torch.stack((torch.cos(heading1), torch.sin(heading1)), dim=1)
    p1 = state.start[:, :2] + FORWARD_DISTANCE_M * forward1
    p2 = p1 + FORWARD_DISTANCE_M * forward2
    return p1, p2


def _target_heading(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _state(env).start[:, 2] + TURN_RADIANS


def _heading_error(env: ManagerBasedRLEnv) -> torch.Tensor:
    return _wrap(_target_heading(env) - _robot(env).data.heading_w)


def _remaining(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Remaining distance (stages 0/2), heading error (stage 1), speed (stage 3)."""
    state = _state(env)
    robot = _robot(env)
    position = robot.data.root_pos_w[:, :2]
    p1, p2 = _targets(env)
    distance_p1 = torch.norm(p1 - position, dim=1)
    distance_p2 = torch.norm(p2 - position, dim=1)
    heading_error = torch.abs(_heading_error(env))
    speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)

    remaining = torch.where(state.stage == 0, distance_p1, distance_p2)
    remaining = torch.where(state.stage == 1, heading_error, remaining)
    remaining = torch.where(state.stage == 3, speed, remaining)
    return remaining


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
    state.stop_anchor[ids] = position[:, :2]
    state.start[ids, 0] = position[:, 0]
    state.start[ids, 1] = position[:, 1]
    state.start[ids, 2] = robot.data.heading_w[ids]


def _advance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Advance stages whose success condition holds; return which envs advanced."""
    state = _state(env)
    robot = _robot(env)
    position = robot.data.root_pos_w[:, :2]
    p1, p2 = _targets(env)
    heading_error = torch.abs(_heading_error(env))
    speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)

    at_p1 = torch.norm(p1 - position, dim=1) <= POSITION_TOLERANCE_M
    at_p2 = torch.norm(p2 - position, dim=1) <= POSITION_TOLERANCE_M
    turned = heading_error <= HEADING_TOLERANCE_RAD

    # hold timer only accumulates during the stop stage; judged on drift
    drift = torch.norm(position - state.stop_anchor, dim=1)
    holding = (state.stage == 3) & (drift <= STOP_DRIFT_TOLERANCE_M)
    state.stop_steps = torch.where(holding, state.stop_steps + 1.0, torch.zeros_like(state.stop_steps))
    held = state.stop_steps >= STOP_HOLD_STEPS

    advanced = (
        ((state.stage == 0) & at_p1)
        | ((state.stage == 1) & turned)
        | ((state.stage == 2) & at_p2)
        | ((state.stage == 3) & held)
    )
    state.stage = torch.where(advanced, state.stage + 1, state.stage)
    state.completed = state.stage >= NUM_STAGES
    # anchor the stop stage at the position where it began
    just_entered_stop = advanced & (state.stage == 3)
    state.stop_anchor = torch.where(
        just_entered_stop.unsqueeze(1), position, state.stop_anchor
    )
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
    """Observation: the closed-loop [vx, wz] command the current stage asks for.

    Stages 0/2 aim at their target point; stage 1 turns in place toward the
    target heading; stage 3 commands a stop.  This instruction is the
    rule-level "high-level command" that separates the groups: group B's S1
    only has to learn when to (re)issue it in 2-dim space, while group A must
    realise the same behaviour in 12-dim joint space.
    """
    state = _state(env)
    robot = _robot(env)
    position = robot.data.root_pos_w[:, :2]

    # stages 0 and 2 share the walk-to-point controller with different targets
    p1, p2 = _targets(env)
    target = torch.where((state.stage == 2).unsqueeze(1), p2, p1)
    delta = target - position
    desired_heading = torch.atan2(delta[:, 1], delta[:, 0])
    walk_error = _wrap(desired_heading - robot.data.heading_w)
    walk_speed = 0.6 * torch.clamp(1.0 - 0.8 * walk_error.abs(), 0.15, 1.0)
    walk_turn = torch.clamp(2.0 * walk_error, -1.0, 1.0)

    # stage 1 turns in place toward the +90 deg heading
    turn_error = _heading_error(env)
    turn_speed = torch.zeros_like(walk_speed)
    turn_turn = torch.clamp(2.0 * turn_error, -1.0, 1.0)

    vx = torch.where(state.stage == 1, turn_speed, walk_speed)
    wz = torch.where(state.stage == 1, turn_turn, walk_turn)

    stopping = state.stage >= 3
    vx = torch.where(stopping, torch.zeros_like(vx), vx)
    wz = torch.where(stopping, torch.zeros_like(wz), wz)
    return torch.stack((vx, wz), dim=1)


def stage_onehot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation: one-hot encoding of the current stage."""
    state = _state(env)
    index = torch.clamp(state.stage, 0, NUM_STAGES - 1)
    return torch.nn.functional.one_hot(index, num_classes=NUM_STAGES).float()


def target_delta(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation: remaining amount for the current stage and target-heading error."""
    return torch.stack((_remaining(env), _heading_error(env)), dim=1)


def stage_velocity_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Auxiliary reward: track the current stage's closed-loop command.

    Without this term the policy converges to standing (progress-only reward
    gives no exploration gradient toward motion); two PPO runs failed that way
    before it was added.
    """
    robot = _robot(env)
    target = stage_command(env)
    lin_error = (robot.data.root_lin_vel_b[:, 0] - target[:, 0]).abs()
    yaw_error = (robot.data.root_ang_vel_b[:, 2] - target[:, 1]).abs()
    return torch.exp(-4.0 * (lin_error + yaw_error))


def stage_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Shaped progress reward: pay per unit of remaining error removed.

    r_t = 10 * clip(remaining_{t-1} - remaining_t, -0.05, 0.05) + 10 * advanced

    Stage transitions reset the baseline so the metric change between stages
    (distance -> heading error -> speed) is not paid as progress.
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
