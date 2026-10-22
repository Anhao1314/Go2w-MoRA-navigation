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

FORWARD_DISTANCE_M = 2.0
TURN_RADIANS = 1.5707963267948966  # +90 degrees
POSITION_TOLERANCE_M = 0.30
HEADING_TOLERANCE_RAD = 0.175  # 10 degrees
STOP_SPEED_TOLERANCE_MPS = 0.10
STOP_HOLD_STEPS = 50  # 1 s at 50 Hz
NUM_STAGES = 4

# command each stage asks the low-level controller for: [vx, wz]
STAGE_COMMANDS = (
    (0.6, 0.0),  # forward
    (0.0, 0.6),  # turn left
    (0.6, 0.0),  # forward
    (0.0, 0.0),  # stop
)


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


def _targets(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (P1, P2) world positions for the current episode."""
    state = _state(env)
    p1 = torch.stack((state.start[:, 0] + FORWARD_DISTANCE_M, state.start[:, 1]), dim=1)
    p2 = torch.stack(
        (state.start[:, 0] + FORWARD_DISTANCE_M, state.start[:, 1] + FORWARD_DISTANCE_M), dim=1
    )
    return p1, p2


def _remaining(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Remaining distance (stage 0/2), heading error (stage 1), or speed (stage 3)."""
    state = _state(env)
    robot = _robot(env)
    position = robot.data.root_pos_w[:, :2]
    p1, p2 = _targets(env)
    distance_p1 = torch.norm(p1 - position, dim=1)
    distance_p2 = torch.norm(p2 - position, dim=1)
    heading_error = torch.abs(
        _wrap(state.start[:, 2] + TURN_RADIANS - robot.data.heading_w)
    )
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
    position = robot.data.root_pos_w[ids]
    state.start[ids, 0] = position[:, 0]
    state.start[ids, 1] = position[:, 1]
    state.start[ids, 2] = robot.data.heading_w[ids]


def _advance(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Advance stages whose success condition holds; return which envs advanced."""
    state = _state(env)
    robot = _robot(env)
    position = robot.data.root_pos_w[:, :2]
    p1, p2 = _targets(env)
    heading_error = torch.abs(
        _wrap(state.start[:, 2] + TURN_RADIANS - robot.data.heading_w)
    )
    speed = torch.norm(robot.data.root_lin_vel_w[:, :2], dim=1)

    at_p1 = torch.norm(p1 - position, dim=1) <= POSITION_TOLERANCE_M
    at_p2 = torch.norm(p2 - position, dim=1) <= POSITION_TOLERANCE_M
    turned = heading_error <= HEADING_TOLERANCE_RAD

    # hold timer only accumulates during stage 3
    holding = (state.stage == 3) & (speed <= STOP_SPEED_TOLERANCE_MPS)
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
    return advanced


def stage_command(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation: the [vx, wz] command requested by the current stage."""
    state = _state(env)
    commands = torch.tensor(STAGE_COMMANDS, device=env.device)
    return commands[torch.clamp(state.stage, 0, NUM_STAGES - 1)]


def stage_onehot(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation: one-hot encoding of the current stage."""
    state = _state(env)
    index = torch.clamp(state.stage, 0, NUM_STAGES - 1)
    return torch.nn.functional.one_hot(index, num_classes=NUM_STAGES).float()


def target_delta(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Observation: remaining distance and heading error to the current target."""
    state = _state(env)
    heading_error = _wrap(state.start[:, 2] + TURN_RADIANS - _robot(env).data.heading_w)
    return torch.stack((_remaining(env), heading_error), dim=1)


def stage_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Progress toward the current stage target, plus a bonus on each advance."""
    advanced = _advance(env)
    remaining = _remaining(env)
    return torch.exp(-2.0 * remaining) + 10.0 * advanced.float()


def sequence_complete(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Termination: all four stages finished."""
    _advance(env)
    return _state(env).completed
