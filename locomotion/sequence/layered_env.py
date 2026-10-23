"""Layered action mapping: S1 emits [vx, wz], a frozen S0 policy emits joints.

This is group B of the comparison in docs/RESEARCH.md.  The single-layer
baseline acts directly in 12-dim joint space; the layered agent acts in 2-dim
command space and delegates execution to the locomotion policy trained in M1.

Observation layout of the sequence task (53 dims), used to rebuild S0's 48-dim
input exactly as it was during M1 training:

    [0:3]   base_lin_vel          -> S0
    [3:6]   base_ang_vel          -> S0
    [6:9]   projected_gravity     -> S0
    [9:11]  stage_command (2)     -> not used by S0 (S1 owns the command now)
    [11:23] joint_pos             -> S0
    [23:35] joint_vel             -> S0
    [35:47] actions (last S0 action, 12) -> S0
    [47:51] stage one-hot         -> S1 only
    [51:53] target delta          -> S1 only

S1 sees all 53 dims, so A and B receive identical task information.
"""

from __future__ import annotations

import pathlib
from typing import Any

import gymnasium as gym
import torch
import torch.nn as nn

NUM_OBS_S0 = 48
NUM_ACTIONS_S0 = 12
NUM_ACTIONS_S1 = 2


class S0Actor(nn.Module):
    """Inference-only MLP matching the RSL-RL actor layout of the M1 policy."""

    def __init__(self, num_obs: int = NUM_OBS_S0, num_actions: int = NUM_ACTIONS_S0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_obs, 512),
            nn.ELU(),
            nn.Linear(512, 256),
            nn.ELU(),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Linear(128, num_actions),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.net(obs)


def load_s0_actor(checkpoint: pathlib.Path, device: str = "cuda:0") -> S0Actor:
    """Load the actor weights from an RSL-RL checkpoint (critic/std discarded)."""
    payload = torch.load(str(checkpoint), map_location=device, weights_only=False)
    state = payload["model_state_dict"]
    # checkpoint keys are "actor.0.weight"; the module nests the MLP under .net
    actor_state = {
        f"net.{key.removeprefix('actor.')}": value
        for key, value in state.items()
        if key.startswith("actor.")
    }
    if not actor_state:
        raise ValueError(f"no actor weights in {checkpoint}")
    model = S0Actor().to(device)
    model.load_state_dict(actor_state)
    model.eval()
    return model


class LayeredSequenceEnv(gym.Wrapper):
    """Wrap the sequence task so S1 controls a 2-dim command instead of joints."""

    def __init__(self, env: gym.Env, s0_actor: S0Actor, device: str = "cuda:0") -> None:
        super().__init__(env)
        self.s0_actor = s0_actor
        self.device = device
        inner = env.unwrapped
        self.num_envs = int(inner.num_envs)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (NUM_ACTIONS_S1,), dtype="float32")
        self._last_policy_obs: torch.Tensor | None = None

    # -- helpers ---------------------------------------------------------
    def _policy_obs(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, dict):
            return obs["policy"]
        return obs

    def _s0_observation(self, obs53: torch.Tensor, command2: torch.Tensor) -> torch.Tensor:
        vx = command2[:, 0]
        wz = command2[:, 1]
        zero = torch.zeros_like(vx)
        command3 = torch.stack((vx, zero, wz), dim=1)
        return torch.cat(
            (
                obs53[:, 0:9],      # base velocities + gravity
                command3,           # the command S1 chose
                obs53[:, 11:35],    # joint positions and velocities
                obs53[:, 35:47],    # last S0 action
            ),
            dim=1,
        )

    # -- gym API ---------------------------------------------------------
    def reset(self, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        self._last_policy_obs = self._policy_obs(obs)
        return obs, info

    def step(self, action: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
        if self._last_policy_obs is None:
            raise RuntimeError("reset() must be called before step()")
        command = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if command.ndim == 1:
            command = command.unsqueeze(0)
        if command.shape[1] != NUM_ACTIONS_S1:
            raise ValueError(
                f"S1 action must have {NUM_ACTIONS_S1} components, got {command.shape}"
            )
        command = command.clamp(-1.0, 1.0)

        s0_obs = self._s0_observation(self._last_policy_obs, command)
        with torch.inference_mode():
            s0_action = self.s0_actor(s0_obs)

        obs, reward, terminated, truncated, info = self.env.step(s0_action)
        self._last_policy_obs = self._policy_obs(obs)
        return obs, reward, terminated, truncated, info
