"""Blind Go2 velocity-tracking base configuration (M1/M2 protocol base).

Derives from the official Isaac Lab ``UnitreeGo2FlatEnvCfg`` and pins the
frozen protocol values (``docs/EXPERIMENT_DESIGN.md`` §2):

- blind 48-dim observation (official ``Isaac-Velocity-Flat-UnitreeGo2`` spec,
  no height scan, no perceptive inputs: base_lin_vel 3 + base_ang_vel 3 +
  projected_gravity 3 + velocity_commands 3 + joint_pos 12 + joint_vel 12 +
  actions 12);
- commands: vx ∈ [-1.0, 1.0] m/s, vy = 0 m/s, wz ∈ [-1.0, 1.0] rad/s, direct
  velocity commands (``heading_command`` disabled so the commanded vx/wz pair
  is exactly what the protocol metrics score);
- 2048 parallel training environments, headless.

Only importable inside the ``.venv311`` Isaac Lab environment.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)

NUM_TRAIN_ENVS = 2048
"""Frozen protocol value: 2048 parallel envs, headless."""


@configclass
class Go2BlindVelocityEnvCfg(UnitreeGo2FlatEnvCfg):
    """Base configuration for the S1/S2 blind velocity-tracking ladder."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # protocol: 2048 parallel envs
        self.scene.num_envs = NUM_TRAIN_ENVS

        # protocol: direct velocity commands, vy pinned to zero
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
