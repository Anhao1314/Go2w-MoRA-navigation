"""S1: flat-ground velocity tracking (M1).

Scenario S1 of ``docs/EXPERIMENT_DESIGN.md`` §5: 2048 envs, blind 48-dim
observation, random velocity commands on flat ground.  The scenario adds no
disturbances on top of the base configuration, so this class exists mainly to
give the registration entry point a stable, protocol-named home.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from locomotion.isaac.velocity_env_cfg import Go2BlindVelocityEnvCfg


@configclass
class Go2S1FlatEnvCfg(Go2BlindVelocityEnvCfg):
    """S1 flat velocity tracking."""


@configclass
class Go2S1FlatEnvCfg_PLAY(Go2S1FlatEnvCfg):
    """Smaller, non-randomized S1 variant for visual inspection."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
