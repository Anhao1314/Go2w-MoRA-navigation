"""S2/S2b: disturbance robustness and light terrain (M2).

Scenario S2 (``docs/EXPERIMENT_DESIGN.md`` §5): 100 N / 0.5 s lateral push,
+/-3 kg payload, friction randomization on top of S1.  Scenario S2b adds
low-amplitude rough ground (noise 2-5 cm) and slopes <= 15 deg while staying
blind (no height scanner).

Skeleton status: the frozen disturbance constants are pinned here and wired
onto the official event terms.  Exact push scheduling per evaluation seed and
the S2b terrain generator parameters are M2 wiring work; both are marked with
TODO(M2) and must be settled before the first M2 training run, not after.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)

from locomotion.isaac.velocity_env_cfg import NUM_TRAIN_ENVS, Go2BlindVelocityEnvCfg

PUSH_FORCE_N = 100.0
"""Frozen S2 lateral push force."""

PUSH_DURATION_S = 0.5
"""Frozen S2 push duration."""

PUSH_IMPULSE_N_S = PUSH_FORCE_N * PUSH_DURATION_S
"""Impulse of the frozen push (50 N s)."""

GO2_BASE_MASS_KG = 15.0
"""Nominal Go2 base mass used to express the frozen impulse as a velocity step.

TODO(M2): read the actual mass from the official asset articulation at wiring
time instead of relying on this nominal constant.
"""

PUSH_VELOCITY_RANGE_MPS = PUSH_IMPULSE_N_S / GO2_BASE_MASS_KG
"""Velocity-step equivalent of the frozen push (~3.33 m/s on 15 kg)."""

PAYLOAD_MASS_RANGE_KG = (-3.0, 3.0)
"""Frozen S2 payload range."""

FRICTION_STEP_RANGE = (0.4, 1.25)
"""Startup static-friction randomization standing in for the frozen "friction step".

TODO(M2): decide whether the protocol's friction step is this startup range
or an in-episode step change; record the decision in the protocol changelog
if it deviates from "randomize at startup".
"""


@configclass
class Go2S2DisturbanceEnvCfg(Go2BlindVelocityEnvCfg):
    """S2: S1 + frozen disturbance set (push, payload, friction)."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # frozen push: 100 N for 0.5 s lateral, expressed as a velocity step
        # TODO(M2): pin the per-seed push schedule for evaluation episodes
        self.events.push_robot.interval_range_s = (10.0, 15.0)
        self.events.push_robot.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (-PUSH_VELOCITY_RANGE_MPS, PUSH_VELOCITY_RANGE_MPS),
        }

        # frozen payload: +/-3 kg on the base
        self.events.add_base_mass.params["mass_distribution_params"] = PAYLOAD_MASS_RANGE_KG

        # frozen friction step (see FRICTION_STEP_RANGE note)
        self.events.physics_material.params["static_friction_range"] = FRICTION_STEP_RANGE


@configclass
class Go2S2bLightTerrainEnvCfg(UnitreeGo2RoughEnvCfg):
    """S2b: light rough terrain, still blind (no height scanner, 87-dim obs)."""

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # stay blind: drop the height scanner and perceptive observations
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum in the frozen protocol
        self.curriculum.terrain_levels = None

        # protocol: 2048 parallel envs and pinned command ranges
        self.scene.num_envs = NUM_TRAIN_ENVS
        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # TODO(M2): replace the official rough terrain generator with the
        # light-terrain spec (height noise 2-5 cm, slopes <= 15 deg)
