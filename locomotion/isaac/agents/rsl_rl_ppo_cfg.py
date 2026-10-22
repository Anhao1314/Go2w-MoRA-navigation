"""RSL-RL PPO runner configuration for the S1/S2/S2b ladder.

Hyperparameters follow the official ``UnitreeGo2RoughPPORunnerCfg`` defaults.
The frozen protocol pins the stack (PPO via RSL-RL, 2048 envs, headless) but
not these values; tuning them is allowed until the first M1 training run
starts, after which each run's ``run_config.json`` is the record.
"""

from __future__ import annotations

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class Go2VelocityPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 1500
    save_interval = 50
    experiment_name = "go2_locomotion"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class Go2S1FlatPPORunnerCfg(Go2VelocityPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_loco_s1_flat"


@configclass
class Go2S2DisturbancePPORunnerCfg(Go2VelocityPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_loco_s2_disturbance"


@configclass
class Go2S2bLightTerrainPPORunnerCfg(Go2VelocityPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_loco_s2b_light_terrain"
