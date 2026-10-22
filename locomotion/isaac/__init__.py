"""Isaac Lab task registrations for the M1/M2 locomotion protocol.

Task IDs are namespaced ``Go2Loco-*`` to avoid colliding with the official
``Isaac-Velocity-*`` registrations.

**Import order contract:** this module must be importable *before* Kit starts
(so a trainer can register tasks early), therefore it imports nothing from
``isaaclab`` / ``isaaclab_tasks`` at module level -- registration only stores
string entry points, which Gymnasium resolves lazily on ``gym.make``.  The
``pxr`` (USD) module that the env configs need is provided by the Kit runtime
and only exists after ``AppLauncher`` has bootstrapped Isaac Sim.

Launch training with Isaac Lab's RSL-RL launcher; ``scripts/train_locomotion.py``
in this repository wraps it and imports this package first::

    .venv311/Scripts/python.exe scripts/train_locomotion.py \
        --task Go2Loco-S1-Flat-Blind-v0 --headless --num_envs 2048
"""

from __future__ import annotations

import gymnasium as gym

##
# Register Gym environments (S1/S2/S2b of docs/EXPERIMENT_DESIGN.md §5).
##

gym.register(
    id="Go2Loco-S1-Flat-Blind-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "locomotion.isaac.flat_env_cfg:Go2S1FlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            "locomotion.isaac.agents.rsl_rl_ppo_cfg:Go2S1FlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Go2Loco-S1-Flat-Blind-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "locomotion.isaac.flat_env_cfg:Go2S1FlatEnvCfg_PLAY",
        "rsl_rl_cfg_entry_point": (
            "locomotion.isaac.agents.rsl_rl_ppo_cfg:Go2S1FlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Go2Loco-S2-Disturbance-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "locomotion.isaac.disturbance_env_cfg:Go2S2DisturbanceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "locomotion.isaac.agents.rsl_rl_ppo_cfg:Go2S2DisturbancePPORunnerCfg"
        ),
    },
)

gym.register(
    id="Go2Loco-S2b-LightTerrain-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "locomotion.isaac.disturbance_env_cfg:Go2S2bLightTerrainEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "locomotion.isaac.agents.rsl_rl_ppo_cfg:Go2S2bLightTerrainPPORunnerCfg"
        ),
    },
)
