"""Isaac Lab task registrations for the M1/M2 locomotion protocol.

Task IDs are namespaced ``Go2Loco-*`` to avoid colliding with the official
``Isaac-Velocity-*`` registrations.  Importing this package requires the
``.venv311`` Isaac Lab environment; launch training with Isaac Lab's own
launcher, e.g. from the IsaacLab checkout::

    .venv311/Scripts/python.exe D:/work/IsaacLab/scripts/reinforcement_learning/rsl_rl/train.py \
        --task Go2Loco-S1-Flat-Blind-v0 --headless --num_envs 2048

with this repository on ``PYTHONPATH``.
"""

from __future__ import annotations

import gymnasium as gym

from locomotion.isaac.agents import (
    Go2S1FlatPPORunnerCfg,
    Go2S2DisturbancePPORunnerCfg,
    Go2S2bLightTerrainPPORunnerCfg,
)

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
