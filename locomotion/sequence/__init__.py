"""Task registrations for the four-stage command-sequence task.

Imports nothing from ``isaaclab`` at module level so this package can be
imported before Kit starts (same contract as :mod:`locomotion.isaac`).
"""

from __future__ import annotations

import gymnasium as gym

gym.register(
    id="Go2Loco-Sequence-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            "locomotion.sequence.sequence_env_cfg:Go2SequenceEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            "locomotion.isaac.agents.rsl_rl_ppo_cfg:Go2S1FlatPPORunnerCfg"
        ),
    },
)
