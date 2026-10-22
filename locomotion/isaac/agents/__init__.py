"""Agent configurations for the Go2 locomotion tasks (RSL-RL PPO)."""

from locomotion.isaac.agents.rsl_rl_ppo_cfg import (
    Go2S1FlatPPORunnerCfg,
    Go2S2DisturbancePPORunnerCfg,
    Go2S2bLightTerrainPPORunnerCfg,
    Go2VelocityPPORunnerCfg,
)

__all__ = [
    "Go2VelocityPPORunnerCfg",
    "Go2S1FlatPPORunnerCfg",
    "Go2S2DisturbancePPORunnerCfg",
    "Go2S2bLightTerrainPPORunnerCfg",
]
