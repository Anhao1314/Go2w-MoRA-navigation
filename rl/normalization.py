"""Single-layer VecNormalize construction for training and evaluation."""

from __future__ import annotations

import pathlib
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper, VecNormalize


def normalize_once(
    raw_env: VecEnv, checkpoint: str | pathlib.Path | None = None,
    *, training: bool = True,
) -> VecNormalize:
    """Attach one normalizer to a raw vector environment, never to another one.

    Checkpoints use pickle and must come from a trusted source. Loading does not
    repair statistics saved by an older, double-normalized training run.
    """
    current = raw_env
    while isinstance(current, VecEnvWrapper):
        if isinstance(current, VecNormalize):
            raise ValueError("VecNormalize is already present; pass a raw VecEnv")
        current = current.venv
    if checkpoint is None:
        result = VecNormalize(raw_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    else:
        path = pathlib.Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(path)
        result = VecNormalize.load(str(path), raw_env)
    result.training = training
    result.norm_reward = False
    return result
