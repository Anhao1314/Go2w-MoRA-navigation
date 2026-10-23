"""Deterministic completion-rate evaluation used for convergence early-stop.

Training-log statistics cannot serve as a stop criterion: PPO samples actions
from the stochastic policy during training, so the logged completion rate is
the rate *under exploration noise* (measured 0.045 for a policy whose
deterministic rate is 1.00).  This module runs the deterministic inference
policy on the *training* environment between training chunks.

Using the training env (2048 environments) rather than a separate 20-env
evaluation env keeps a single scene per process and gives a far larger sample;
the evaluation happens between chunks, after the rollout buffer has been
consumed, so it does not perturb training.  The measurement convention
(training env, capped rollout) differs from the frozen-seed protocol and is
recorded as such wherever it is used.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch


def deterministic_completion_rate(
    vec_env: Any,
    policy: Callable[[Any], Any],
    inner: Any,
    max_steps: int = 900,
) -> tuple[float, int]:
    """Run the deterministic policy once; return (completion_rate, steps_used).

    ``inner`` is the unwrapped manager-based env carrying ``_seq_state`` and
    ``_seq_success_log``.  Episodes are judged at their *first* termination:
    an env that terminates inside ``_seq_success_log`` succeeded (the log is
    written before the same-step reset wipes the stage); anything else, or
    still running at ``max_steps``, is a failure.
    """
    num_envs = int(inner.num_envs)
    inner._seq_success_log = set()
    results: dict[int, bool] = {}

    with torch.inference_mode():
        obs, _ = vec_env.reset()
        for step in range(max_steps):
            actions = policy(obs)
            obs, _, dones, _ = vec_env.step(actions)
            dones_np = (
                dones.cpu().numpy().astype(bool)
                if torch.is_tensor(dones)
                else np.asarray(dones, dtype=bool)
            )
            newly_done = [i for i in range(num_envs) if dones_np[i] and i not in results]
            if newly_done:
                succeeded = set(inner._seq_success_log)
                for env_id in newly_done:
                    results[env_id] = env_id in succeeded
            if len(results) == num_envs:
                return sum(results.values()) / num_envs, step + 1

    rate = sum(results.values()) / num_envs
    return rate, max_steps
