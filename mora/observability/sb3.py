"""Stable-Baselines3 callback that emits MoRA-Train training events."""

from __future__ import annotations

from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from mora.observability.events import EventWriter

SAMPLE_KEYS = (
    "reward",
    "approx_kl",
    "value_loss",
    "memory_percent",
    "progress",
    "success_rate",
    "segment",
)


class Sb3EventCallback(BaseCallback):
    """Write ``start`` / ``sample`` / ``finish`` events around PPO training.

    The callback samples aggregate metrics at a fixed timestep interval.  It
    does not write a label: the final acceptance evaluation owns the pass/fail
    label and must append it after ``finish``.
    """

    def __init__(
        self,
        writer: EventWriter,
        *,
        sample_interval: int = 1000,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if sample_interval < 1:
            raise ValueError("sample_interval must be positive")
        self.writer = writer
        self.sample_interval = int(sample_interval)

    def _on_training_start(self) -> None:
        if not self.writer.started:
            self.writer.start()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.sample_interval != 0:
            return True
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        payload: dict[str, Any] = {"step": int(self.num_timesteps)}
        if len(rewards) > 0:
            payload["reward"] = float(np.mean(rewards))
        if isinstance(infos, list) and infos:
            self._aggregate_infos(infos, payload)
        self.writer.sample(payload)
        return True

    @staticmethod
    def _aggregate_infos(
        infos: list[dict[str, Any]], payload: dict[str, Any]
    ) -> None:
        for key in SAMPLE_KEYS:
            if key == "success_rate":
                values = [
                    info.get("success_rate", info.get("goal"))
                    for info in infos
                    if isinstance(info, dict)
                    and ("success_rate" in info or "goal" in info)
                ]
            else:
                values = [
                    info[key]
                    for info in infos
                    if isinstance(info, dict) and key in info
                ]
            values = [value for value in values if value is not None]
            if not values:
                continue
            if key == "success_rate":
                payload[key] = float(np.mean([bool(value) for value in values]))
            else:
                try:
                    payload[key] = float(np.mean([float(value) for value in values]))
                except (TypeError, ValueError):
                    continue

    def _on_training_end(self) -> None:
        if self.writer.started and not self.writer.finished:
            self.writer.finish(
                {
                    "status": "completed",
                    "final_step": int(self.num_timesteps),
                }
            )
        self.writer.close()
