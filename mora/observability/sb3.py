"""Stable-Baselines3 callback that emits MoRA-Train training events."""

from __future__ import annotations

import math
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
        manage_lifecycle: bool = True,
        step_offset: int = 0,
        verbose: int = 0,
    ) -> None:
        super().__init__(verbose=verbose)
        if sample_interval < 1:
            raise ValueError("sample_interval must be positive")
        if step_offset < 0:
            raise ValueError("step_offset must be non-negative")
        self.writer = writer
        self.sample_interval = int(sample_interval)
        self.manage_lifecycle = bool(manage_lifecycle)
        self.step_offset = int(step_offset)
        self._stage_offset = int(step_offset)
        self._stage_start_step: int | None = None

    def set_step_offset(self, offset: int) -> None:
        """Set the global step offset when a curriculum stage reloads a model."""
        if offset < 0:
            raise ValueError("step_offset must be non-negative")
        self.step_offset = int(offset)
        self._stage_start_step = None

    @property
    def global_step(self) -> int:
        if self._stage_start_step is None:
            return self.step_offset + int(self.num_timesteps)
        return self._stage_offset + (
            int(self.num_timesteps) - self._stage_start_step
        )

    def _on_training_start(self) -> None:
        # SB3 resets the callback's counter from the newly loaded model at the
        # start of each curriculum stage.  Capture that local origin so the
        # event stream stays monotonic across model reloads.
        self._stage_offset = self.step_offset
        self._stage_start_step = int(self.num_timesteps)
        if self.manage_lifecycle and not self.writer.started:
            self.writer.start()

    def _on_step(self) -> bool:
        step = self.global_step
        if step % self.sample_interval != 0:
            return True
        infos = self.locals.get("infos", [])
        rewards = self.locals.get("rewards", [])
        payload: dict[str, Any] = {"step": step}
        reward = self._finite_mean(rewards)
        if reward is not None:
            payload["reward"] = reward
        model = getattr(self, "model", None)
        logger = getattr(model, "logger", None)
        logger_values = getattr(logger, "name_to_value", {}) if logger else {}
        for key, name in (
            ("approx_kl", "train/approx_kl"),
            ("value_loss", "train/value_loss"),
        ):
            value = logger_values.get(name)
            if value is None:
                continue
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                payload[key] = number
        if isinstance(infos, list) and infos:
            self._aggregate_infos(infos, payload)
        self.writer.sample(payload)
        return True

    @staticmethod
    def _finite_mean(values: list[Any]) -> float | None:
        clean: list[float] = []
        for value in values:
            try:
                number = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(number):
                clean.append(number)
        if not clean:
            return None
        return float(np.mean(clean))

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
                mean = Sb3EventCallback._finite_mean(values)
                if mean is not None:
                    payload[key] = mean

    def _on_training_end(self) -> None:
        if not self.manage_lifecycle:
            return
        if self.writer.started and not self.writer.finished:
            self.writer.finish({"status": "completed"})
        self.writer.close()
