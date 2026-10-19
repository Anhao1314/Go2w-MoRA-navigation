"""Append-only v0.2-style training event writer.

The writer is the producer side of the RL-Sentinel contract.  It enforces the
timing invariant ``event_time <= available_at`` at the source and refuses
unknown lifecycle transitions; the consumer still performs its own strict
validation and evidence-bundle checks.

The first MoRA-Train task metrics (progress, success rate, segment) are written
as sample fields.  The RL-Sentinel Go2W ingester will accept them as an
explicit schema extension rather than silently treating them as legacy fields.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

ORIGINS = ("observed", "controlled", "synthetic")
KINDS = ("start", "sample", "finish", "label")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "item") and callable(value.item):
        return value.item()
    if hasattr(value, "tolist") and callable(value.tolist):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return value


class EventWriter:
    """Write one run's JSONL event stream.

    Parameters
    ----------
    path:
        Destination JSONL path.  The file is created exclusively by default so
        a run cannot accidentally append to another run's evidence.
    run_id:
        Stable identifier for the run.
    task, seed, planned_steps:
        Defaults for the ``start`` event.  ``planned_steps`` must be positive.
    """

    def __init__(
        self,
        path: str | Path,
        run_id: str,
        *,
        origin: str = "observed",
        task: str | None = None,
        seed: int | None = None,
        planned_steps: int | None = None,
        append: bool = False,
    ) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        if origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}")
        if planned_steps is not None and planned_steps < 1:
            raise ValueError("planned_steps must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self.origin = origin
        self.task = task
        self.seed = seed
        self.planned_steps = planned_steps
        self._sequence = 0
        self._started = False
        self._finished = False
        self._closed = False
        mode = "a" if append else "x"
        self._handle = open(self.path, mode, encoding="utf-8")
        self._write_line({"schema_version": 2, "origin": origin})

    @property
    def started(self) -> bool:
        return self._started

    @property
    def finished(self) -> bool:
        return self._finished

    def _write_line(self, payload: Mapping[str, Any]) -> None:
        json.dump(
            _jsonable(payload),
            self._handle,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        )
        self._handle.write("\n")
        self._handle.flush()

    def emit(
        self,
        kind: str,
        payload: Mapping[str, Any] | None = None,
        *,
        event_time: float | None = None,
        available_at: float | None = None,
    ) -> dict[str, Any]:
        if self._closed:
            raise RuntimeError("event writer is closed")
        if kind not in KINDS:
            raise ValueError(f"unknown event kind: {kind}")
        now = time.time()
        event_time = now if event_time is None else float(event_time)
        available_at = now if available_at is None else float(available_at)
        for name, value in (("event_time", event_time), ("available_at", available_at)):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if available_at < event_time:
            raise ValueError("available_at must not precede event_time")

        if kind == "start":
            if self._started:
                raise ValueError("run already started")
            self._started = True
        elif kind == "sample":
            if not self._started or self._finished:
                raise ValueError("sample requires an active run")
        elif kind == "finish":
            if not self._started or self._finished:
                raise ValueError("finish requires an active run")
            self._finished = True
        else:  # label
            if not self._finished:
                raise ValueError("label requires a finished run")

        self._sequence += 1
        event = {
            "schema_version": 2,
            "event_id": f"{self.run_id}-e{self._sequence}",
            "run_id": self.run_id,
            "kind": kind,
            "event_time": event_time,
            "available_at": available_at,
            "payload": _jsonable(payload or {}),
        }
        self._write_line(event)
        return event

    def start(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        event_time: float | None = None,
        available_at: float | None = None,
    ) -> dict[str, Any]:
        data: dict[str, Any] = {}
        if self.task is not None:
            data["task"] = self.task
        if self.seed is not None:
            data["seed"] = str(self.seed)
        if self.planned_steps is not None:
            data["planned_steps"] = self.planned_steps
        data.update(payload or {})
        if "seed" in data:
            data["seed"] = str(data["seed"])
        return self.emit(
            "start", data, event_time=event_time, available_at=available_at
        )

    def sample(
        self,
        payload: Mapping[str, Any],
        *,
        event_time: float | None = None,
        available_at: float | None = None,
    ) -> dict[str, Any]:
        return self.emit(
            "sample", payload, event_time=event_time, available_at=available_at
        )

    def finish(
        self,
        payload: Mapping[str, Any] | None = None,
        *,
        event_time: float | None = None,
        available_at: float | None = None,
    ) -> dict[str, Any]:
        return self.emit(
            "finish", payload or {}, event_time=event_time, available_at=available_at
        )

    def label(
        self,
        payload: Mapping[str, Any],
        *,
        event_time: float | None = None,
        available_at: float | None = None,
    ) -> dict[str, Any]:
        return self.emit(
            "label", payload, event_time=event_time, available_at=available_at
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self._handle.close()

    def __enter__(self) -> "EventWriter":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()
