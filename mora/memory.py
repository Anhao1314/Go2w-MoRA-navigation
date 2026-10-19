"""Multi-granularity memory.

This is the first, inspectable version of the MoRA-inspired memory layer.  It
is deliberately not a neural Memory Token implementation: the runtime exposes
three explicit time scales, and a future policy can consume their snapshot or
learn an internal representation from it.

- short-term: recent transitions (control/execution continuity)
- episodic: completed episode summaries (what happened, outcome, progress)
- task: persistent key/value state (instructions, maps, branch decisions)
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable

from mora.types import EpisodeSummary, MemorySnapshot, Transition


class ShortTermMemory:
    """Fixed-capacity recent-transition buffer."""

    def __init__(self, capacity: int = 64) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._items: deque[Transition] = deque(maxlen=self.capacity)

    def push(self, transition: Transition) -> None:
        self._items.append(transition)

    def recent(self, n: int | None = None) -> tuple[Transition, ...]:
        if n is None or n >= len(self._items):
            return tuple(self._items)
        if n <= 0:
            return ()
        return tuple(list(self._items)[-n:])

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


class EpisodicMemory:
    """Bounded store of completed episode summaries."""

    def __init__(self, capacity: int = 32) -> None:
        if capacity < 1:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._items: deque[EpisodeSummary] = deque(maxlen=self.capacity)

    def record(self, summary: EpisodeSummary) -> None:
        self._items.append(summary)

    def recent(self, n: int | None = None) -> tuple[EpisodeSummary, ...]:
        if n is None or n >= len(self._items):
            return tuple(self._items)
        if n <= 0:
            return ()
        return tuple(list(self._items)[-n:])

    def clear(self) -> None:
        self._items.clear()

    def __len__(self) -> int:
        return len(self._items)


class TaskMemory:
    """Persistent task-level state shared across episodes.

    Examples: target label, map metadata, selected branch, or a human
    instruction.  The runtime never clears this on ``reset_episode``; callers
    decide when task context changes.
    """

    def __init__(self) -> None:
        self._state: dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._state.get(key, default)

    def update(self, values: dict[str, Any]) -> None:
        self._state.update(values)

    def snapshot(self) -> dict[str, Any]:
        return dict(self._state)

    def clear(self) -> None:
        self._state.clear()


class MultiGranularityMemory:
    """Aggregate short-term, episodic, and task memory."""

    def __init__(
        self,
        short_term_capacity: int = 64,
        episodic_capacity: int = 32,
    ) -> None:
        self.short_term = ShortTermMemory(short_term_capacity)
        self.episodic = EpisodicMemory(episodic_capacity)
        self.task = TaskMemory()

    def observe(self, transition: Transition) -> None:
        self.short_term.push(transition)

    def record_episode(self, summary: EpisodeSummary) -> None:
        self.episodic.record(summary)

    def reset_episode(self) -> None:
        """Clear within-episode memory; task memory intentionally persists."""
        self.short_term.clear()

    def snapshot(self) -> MemorySnapshot:
        return MemorySnapshot(
            recent=self.short_term.recent(),
            episodes=self.episodic.recent(),
            task_state=self.task.snapshot(),
        )

    def extend_recent(self, transitions: Iterable[Transition]) -> None:
        for transition in transitions:
            self.observe(transition)

    def clear(self) -> None:
        self.short_term.clear()
        self.episodic.clear()
        self.task.clear()
