"""Tests for MoRA-Train multi-granularity memory."""

from __future__ import annotations

import unittest

import numpy as np

from mora.memory import (
    EpisodicMemory,
    MultiGranularityMemory,
    ShortTermMemory,
    TaskMemory,
)
from mora.types import EpisodeSummary, Goal, Transition


def _transition(step: int) -> Transition:
    return Transition(
        observation=np.zeros(2, dtype=np.float32),
        action=np.array([float(step)], dtype=np.float32),
        reward=0.0,
        terminated=False,
        truncated=False,
        info={"step": step},
        progress=float(step),
    )


def _summary(step: int) -> EpisodeSummary:
    return EpisodeSummary(
        goal=Goal(kind="test", label=f"goal-{step}"),
        steps=step,
        total_reward=1.0,
        success=True,
        max_progress=float(step),
        terminated=True,
        truncated=False,
        interventions=0,
    )


class MemoryTests(unittest.TestCase):
    def test_short_term_capacity(self) -> None:
        memory = ShortTermMemory(capacity=2)
        memory.push(_transition(1))
        memory.push(_transition(2))
        memory.push(_transition(3))
        self.assertEqual(len(memory), 2)
        self.assertEqual([t.info["step"] for t in memory.recent()], [2, 3])
        self.assertEqual([t.info["step"] for t in memory.recent(1)], [3])
        memory.clear()
        self.assertEqual(len(memory), 0)

    def test_episodic_capacity(self) -> None:
        memory = EpisodicMemory(capacity=2)
        memory.record(_summary(1))
        memory.record(_summary(2))
        memory.record(_summary(3))
        self.assertEqual(len(memory), 2)
        self.assertEqual([e.steps for e in memory.recent()], [2, 3])

    def test_task_memory_persists_across_episode_reset(self) -> None:
        memory = TaskMemory()
        memory.set("target", "A")
        memory.update({"branch": "left"})
        self.assertEqual(memory.get("target"), "A")
        memory.clear()
        self.assertIsNone(memory.get("target"))

    def test_multi_granularity_snapshot(self) -> None:
        memory = MultiGranularityMemory(
            short_term_capacity=2, episodic_capacity=2
        )
        memory.observe(_transition(1))
        memory.observe(_transition(2))
        memory.record_episode(_summary(2))
        memory.task.set("task_id", "multi-segment")
        snapshot = memory.snapshot()
        self.assertEqual(len(snapshot.recent), 2)
        self.assertEqual(len(snapshot.episodes), 1)
        self.assertEqual(snapshot.task_state["task_id"], "multi-segment")
        memory.reset_episode()
        self.assertEqual(len(memory.short_term), 0)
        self.assertEqual(len(memory.episodic), 1)
        self.assertEqual(memory.task.get("task_id"), "multi-segment")


if __name__ == "__main__":
    unittest.main()
