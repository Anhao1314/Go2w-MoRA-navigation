"""Smoke test for the first MoRA-Train task adapter (Go2W multi-segment)."""

from __future__ import annotations

import unittest

import numpy as np

from mora.adapters.go2w import Go2wTaskAdapter


class Go2wAdapterTests(unittest.TestCase):
    def test_multi_segment_contract(self) -> None:
        adapter = Go2wTaskAdapter("multi-segment", scope=2)
        try:
            observation, info = adapter.reset(seed=0)
            self.assertEqual(observation.shape, adapter.observation_space.shape)
            self.assertIsInstance(info, dict)

            goal = adapter.goal()
            self.assertEqual(goal.kind, "subgoal")
            self.assertIsNotNone(goal.position)
            self.assertIn("total", goal.metadata)

            action = np.zeros(adapter.action_space.shape, dtype=np.float32)
            step = adapter.step(action)
            self.assertEqual(len(step), 5)
            self.assertIsInstance(step[1], float)
            self.assertIsInstance(step[4], dict)
            self.assertIsInstance(adapter.progress(), float)
            self.assertIsInstance(adapter.success(), bool)

            metadata = adapter.metadata()
            self.assertEqual(metadata["task"], "multi-segment")
            self.assertTrue(metadata["multi_segment"])
        finally:
            adapter.close()


if __name__ == "__main__":
    unittest.main()
