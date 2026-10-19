"""Tests for MoRA-Train progress estimation."""

from __future__ import annotations

import unittest

from mora.progress import HeuristicProgressEstimator, NullProgressEstimator


class ProgressTests(unittest.TestCase):
    def test_null_estimator_abstains(self) -> None:
        estimator = NullProgressEstimator()
        estimate = estimator.update(1.0)
        self.assertEqual(estimate.confidence, 0.0)
        self.assertFalse(estimate.should_replan)

    def test_increasing_progress_does_not_replan(self) -> None:
        estimator = HeuristicProgressEstimator(
            stall_epsilon=0.0,
            stall_patience=2,
            replan_patience=2,
        )
        estimator.update(0.0)
        estimate = estimator.update(1.0)
        self.assertGreater(estimate.delta, 0.0)
        self.assertFalse(estimate.should_replan)

    def test_stall_raises_replan_signal(self) -> None:
        estimator = HeuristicProgressEstimator(
            stall_epsilon=0.0,
            stall_patience=2,
            replan_patience=2,
        )
        estimator.update(0.0)
        estimator.update(1.0)
        estimator.update(1.0)
        estimate = estimator.update(1.0)
        self.assertEqual(estimate.stalled_steps, 2)
        self.assertTrue(estimate.should_replan)
        self.assertEqual(estimate.confidence, 1.0)

    def test_missing_value_abstains_and_keeps_state(self) -> None:
        estimator = HeuristicProgressEstimator()
        estimator.update(0.0)
        estimate = estimator.update(None)
        self.assertEqual(estimate.confidence, 0.0)
        self.assertFalse(estimate.should_replan)
        self.assertEqual(estimate.stalled_steps, 1)

    def test_value_scale(self) -> None:
        estimator = HeuristicProgressEstimator(value_scale=10.0)
        estimate = estimator.update(5.0)
        self.assertAlmostEqual(estimate.value, 0.5)

    def test_reset(self) -> None:
        estimator = HeuristicProgressEstimator()
        estimator.update(0.0)
        estimator.reset()
        self.assertEqual(estimator.estimate.value, 0.0)
        self.assertEqual(estimator.estimate.stalled_steps, 0)


if __name__ == "__main__":
    unittest.main()
