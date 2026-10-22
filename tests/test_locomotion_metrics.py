"""Tests for the frozen M1/M2 evaluation metrics and seed set.

These pin the frozen protocol values (docs/EXPERIMENT_DESIGN.md §3) so an
accidental edit of locomotion/eval is caught before a run starts.
"""

from __future__ import annotations

import unittest

import numpy as np

from locomotion.eval import (
    TRANSIENT_SKIP_S,
    EVAL_SEEDS,
    energy_per_meter,
    fall_rate,
    recovery_time,
    success_rate,
    vx_rmse,
    wz_rmse,
)


class TestSeedSet(unittest.TestCase):
    """The frozen seed set E = {0..19} is a protocol constant, not a detail."""

    def test_seeds_are_exactly_zero_to_nineteen(self) -> None:
        self.assertEqual(EVAL_SEEDS, tuple(range(20)))

    def test_seeds_are_unique_and_sorted(self) -> None:
        self.assertEqual(len(set(EVAL_SEEDS)), len(EVAL_SEEDS))
        self.assertEqual(list(EVAL_SEEDS), sorted(EVAL_SEEDS))

    def test_episode_count_is_twenty(self) -> None:
        self.assertEqual(len(EVAL_SEEDS), 20)


class TestTrackingRmse(unittest.TestCase):
    def _time_axis(self, duration_s: float = 4.0, rate_hz: float = 100.0) -> np.ndarray:
        return np.arange(0.0, duration_s, 1.0 / rate_hz)

    def test_perfect_tracking_is_zero_rmse(self) -> None:
        t = self._time_axis()
        command = np.full_like(t, 0.5)
        self.assertAlmostEqual(vx_rmse(t, command, command.copy()), 0.0, places=12)

    def test_constant_error_rmse_matches_hand_computation(self) -> None:
        t = self._time_axis()
        commanded = np.full_like(t, 1.0)
        measured = np.full_like(t, 0.9)
        # samples after 0.5 s all carry error 0.1
        self.assertAlmostEqual(vx_rmse(t, commanded, measured), 0.1, places=12)

    def test_transient_samples_are_excluded(self) -> None:
        t = self._time_axis()
        commanded = np.full_like(t, 0.0)
        measured = np.where(t <= TRANSIENT_SKIP_S, 10.0, 0.0)
        # a huge error confined to the reset transient must not leak into RMSE
        self.assertAlmostEqual(vx_rmse(t, commanded, measured), 0.0, places=12)

    def test_transient_boundary_is_strictly_after_half_second(self) -> None:
        # the protocol says "after the first 0.5 s"; a sample exactly at 0.5 s
        # is transient (excluded), the first counted sample is > 0.5 s
        t = np.array([0.0, 0.25, 0.5, 0.6])
        commanded = np.zeros(4)
        measured = np.array([5.0, 5.0, 5.0, 0.0])
        self.assertAlmostEqual(vx_rmse(t, commanded, measured), 0.0, places=12)

    def test_episode_shorter_than_transient_raises(self) -> None:
        t = np.array([0.0, 0.2, 0.4])
        command = np.zeros(3)
        with self.assertRaises(ValueError):
            vx_rmse(t, command, command.copy())

    def test_shape_mismatch_raises(self) -> None:
        t = self._time_axis()
        with self.assertRaises(ValueError):
            vx_rmse(t, np.zeros(t.size), np.zeros(t.size + 1))

    def test_wz_rmse_uses_same_definition(self) -> None:
        t = self._time_axis()
        commanded = np.full_like(t, 0.5)
        measured = np.full_like(t, 0.4)
        self.assertAlmostEqual(wz_rmse(t, commanded, measured), 0.1, places=12)


class TestFallRate(unittest.TestCase):
    def test_fraction_of_falls(self) -> None:
        self.assertAlmostEqual(fall_rate([False, False, True]), 1 / 3, places=12)
        self.assertEqual(fall_rate([False] * 20), 0.0)
        self.assertEqual(fall_rate([True] * 20), 1.0)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            fall_rate([])


class TestRecoveryTime(unittest.TestCase):
    def test_time_to_recovery(self) -> None:
        # disturbance at 1.0 s; error stays >= 0.1 until 1.5 s
        t = np.arange(0.0, 3.0, 0.1)
        commanded = np.ones_like(t)
        measured = np.where(t < 1.5, 0.8, 0.95)
        self.assertAlmostEqual(
            recovery_time(t, commanded, measured, disturbance_start_s=1.0),
            0.5,
            places=9,
        )

    def test_never_recovering_returns_none(self) -> None:
        t = np.arange(0.0, 3.0, 0.1)
        commanded = np.ones_like(t)
        measured = np.full_like(t, 0.8)
        self.assertIsNone(
            recovery_time(t, commanded, measured, disturbance_start_s=1.0)
        )

    def test_disturbance_beyond_episode_raises(self) -> None:
        t = np.arange(0.0, 1.0, 0.1)
        command = np.ones_like(t)
        with self.assertRaises(ValueError):
            recovery_time(t, command, command.copy(), disturbance_start_s=5.0)


class TestSuccessRate(unittest.TestCase):
    def test_requires_no_fall_and_within_target(self) -> None:
        no_fall = [True, True, False, True]
        within = [True, False, True, True]
        # episodes 0 and 3 satisfy both conditions
        self.assertEqual(success_rate(no_fall, within), 0.5)

    def test_shape_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            success_rate([True, True], [True])

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            success_rate([], [])


class TestEnergyPerMeter(unittest.TestCase):
    def test_simple_ratio(self) -> None:
        self.assertAlmostEqual(energy_per_meter(200.0, 4.0), 50.0, places=9)

    def test_zero_distance_returns_none(self) -> None:
        self.assertIsNone(energy_per_meter(200.0, 0.0))


class TestFrozenConstants(unittest.TestCase):
    """Pin the constants that eval_protocol --verify also checks."""

    def test_transient_skip_is_half_second(self) -> None:
        self.assertEqual(TRANSIENT_SKIP_S, 0.5)

    def test_thresholds_match_protocol_section_4(self) -> None:
        from scripts.eval_protocol import THRESHOLDS

        self.assertEqual(THRESHOLDS.vx_rmse_max, 0.15)
        self.assertEqual(THRESHOLDS.wz_rmse_max, 0.2)
        self.assertEqual(THRESHOLDS.s2_no_fall_min, 0.90)
        self.assertEqual(THRESHOLDS.s4_success_min, 0.90)


if __name__ == "__main__":
    unittest.main()
