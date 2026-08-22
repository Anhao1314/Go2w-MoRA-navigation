"""自动化套件单元测试：守护调度、报告判定、SVG 曲线。"""

import unittest

from rl.go2w_env import TRAVERSE_TASKS
from rl.guard import (
    MAX_ATTEMPTS,
    RERUNS,
    Run,
    build_command,
    default_state,
    next_pending,
    should_retry,
)
from rl.report import build_verdict, make_curve_svg


def make_state(runs):
    return {
        "pipeline": {"phase12": None, "phase3a": None, "phase3b": None, "phase3c": None},
        "runs": {r.key: {"status": s} for r, s in runs},
    }


class GuardLogicTest(unittest.TestCase):
    def test_next_pending(self):
        runs = [Run("balance", 0), Run("balance", 1), Run("balance", 2)]
        state = make_state([
            (runs[0], "completed"),
            (runs[1], "pending"),
            (runs[2], "pending"),
        ])
        run = next_pending(state, runs)
        self.assertIsNotNone(run)
        assert run is not None
        self.assertEqual(run.key, "balance/seed01")

    def test_next_pending_skips_failed(self):
        runs = [Run("traverse_curve", 0), Run("traverse_curve", 1)]
        state = make_state([(runs[0], "failed"), (runs[1], "pending")])
        run = next_pending(state, runs)
        self.assertIsNotNone(run)
        assert run is not None
        self.assertEqual(run.key, "traverse_curve/seed01")

    def test_should_retry(self):
        self.assertTrue(should_retry(1))
        self.assertTrue(should_retry(MAX_ATTEMPTS))
        self.assertFalse(should_retry(MAX_ATTEMPTS + 1))

    def test_build_command_fresh(self):
        run = Run("traverse_curve", 0)
        cmd = build_command(run, fresh=True)
        self.assertIn("--fresh", cmd)
        self.assertNotIn("--resume-from", cmd)
        self.assertNotIn("--init-from", cmd)

    def test_default_state_reruns(self):
        state = default_state()
        for task in TRAVERSE_TASKS:
            self.assertEqual(state["runs"][f"{task}/seed00"]["reruns_left"], RERUNS.get(f"{task}/seed00", 0))
            self.assertFalse(state["runs"][f"{task}/seed00"]["fresh"])


class ReportLogicTest(unittest.TestCase):
    def test_balance_verdict(self):
        ok, checks = build_verdict("balance", [
            {"max_dev": "0.05", "min_clear": "0.15"},
        ])
        self.assertTrue(ok)
        self.assertEqual(len(checks), 2)

    def test_balance_verdict_fail(self):
        ok, checks = build_verdict("balance", [
            {"max_dev": "0.2", "min_clear": "0.05"},
        ])
        self.assertFalse(ok)

    def test_full_chain_verdict(self):
        ok, _ = build_verdict("full_chain", [
            {"recovered": "True", "settle_seconds": "3.0", "dual_hold": "6.0"},
        ])
        self.assertTrue(ok)

    def test_traverse_verdict(self):
        rows = [{"label": "RL PPO (slope)", "success_rate": "0.8"}]
        ok, _ = build_verdict("traverse_slope", rows)
        self.assertTrue(ok)
        bad_ok, _ = build_verdict("traverse_slope", [{"label": "RL PPO (slope)", "success_rate": "0.4"}])
        self.assertFalse(bad_ok)

    def test_curve_svg(self):
        svg = make_curve_svg([(0.0, 1.0), (1000.0, 2.0)], "test")
        self.assertIn("<svg", svg)
        self.assertIn("polyline", svg)
        self.assertIn("test", svg)


if __name__ == "__main__":
    unittest.main()
