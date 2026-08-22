"""DAgger 教师路线纯函数测试：教师动作、犯错状态收集、停止条件。"""

import unittest

import numpy as np

from rl.go2w_env import curve_center_y
from scripts.dagger_train import collect_error_indices, decide_stop
from scripts.gen_demo_trajectory import compute_teacher_action, default_params


class TeacherActionTest(unittest.TestCase):
    def test_shape_and_bounds(self):
        p = default_params()
        a = compute_teacher_action(1.0, 0.0, 0.0, p)
        self.assertEqual(a.shape, (6,))
        self.assertEqual(a.dtype, np.float32)
        self.assertTrue(np.all(np.abs(a) <= 1.0))
        self.assertTrue(np.allclose(a[4:], 0.0))

    def test_centered_yaw_gives_symmetric_action(self):
        p = default_params()
        # 中心线上、yaw=前视目标角 → bias≈0，动作对称
        x = 0.9
        y = float(curve_center_y(x))
        xt = x + p["lookahead"]
        yt = float(np.arctan2(curve_center_y(xt) - y, xt - x))
        a = compute_teacher_action(x, y, float(yt), p)
        self.assertAlmostEqual(float(a[0]), float(a[1]), places=4)
        self.assertAlmostEqual(float(a[0]), p["forward"], places=3)

    def test_heading_error_creates_differential(self):
        p = default_params()
        a = compute_teacher_action(1.0, 0.0, 0.8, p)  # 明显偏航
        self.assertNotAlmostEqual(float(a[0]), float(a[1]), places=3)


class ErrorStateCollectTest(unittest.TestCase):
    def test_pre_fall_and_thresholds(self):
        devs = [0.0, 0.3, 0.0, 0.0, 0.0]
        hers = [0.0, 0.0, 0.5, 0.0, 0.0]
        idx = collect_error_indices(devs, hers, k_fall=2, fell=True)
        self.assertIn(1, idx)   # |dev|>0.20
        self.assertIn(2, idx)   # |herr|>0.30
        self.assertIn(3, idx)   # 摔倒前最后 2 步（3,4）
        self.assertIn(4, idx)
        self.assertEqual(set(idx), {1, 2, 3, 4})

    def test_dedupe_and_cap(self):
        devs = [0.0, 0.5, 0.5, 0.5, 0.5, 0.5]
        idx = collect_error_indices(devs, [0.0] * 6, k_fall=0, max_states=3)
        self.assertEqual(idx, [1, 2, 3])
        self.assertEqual(len(idx), 3)

    def test_no_fall_no_threshold(self):
        idx = collect_error_indices([0.0, 0.0], [0.0, 0.0], fell=False)
        self.assertEqual(idx, [])


class StopConditionTest(unittest.TestCase):
    def _row(self, dist: float) -> dict:
        return {"mean_distance": dist}

    def test_success(self):
        stop, reason = decide_stop(
            [self._row(0.2), self._row(2.1)], max_iter=5
        )
        self.assertTrue(stop)
        self.assertEqual(reason, "success")

    def test_converged(self):
        stop, reason = decide_stop(
            [self._row(0.2), self._row(0.4), self._row(0.45), self._row(0.48)],
            max_iter=5,
        )
        self.assertTrue(stop)
        self.assertEqual(reason, "converged")

    def test_max_iter(self):
        rows = [self._row(0.5 + i * 0.2) for i in range(6)]
        stop, reason = decide_stop(rows, max_iter=5)
        self.assertTrue(stop)
        self.assertEqual(reason, "max_iter")

    def test_not_stop(self):
        stop, _reason = decide_stop([self._row(0.2)], max_iter=5)
        self.assertFalse(stop)


if __name__ == "__main__":
    unittest.main()
