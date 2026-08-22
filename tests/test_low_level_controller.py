"""LowLevelController 单元测试：输出形状、教师一致性、高层指令、边界。"""

import unittest

import mujoco
import numpy as np

from rl.go2w_env import Go2wEnv, curve_center_y
from rl.low_level_controller import LowLevelController
from scripts.gen_demo_trajectory import (
    compute_teacher_action,
    default_params,
)


class LowLevelControllerTest(unittest.TestCase):
    def setUp(self):
        self.controller = LowLevelController()

    def test_output_shape_and_dtype(self):
        a = self.controller.compute_action(1.0, 0.0, 0.0, {})
        self.assertEqual(a.shape, (6,))
        self.assertEqual(a.dtype, np.float32)

    def test_default_matches_teacher(self):
        p = default_params()
        for x, y, yaw in [(0.9, 0.1, 0.2), (1.5, -0.1, -0.3), (3.0, 0.05, 0.1)]:
            hl = self.controller.compute_action(
                x, y, yaw, {"speed_scale": 1.0, "turn_adjust": 0.0}
            )
            teacher = compute_teacher_action(x, y, yaw, p)
            np.testing.assert_allclose(hl, teacher, atol=1e-6)

    def test_speed_scale(self):
        # 中心线上且 yaw=前视目标角 → teacher_bias≈0，便于验证 forward 缩放
        x = 0.9
        y = float(curve_center_y(x))
        xt = x + 0.7
        yaw = float(np.arctan2(curve_center_y(xt) - y, xt - x))
        base = self.controller.compute_action(x, y, yaw, {})
        half = self.controller.compute_action(
            x, y, yaw, {"speed_scale": 0.5, "turn_adjust": 0.0}
        )
        np.testing.assert_allclose(half[:4], base[:4] * 0.5, atol=1e-6)
        np.testing.assert_allclose(half[4:], base[4:], atol=1e-6)

    def test_turn_adjust(self):
        x = 0.9
        y = float(curve_center_y(x))
        xt = x + 0.7
        yaw = float(np.arctan2(curve_center_y(xt) - y, xt - x))
        base = self.controller.compute_action(x, y, yaw, {})
        adjusted = self.controller.compute_action(
            x, y, yaw, {"speed_scale": 1.0, "turn_adjust": 0.3}
        )
        self.assertAlmostEqual(float(adjusted[0]), float(base[0]) - 0.3, places=5)
        self.assertAlmostEqual(float(adjusted[1]), float(base[1]) + 0.3, places=5)

    def test_clip_bounds(self):
        a = self.controller.compute_action(
            1.0, 0.0, 0.0, {"speed_scale": 5.0, "turn_adjust": 5.0}
        )
        self.assertTrue(np.all(np.abs(a) <= 1.0))

    def test_get_state_from_env(self):
        env = Go2wEnv(task="traverse_curve", domain_randomize=False)
        env.reset(seed=0)
        env.data.qpos[0] = 1.5
        env.data.qpos[1] = 0.2
        mujoco.mj_forward(env.model, env.data)
        x, y, yaw = self.controller.get_state_from_env(env)
        self.assertAlmostEqual(x, 1.5)
        self.assertAlmostEqual(y, 0.2)
        self.assertIsInstance(yaw, float)
        env.close()


if __name__ == "__main__":
    unittest.main()
