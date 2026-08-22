"""多段路径环境（MoRA 阶段2）单元测试。"""

import math
import unittest

import numpy as np

from rl.go2w_env import Go2wEnv
from scripts.gen_demo_multi_segment import compute_action, default_params


class MultiSegmentEnvTest(unittest.TestCase):
    def _env(self, **kw):
        return Go2wEnv(task="traverse_curve", domain_randomize=False,
                       multi_segment=True, **kw)

    def test_geometry_and_goals(self):
        env = self._env()
        env.reset(seed=0)
        ax, ay = env._segment_goal(0)
        bx, by = env._segment_goal(1)
        self.assertAlmostEqual(ax, 4.3)
        self.assertAlmostEqual(ay, 0.0, places=6)
        self.assertAlmostEqual(bx, 10.8)
        self.assertAlmostEqual(by, 0.0, places=6)
        self.assertAlmostEqual(
            env._segment_center_y(4.3, 0), env._segment_center_y(4.3, 1), places=6
        )
        self.assertTrue(math.isfinite(env._segment_tangent_angle(4.3, 0)))
        self.assertTrue(math.isfinite(env._segment_tangent_angle(4.3, 1)))
        env.close()

    def test_subgoal_hold_detection(self):
        env = self._env()
        env.reset(seed=0)
        ax, ay = env._segment_goal(0)
        env.data.qpos[0] = ax
        env.data.qpos[1] = ay
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        env.data.time = 10.0
        self.assertFalse(env._check_subgoal_reached(ax, ay, 0.0))
        env.data.time = 10.35
        self.assertTrue(env._check_subgoal_reached(ax, ay, 0.0))
        env.close()

    def test_switch_to_next_segment(self):
        env = self._env()
        env.reset(seed=0)
        self.assertEqual(env._current_segment(), 0)
        env._switch_to_next_segment()
        self.assertEqual(env._current_segment(), 1)
        self.assertTrue(env._subgoal_reached_a)
        self.assertAlmostEqual(env._goal_arc_len(), 10.8)
        env.close()

    def test_wall_termination_per_segment(self):
        env = self._env()
        env.reset(seed=0)
        import mujoco
        # 段1：dev=0.6 > 0.50 触发终止
        env.data.qpos[0] = 2.0
        env.data.qpos[1] = float(env._segment_center_y(2.0, 0)) + 0.6
        mujoco.mj_forward(env.model, env.data)
        self.assertTrue(env._check_termination())
        # 段2：dev=0.5 > 0.40 触发终止
        env._segment_idx = 1
        env.data.qpos[0] = 6.0
        env.data.qpos[1] = float(env._segment_center_y(6.0, 1)) + 0.5
        mujoco.mj_forward(env.model, env.data)
        self.assertTrue(env._check_termination())
        # 段2内 dev=0.2 不终止
        env.data.qpos[1] = float(env._segment_center_y(6.0, 1)) + 0.2
        mujoco.mj_forward(env.model, env.data)
        self.assertFalse(env._check_termination())
        env.close()

    def test_backward_compat_default(self):
        env = Go2wEnv(task="traverse_curve", domain_randomize=False)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (61,))
        self.assertFalse(env.multi_segment)
        self.assertAlmostEqual(env._goal_arc_len(), 1.400972, places=4)
        env.close()

    def test_controller_action_bounds(self):
        env = self._env()
        env.reset(seed=0)
        params = default_params()
        action, scale = compute_action(env, params)
        self.assertEqual(action.shape, (6,))
        self.assertTrue(np.all(np.abs(action) <= 1.0))
        self.assertEqual(scale, 1.0)
        env.close()


if __name__ == "__main__":
    unittest.main()
