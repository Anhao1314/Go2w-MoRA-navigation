"""Goal-Conditioned 高层环境测试：63 维观测、目标计算、奖励不变。"""

import unittest

import mujoco
import numpy as np

from rl.go2w_env import GOAL_X, Go2wEnv
from rl.high_level_env_wrapper import HighLevelEnvWrapper, relabel_episode
from rl.low_level_controller import LowLevelController


class GoalConditionedEnvTest(unittest.TestCase):
    def _pair(self, seed: int = 5):
        base = Go2wEnv(task="traverse_curve", domain_randomize=False,
                       reward_version="v5", fall_penalty=-10.0, goal_bonus=200.0)
        wrapped = HighLevelEnvWrapper(
            Go2wEnv(task="traverse_curve", domain_randomize=False,
                    reward_version="v5", fall_penalty=-10.0, goal_bonus=200.0)
        )
        base.reset(seed=seed)
        wrapped.reset(seed=seed)
        return base, wrapped

    def test_observation_space_63dim(self):
        _, wrapped = self._pair()
        self.assertEqual(wrapped.observation_space.shape, (63,))

    def test_reset_returns_63dim_with_goal(self):
        base, wrapped = self._pair()
        obs_b, _ = base.reset(seed=11, options={"goal_arc": 1.0})
        obs_w, info = wrapped.reset(seed=11, options={"goal_arc": 1.0})
        self.assertEqual(obs_w.shape, (63,))
        np.testing.assert_allclose(obs_w[:61], obs_b, atol=1e-6)
        self.assertGreater(float(obs_w[-2]), 0.0)
        self.assertGreaterEqual(float(obs_w[-1]), -np.pi)
        self.assertLessEqual(float(obs_w[-1]), np.pi)
        self.assertIsInstance(info, dict)
        base.close()
        wrapped.close()

    def test_step_returns_63dim_and_goal_updates(self):
        _, wrapped = self._pair()
        obs, _ = wrapped.reset(seed=3)
        remaining_before = float(obs[-2])
        for _ in range(20):
            obs, _r, term, trunc, _i = wrapped.step(
                np.array([0.9, 0.0], dtype=np.float32)
            )
            if term or trunc:
                break
        self.assertEqual(obs.shape, (63,))
        self.assertLessEqual(float(obs[-2]), remaining_before)
        wrapped.close()

    def test_remaining_arc_decreases(self):
        _, wrapped = self._pair()
        obs, _ = wrapped.reset(seed=4)
        r0 = float(obs[-2])
        env = wrapped.base_env
        env.data.qpos[0] = 1.0
        mujoco.mj_forward(env.model, env.data)
        self.assertLess(float(wrapped._compute_goal()[0]), r0)
        wrapped.close()

    def test_target_heading_in_range(self):
        _, wrapped = self._pair()
        wrapped.reset(seed=6)
        for _ in range(40):
            obs, _r, term, trunc, _i = wrapped.step(
                np.array([1.0, 0.0], dtype=np.float32)
            )
            self.assertGreaterEqual(float(obs[-1]), -np.pi)
            self.assertLessEqual(float(obs[-1]), np.pi)
            if term or trunc:
                break
        wrapped.close()

    def test_goal_at_finish_is_zero(self):
        _, wrapped = self._pair()
        wrapped.reset(seed=8)
        env = wrapped.base_env
        env.data.qpos[0] = GOAL_X
        mujoco.mj_forward(env.model, env.data)
        goal = wrapped._compute_goal()
        self.assertAlmostEqual(float(goal[0]), 0.0, places=3)

    def test_reward_unchanged_vs_base(self):
        base, wrapped = self._pair(seed=123)
        controller = LowLevelController()
        for _ in range(10):
            x, y, yaw = controller.get_state_from_env(base)
            low = controller.compute_action(x, y, yaw, {})
            _ob, rew_b, term_b, trunc_b, _i = base.step(low)
            _ow, rew_w, term_w, trunc_w, _i = wrapped.step(
                np.array([1.0, 0.0], dtype=np.float32)
            )
            self.assertAlmostEqual(float(rew_w), float(rew_b), places=6)
            self.assertEqual(term_w, term_b)
            self.assertEqual(trunc_w, trunc_b)
            if term_b or trunc_b:
                break
        base.close()
        wrapped.close()

    def test_goal_arc_parameter_and_clamp(self):
        env = Go2wEnv(task="traverse_curve", domain_randomize=False, goal_arc=0.8)
        env.reset(seed=0)
        self.assertAlmostEqual(env.get_goal_arc(), 0.8, places=4)
        env2 = Go2wEnv(task="traverse_curve", domain_randomize=False, goal_arc=2.0)
        env2.reset(seed=0)
        self.assertLessEqual(env2.get_goal_arc(), 1.40098)
        env3 = Go2wEnv(task="traverse_curve", domain_randomize=False, goal_arc=0.1)
        env3.reset(seed=0)
        self.assertGreaterEqual(env3.get_goal_arc(), 0.3)
        env.close()
        env2.close()
        env3.close()

    def test_reset_options_sets_goal(self):
        env = Go2wEnv(task="traverse_curve", domain_randomize=False)
        env.reset(seed=0, options={"goal_arc": 0.9})
        self.assertAlmostEqual(env.get_goal_arc(), 0.9, places=4)
        self.assertAlmostEqual(env.get_remaining_arc(), 0.9, places=4)
        env.close()

    def test_use_goal_condition_false_returns_61dim(self):
        wrapped = HighLevelEnvWrapper(
            Go2wEnv(task="traverse_curve", domain_randomize=False),
            use_goal_condition=False,
            goal_min=1.0,
            goal_max=1.0,
        )
        obs, _ = wrapped.reset(seed=0, options={"goal_arc": 1.0})
        self.assertEqual(wrapped.observation_space.shape, (61,))
        self.assertEqual(obs.shape, (61,))
        self.assertEqual(float(obs[60]), 0.0)  # 底层 remaining 被屏蔽
        wrapped.close()

    def test_relabel_episode(self):
        transitions = [
            {"base_obs": np.zeros(61, dtype=np.float32), "action": np.zeros(2, dtype=np.float32),
             "achieved_arc": 0.2, "reward": 0.0},
            {"base_obs": np.zeros(61, dtype=np.float32), "action": np.zeros(2, dtype=np.float32),
             "achieved_arc": 0.5, "reward": 0.0},
            {"base_obs": np.zeros(61, dtype=np.float32), "action": np.zeros(2, dtype=np.float32),
             "achieved_arc": 0.8, "reward": 0.0},
        ]
        samples = relabel_episode(
            transitions, n_sampled_goal=1, use_goal_condition=True,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(len(samples), 2)  # 最后一步无 future
        obs, act, rew = samples[0]
        self.assertEqual(obs.shape, (63,))
        self.assertEqual(act.shape, (2,))
        self.assertIn(rew, (0.0, 1.0))


if __name__ == "__main__":
    unittest.main()
