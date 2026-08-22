"""HighLevelEnvWrapper 单元测试：动作/观测空间、step/reset、奖励一致性。"""

import unittest

from gymnasium import spaces
import numpy as np
from typing import cast

from rl.go2w_env import Go2wEnv
from rl.high_level_env_wrapper import HighLevelEnvWrapper
from rl.low_level_controller import LowLevelController


def make_pair(seed: int = 42):
    base = Go2wEnv(task="traverse_curve", domain_randomize=False,
                   reward_version="v5", fall_penalty=-10.0, goal_bonus=200.0)
    wrapped = HighLevelEnvWrapper(
        Go2wEnv(task="traverse_curve", domain_randomize=False,
                reward_version="v5", fall_penalty=-10.0, goal_bonus=200.0)
    )
    base.reset(seed=seed)
    wrapped.reset(seed=seed)
    return base, wrapped


class HighLevelEnvWrapperTest(unittest.TestCase):
    def test_action_space(self):
        _, wrapped = make_pair()
        self.assertIsInstance(wrapped.action_space, spaces.Box)
        box = cast(spaces.Box, wrapped.action_space)
        self.assertEqual(box.shape, (2,))
        np.testing.assert_allclose(box.low, [0.9, -0.5])
        np.testing.assert_allclose(box.high, [1.5, 0.5])
        self.assertEqual(box.dtype, np.float32)

    def test_min_speed_action_runs(self):
        _, wrapped = make_pair()
        wrapped.reset(seed=2)
        obs, _r, term, trunc, _i = wrapped.step(
            np.array([0.9, 0.0], dtype=np.float32)
        )
        self.assertEqual(obs.shape, wrapped.observation_space.shape)

    def test_observation_space_extended(self):
        _, wrapped = make_pair()
        base = Go2wEnv(task="traverse_curve", domain_randomize=False)
        self.assertIsInstance(wrapped.observation_space, spaces.Box)
        self.assertIsInstance(base.observation_space, spaces.Box)
        wrapped_box = cast(spaces.Box, wrapped.observation_space)
        base_box = cast(spaces.Box, base.observation_space)
        self.assertEqual(wrapped_box.shape, (base_box.shape[0] + 2,))
        np.testing.assert_allclose(
            wrapped_box.low[: base_box.shape[0]], base_box.low
        )
        np.testing.assert_allclose(
            wrapped_box.high[: base_box.shape[0]], base_box.high
        )
        self.assertAlmostEqual(float(wrapped_box.low[-2]), 0.0)
        np.testing.assert_allclose(float(wrapped_box.low[-1]), -np.pi, atol=1e-6)
        np.testing.assert_allclose(float(wrapped_box.high[-1]), np.pi, atol=1e-6)

    def test_reset(self):
        _, wrapped = make_pair()
        obs, info = wrapped.reset(seed=7)
        self.assertEqual(obs.shape, wrapped.observation_space.shape)
        self.assertGreater(float(obs[-2]), 0.0)   # 起点剩余弧长 > 0
        self.assertGreaterEqual(float(obs[-1]), -np.pi)
        self.assertLessEqual(float(obs[-1]), np.pi)
        self.assertIsInstance(info, dict)

    def test_step_returns_5_tuple(self):
        _, wrapped = make_pair()
        wrapped.reset(seed=0)
        out = wrapped.step(np.array([1.0, 0.0], dtype=np.float32))
        self.assertEqual(len(out), 5)
        obs, reward, term, trunc, info = out
        self.assertEqual(obs.shape, wrapped.observation_space.shape)
        self.assertIsInstance(reward, float)
        self.assertIsInstance(term, bool)
        self.assertIsInstance(trunc, bool)
        self.assertIsInstance(info, dict)

    def test_default_action_runs(self):
        _, wrapped = make_pair()
        wrapped.reset(seed=1)
        obs = None
        for _ in range(20):
            obs, _r, term, trunc, _i = wrapped.step(
                np.array([1.0, 0.0], dtype=np.float32)
            )
            if term or trunc:
                wrapped.reset(seed=1)
                break
        assert obs is not None
        self.assertEqual(obs.shape, wrapped.observation_space.shape)

    def test_reward_matches_base_for_same_low_action(self):
        base, wrapped = make_pair(seed=123)
        controller = LowLevelController()
        for _ in range(10):
            x, y, yaw = controller.get_state_from_env(base)
            low = controller.compute_action(x, y, yaw, {})
            obs_b, rew_b, term_b, trunc_b, info_b = base.step(low)
            obs_w, rew_w, term_w, trunc_w, info_w = wrapped.step(
                np.array([1.0, 0.0], dtype=np.float32)
            )
            self.assertAlmostEqual(float(rew_w), float(rew_b), places=6)
            self.assertEqual(term_w, term_b)
            self.assertEqual(trunc_w, trunc_b)
            if term_b or trunc_b:
                break
        base.close()
        wrapped.close()


if __name__ == "__main__":
    unittest.main()
