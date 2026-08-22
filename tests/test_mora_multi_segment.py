"""MoRA 多段 RL：67 维观测、mseg 奖励、scope/start_at_subgoal、BC 动作反推。"""

import math
import unittest

import numpy as np

from rl.go2w_env import Go2wEnv
from rl.high_level_env_wrapper import HighLevelEnvWrapper, compute_mora_obs
from scripts.gen_demo_multi_segment import (
    compute_mseg_teacher_bias,
    default_params,
    make_env,
    run_episode,
)


def make_wrapper(scope: int = 2, start_a: bool = False) -> HighLevelEnvWrapper:
    base = Go2wEnv(
        task="traverse_curve", domain_randomize=False,
        multi_segment=True, reward_version="mseg", scope=scope,
    )
    return HighLevelEnvWrapper(
        base,
        append_mora=True,
        use_goal_condition=False,
        goal_min=1.4,
        goal_max=1.4,
    )


class MoraObsTest(unittest.TestCase):
    def test_obs_67_and_values(self):
        w = make_wrapper()
        obs, info = w.reset(seed=0)
        self.assertEqual(obs.shape, (67,))
        self.assertEqual(w.observation_space.shape, (67,))
        mora = obs[-6:]
        self.assertEqual(mora[0], 0.0)          # segment 0
        self.assertAlmostEqual(float(mora[1]), 4.3, places=3)
        self.assertAlmostEqual(float(mora[2]) ** 2 + float(mora[3]) ** 2, 1.0, places=4)
        self.assertEqual(mora[4], 0.0)          # 起点不在停靠区
        self.assertEqual(mora[5], 0.0)          # 未过 A
        w.close()

    def test_start_at_subgoal_sets_passed(self):
        base = Go2wEnv(
            task="traverse_curve", domain_randomize=False,
            multi_segment=True, reward_version="mseg", scope=1,
        )
        w = HighLevelEnvWrapper(
            base, append_mora=True, use_goal_condition=False,
            goal_min=1.4, goal_max=1.4,
        )
        obs, info = w.reset(seed=0, options={"start_at_subgoal": "A"})
        self.assertEqual(info["segment"], 1)
        self.assertTrue(info["passed_subgoal"])
        self.assertEqual(float(obs[-1]), 1.0)
        self.assertAlmostEqual(float(obs[-5]), 6.5, places=3)  # 到 B 的距离
        w.close()

    def test_docking_continuous(self):
        base = Go2wEnv(
            task="traverse_curve", domain_randomize=False,
            multi_segment=True, reward_version="mseg",
        )
        base.reset(seed=0)
        base.data.qpos[0] = 3.9  # 到 A 距离 0.4m
        base.data.qpos[1] = 0.0
        import mujoco
        mujoco.mj_forward(base.model, base.data)
        mora = compute_mora_obs(base)
        self.assertAlmostEqual(float(mora[4]), 0.5, places=3)
        base.close()


class MsegRewardTest(unittest.TestCase):
    def test_bonuses_one_shot(self):
        env = Go2wEnv(
            task="traverse_curve", domain_randomize=False,
            multi_segment=True, reward_version="mseg",
        )
        env.reset(seed=0)
        env._subgoal_in_range = True
        env._docking_done = True
        env._passed_subgoal = True
        env.scope = 2
        r1 = env._get_reward(np.zeros(6), False, False)
        r2 = env._get_reward(np.zeros(6), False, False)
        self.assertGreaterEqual(r1, 80.0)  # 50+30 (+10 视顺序)
        self.assertLess(r2, r1 + 1.0)
        env.close()

    def test_docking_reward_is_100(self):
        env = Go2wEnv(
            task="traverse_curve", domain_randomize=False,
            multi_segment=True, reward_version="mseg",
        )
        env.reset(seed=0)
        env._docking_done = True
        r1 = env._get_reward(np.zeros(6), False, False)
        r2 = env._get_reward(np.zeros(6), False, False)
        self.assertGreaterEqual(r1, 99.0)
        self.assertGreater(r1 - r2, 90.0)  # 一次性：第二次不再发 +100
        env.close()

    def test_overshoot_penalty(self):
        env = Go2wEnv(
            task="traverse_curve", domain_randomize=False,
            multi_segment=True, reward_version="mseg",
        )
        env.reset(seed=0)
        env._segment_idx = 1
        env._passed_subgoal = True
        env.data.qpos[0] = 11.0
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        env._get_reward(np.zeros(6), False, False)  # 消耗切换/到达一次性奖励
        r = env._get_reward(np.zeros(6), False, False)
        self.assertLessEqual(r, -0.19)  # 含 -1*0.2 过冲等
        env.close()


class ScopeAndStartTest(unittest.TestCase):
    def test_scope0_reaching_a_terminates(self):
        env = Go2wEnv(
            task="traverse_curve", domain_randomize=False,
            multi_segment=True, reward_version="mseg", scope=0,
        )
        env.reset(seed=0)
        import mujoco
        ax, ay = env._segment_goal(0)
        env.data.qpos[0] = ax
        env.data.qpos[1] = ay
        env.data.qvel[:] = 0.0
        mujoco.mj_forward(env.model, env.data)
        env.data.time = 10.35
        env._subgoal_hold_start = 10.0
        done = env._update_subgoals()
        self.assertTrue(done)
        self.assertTrue(env._goal_reached)
        env.close()

    def test_b_arrival_triggers_goal_and_docking(self):
        env = Go2wEnv(
            task="traverse_curve", domain_randomize=False,
            multi_segment=True, reward_version="mseg", scope=2,
        )
        env.reset(seed=0)
        env._segment_idx = 1
        bx, by = env._segment_goal(1)
        env.data.qpos[0] = bx
        env.data.qpos[1] = by
        env.data.qvel[:] = 0.0
        import mujoco
        mujoco.mj_forward(env.model, env.data)
        done = env._update_subgoals()
        self.assertTrue(done)
        self.assertTrue(env._goal_reached)
        self.assertTrue(env._docking_done)
        env.close()


class BcActionDerivationTest(unittest.TestCase):
    def test_turn_adjust_reconstruction(self):
        x, y, yaw = 2.0, 0.1, 0.2
        seg_idx = 0
        p = default_params()
        teacher = compute_mseg_teacher_bias(x, y, yaw, seg_idx, p)
        controller_bias = teacher + 0.2
        fwd = 0.12
        action = np.array(
            [fwd - controller_bias, fwd + controller_bias,
             fwd - controller_bias, fwd + controller_bias, 0.0, 0.0],
            dtype=np.float32,
        )
        derived_bias = float((action[1] - action[0]) / 2.0)
        turn_adjust = float(np.clip(derived_bias - teacher, -0.5, 0.5))
        self.assertAlmostEqual(turn_adjust, 0.2, places=5)


class Stage2DemoTest(unittest.TestCase):
    def test_stage2_episode_reaches_b(self):
        p = default_params()
        p["stage2"] = True
        env = make_env()
        rec = run_episode(env, p, seed=0)
        env.close()
        self.assertTrue(rec["goal"])


if __name__ == "__main__":
    unittest.main()
