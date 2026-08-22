"""Junction 岔路口任务（MoRA System 2 + System 1）单元测试。"""

import math
import unittest

import mujoco
import numpy as np

from rl.go2w_env import Go2wEnv
from rl.high_level_env_wrapper import (
    HighLevelEnvWrapper,
    compute_junction_obs,
    junction_teacher_bias,
)
from scripts.train_junction_curriculum import (
    JunctionWrapper,
    STAGES,
    stage_done,
)


def make_env(scope: int = 2, target: str = "A", **kw) -> Go2wEnv:
    return Go2wEnv(
        task="traverse_curve",
        domain_randomize=False,
        junction=True,
        reward_version="junction",
        scope=scope,
        target_goal=target,
        **kw,
    )


class JunctionEnvTest(unittest.TestCase):
    def test_geometry_and_goals(self):
        env = make_env()
        env.reset(seed=0)
        cfg = env.junction_config
        self.assertAlmostEqual(cfg["goal_x"], 5.5)
        self.assertAlmostEqual(cfg["goal_y"], 0.30)
        ax, ay = env._junction_goal()  # target A → 左分支
        self.assertAlmostEqual(ax, 5.5)
        self.assertAlmostEqual(ay, 0.30)
        env.target_goal = "B"
        bx, by = env._junction_goal()
        self.assertAlmostEqual(bx, 5.5)
        self.assertAlmostEqual(by, -0.30)
        xj = cfg["decision_x_end"]
        self.assertAlmostEqual(
            env._junction_center_y(xj, "left"),
            env._junction_center_y(xj, None),
            places=6,
        )
        self.assertAlmostEqual(env._junction_center_y(0.0), 0.0, places=6)
        env.close()

    def test_scope0_goal_is_junction(self):
        env = make_env(scope=0)
        env.reset(seed=1)
        env.data.qpos[0] = 2.3
        env.data.qpos[1] = float(env._junction_center_y(2.3))
        mujoco.mj_forward(env.model, env.data)
        self.assertTrue(env._traverse_goal_reached())
        self.assertTrue(env._junction_reached())
        env.close()

    def test_junction_goal_truncates(self):
        env = make_env(scope=2, target="A")
        env.reset(seed=0)
        env.data.qpos[0] = 5.5
        env.data.qpos[1] = 0.30
        env.branch_selected = "left"
        mujoco.mj_forward(env.model, env.data)
        obs, _r, term, trunc, info = env.step(np.zeros(6, dtype=np.float32))
        self.assertTrue(info.get("goal"))
        self.assertTrue(trunc or term)
        env.close()

    def test_corridor_termination(self):
        env = make_env(scope=2, target="A")
        env.reset(seed=0)
        # 分支走廊半宽 0.40：dev=0.5 触发终止
        env.data.qpos[0] = 3.0
        env.data.qpos[1] = float(env._junction_center_y(3.0, "left")) + 0.5
        env.branch_selected = "left"
        mujoco.mj_forward(env.model, env.data)
        self.assertTrue(env._check_termination())
        # 公共段走廊半宽 0.50：dev=0.6 触发终止
        env.data.qpos[0] = 1.0
        env.data.qpos[1] = float(env._junction_center_y(1.0)) + 0.6
        env.branch_selected = None
        mujoco.mj_forward(env.model, env.data)
        self.assertTrue(env._check_termination())
        # 公共段内 dev=0.2 不终止
        env.data.qpos[1] = float(env._junction_center_y(1.0)) + 0.2
        mujoco.mj_forward(env.model, env.data)
        self.assertFalse(env._check_termination())
        env.close()

    def test_junction_reward_one_time(self):
        env = make_env(scope=2, target="A")
        env.reset(seed=0)
        env.data.qpos[0] = 2.25
        env.data.qpos[1] = float(env._junction_center_y(2.25))
        mujoco.mj_forward(env.model, env.data)
        env._get_reward(np.zeros(6, dtype=np.float32), False, False)
        self.assertTrue(env._junction_reached_bonus_given)
        # 第二次不再发放
        env._get_reward(np.zeros(6, dtype=np.float32), False, False)
        self.assertTrue(env._junction_reached_bonus_given)
        self.assertEqual(env._junction_decision_bonus_given, False)
        env.close()

    def test_decision_reward_correct_and_wrong(self):
        env = make_env(scope=2, target="A")
        env.reset(seed=0)
        env.branch_selected = "left"  # A → left 正确
        env._get_reward(np.zeros(6, dtype=np.float32), False, False)
        self.assertTrue(env._junction_decision_bonus_given)
        self.assertFalse(env._junction_wrong_branch_given)
        env.close()

        env2 = make_env(scope=2, target="A")
        env2.reset(seed=1)
        env2.branch_selected = "right"  # 错误分支
        env2._get_reward(np.zeros(6, dtype=np.float32), False, False)
        self.assertTrue(env2._junction_decision_bonus_given)
        self.assertTrue(env2._junction_wrong_branch_given)
        env2.close()

    def test_goal_bonus_one_time(self):
        env = make_env(scope=2, target="A")
        env.reset(seed=0)
        env.data.qpos[0] = 5.5
        env.data.qpos[1] = 0.30
        env.branch_selected = "left"
        env._goal_reached = True
        mujoco.mj_forward(env.model, env.data)
        env._get_reward(np.zeros(6, dtype=np.float32), False, True)
        self.assertTrue(env._junction_goal_bonus_given)
        env._get_reward(np.zeros(6, dtype=np.float32), False, True)
        self.assertTrue(env._junction_goal_bonus_given)
        env.close()

    def test_start_at_junction_option(self):
        env = make_env(scope=1)
        env.reset(seed=3, options={"start_at_junction": True})
        self.assertAlmostEqual(
            float(env.data.body("base_link").xpos[0]),
            env.junction_config["decision_x_end"],
            places=2,
        )
        self.assertIn(env.get_branch_selected(), ("left", "right"))
        env.close()

    def test_backward_compat_default(self):
        env = Go2wEnv(task="traverse_curve", domain_randomize=False)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs.shape, (61,))
        self.assertFalse(env.junction)
        env.close()


class JunctionWrapperTest(unittest.TestCase):
    def test_obs_68_dim(self):
        base = make_env(scope=2, target="A")
        wrapped = HighLevelEnvWrapper(
            base, append_junction=True, use_goal_condition=False
        )
        obs, _ = wrapped.reset(seed=0)
        self.assertEqual(obs.shape, (68,))
        self.assertEqual(wrapped.observation_space.shape, (68,))
        jun = compute_junction_obs(base)
        np.testing.assert_allclose(obs[-7:], jun)
        self.assertEqual(float(obs[-7]), 1.0)  # target A onehot
        self.assertEqual(float(obs[-6]), 0.0)
        wrapped.close()

    def test_compute_junction_obs_values(self):
        base = make_env(scope=2, target="A")
        base.reset(seed=0)
        base.data.qpos[0] = 3.0
        base.data.qpos[1] = 0.15
        base.branch_selected = "left"
        mujoco.mj_forward(base.model, base.data)
        jun = compute_junction_obs(base)
        self.assertEqual(jun.shape, (7,))
        np.testing.assert_allclose(jun[:2], [1.0, 0.0])
        np.testing.assert_allclose(jun[2:4], [1.0, 0.0])
        self.assertAlmostEqual(float(jun[4]), 0.0, places=4)  # 已过岔路口
        self.assertAlmostEqual(
            float(jun[5]), math.hypot(2.5, 0.15), places=3  # 到终点距离
        )
        self.assertEqual(float(jun[6]), 1.0)  # System 2 已决策
        base.close()

    def test_system2_decision_in_step(self):
        base = make_env(scope=2, target="A")
        wrapped = HighLevelEnvWrapper(
            base, append_junction=True, use_goal_condition=False
        )
        wrapped.reset(seed=0)
        base.data.qpos[0] = 2.0
        base.data.qpos[1] = float(base._junction_center_y(2.0))
        mujoco.mj_forward(base.model, base.data)
        wrapped.step(np.array([1.0, 0.0], dtype=np.float32))
        self.assertEqual(base.get_branch_selected(), "left")
        wrapped.close()

        base2 = make_env(scope=2, target="B")
        wrapped2 = HighLevelEnvWrapper(
            base2, append_junction=True, use_goal_condition=False
        )
        wrapped2.reset(seed=0)
        base2.data.qpos[0] = 2.0
        base2.data.qpos[1] = float(base2._junction_center_y(2.0))
        mujoco.mj_forward(base2.model, base2.data)
        wrapped2.step(np.array([1.0, 0.0], dtype=np.float32))
        self.assertEqual(base2.get_branch_selected(), "right")
        wrapped2.close()

    def test_junction_wrapper_scope2_fixed_target(self):
        base = make_env(scope=2)
        wrapped = JunctionWrapper(base, scope=2, start_junction=False)
        wrapped.reset(seed=0, options={"target_goal": "B"})
        self.assertEqual(base.get_target_goal(), "B")
        wrapped.reset(seed=1, options={"target_goal": "A"})
        self.assertEqual(base.get_target_goal(), "A")
        wrapped.close()

    def test_junction_wrapper_scope1_sync(self):
        base = make_env(scope=1)
        wrapped = JunctionWrapper(base, scope=1, start_junction=True)
        for seed in range(10):
            wrapped.reset(seed=seed)
            branch = base.get_branch_selected()
            self.assertIn(branch, ("left", "right"))
            self.assertEqual(
                base.get_target_goal(), "A" if branch == "left" else "B"
            )
            self.assertAlmostEqual(
                float(base.data.body("base_link").xpos[0]),
                base.junction_config["decision_x_end"],
                places=2,
            )
            self.assertTrue(base._junction_reached_bonus_given)
        wrapped.close()

    def test_teacher_bias_bounded(self):
        base = make_env(scope=2, target="A")
        base.reset(seed=0)
        base.branch_selected = "left"
        bias = junction_teacher_bias(2.5, 0.1, 0.2, base)
        self.assertTrue(-1.0 <= bias <= 1.0)
        base.close()


class JunctionCurriculumTest(unittest.TestCase):
    def test_stages(self):
        self.assertEqual(
            [s["name"] for s in STAGES], ["stage1", "stage2", "stage3"]
        )
        self.assertEqual([s["scope"] for s in STAGES], [0, 1, 2])
        self.assertEqual(sum(s["steps"] for s in STAGES), 1_800_000)

    def test_stage_done(self):
        self.assertFalse(stage_done([], 0.8))
        self.assertFalse(
            stage_done([{"success_rate": 1.0}] * 4, 0.8)
        )
        self.assertTrue(
            stage_done([{"success_rate": 1.0}] * 5, 0.8)
        )
        self.assertFalse(
            stage_done(
                [{"success_rate": 1.0}] * 4 + [{"success_rate": 0.7}],
                0.8,
            )
        )


if __name__ == "__main__":
    unittest.main()
