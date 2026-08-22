"""RL 环境单元测试：形状、终止、NaN 防护、确定性、域随机化、阶段脚本。"""

import math
import unittest

import mujoco
import numpy as np

from rl.go2w_env import (
    LEG_FOUR,
    LEG_TWO,
    HOLD_MIN,
    PITCH_REF,
    SCENARIOS,
    TASK_SCENARIO,
    TRAVERSE_TASKS,
    TRAVERSE_HEADING_PENALTY,
    Go2wEnv,
    curve_center_y,
    heading_error,
    init_two_wheel_pose,
    path_tangent_angle,
    pitch_of,
    traverse_backward_penalty,
    traverse_speed_penalty,
    traverse_shaping,
    traverse_stalled,
    wrap_angle,
)
from rl.terrain import (
    TERRAIN_NCOL,
    TERRAIN_NROW,
    make_terrain,
    write_heightfield,
)


class Go2wRlEnvTest(unittest.TestCase):
    def test_balance_reset_step_shapes(self):
        env = Go2wEnv(task="balance", domain_randomize=False)
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, (44,))
        self.assertTrue(np.isfinite(obs).all())
        self.assertEqual(info["stage"], 0)
        self.assertAlmostEqual(info["pitch_ref"], 0.0)
        obs, reward, terminated, truncated, info = env.step(np.zeros(2))
        self.assertEqual(obs.shape, (44,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertAlmostEqual(reward, 0.1, delta=0.05)

    def test_balance_v2_impacts(self):
        env = Go2wEnv(task="balance", domain_randomize=False)
        env.reset(seed=3)
        self.assertGreaterEqual(len(env._impacts), 2)
        self.assertLessEqual(len(env._impacts), 4)
        imp = env._impacts[0]
        while env.data.time < imp["t0"]:
            env.step(np.zeros(2))
        self.assertAlmostEqual(
            env.data.xfrc_applied[env.base_body_id, 0],
            imp["dir"] * imp["force"],
            delta=1e-6,
        )
        # 撞击窗口结束后外力应归零
        while env.data.time < imp["t0"] + imp["dur"] + 0.02:
            env.step(np.zeros(2))
            if env.data.time >= imp["t0"] + imp["dur"]:
                self.assertAlmostEqual(
                    env.data.xfrc_applied[env.base_body_id, 0], 0.0, delta=1e-6
                )
        env.close()

    def test_balance_v2_stage_onehot(self):
        env = Go2wEnv(task="balance", domain_randomize=False)
        obs, _ = env.reset(seed=0)
        self.assertEqual(obs[37], 1.0)  # stage0 one-hot
        self.assertEqual(obs[39], 0.0)  # 不是双轮保持阶段
        env.close()

    def test_full_chain_reset_stage(self):
        env = Go2wEnv(task="full_chain", domain_randomize=False)
        obs, info = env.reset(seed=0)
        self.assertEqual(info["stage"], 0)
        self.assertEqual(info["phi"], 0.0)

    def test_termination_on_fall(self):
        env = Go2wEnv(task="balance", domain_randomize=False)
        env.reset(seed=0)
        theta = 0.8  # 远大于 0.4 rad 偏差
        env.data.qpos[3:7] = (np.cos(theta / 2), 0.0, np.sin(theta / 2), 0.0)
        _, _, terminated, _, _ = env.step(np.zeros(2))
        self.assertTrue(terminated)

    def test_nan_guard(self):
        env = Go2wEnv(task="balance", domain_randomize=False)
        env.reset(seed=0)
        env.data.qpos[8] = np.nan
        _, _, terminated, _, info = env.step(np.zeros(2))
        self.assertTrue(terminated)
        self.assertTrue(info["nan"])

    def test_seed_reproducibility(self):
        env1 = Go2wEnv(task="balance", domain_randomize=True)
        env2 = Go2wEnv(task="balance", domain_randomize=True)
        obs1, _ = env1.reset(seed=123)
        obs2, _ = env2.reset(seed=123)
        np.testing.assert_allclose(obs1, obs2, rtol=0, atol=1e-6)

    def test_domain_randomize_toggle(self):
        env = Go2wEnv(task="balance", domain_randomize=True)
        base_mass = env._base_body_mass.copy()
        env.reset(seed=7)
        self.assertFalse(np.allclose(env.model.body_mass, base_mass))
        env.domain_randomize = False
        env.reset(seed=7)
        np.testing.assert_allclose(env.model.body_mass, base_mass)

    def test_full_chain_phase_script(self):
        env = Go2wEnv(task="full_chain", domain_randomize=False)
        env.reset(seed=0)
        init_two_wheel_pose(env.model, env.data)
        mujoco.mj_forward(env.model, env.data)
        env.data.time = 7.5
        target, stage, phi = env._phase_info(env.data.time)
        self.assertEqual(stage, 2)
        self.assertTrue(env._lower_started)
        _, stage, phi = env._phase_info(8.0)
        self.assertEqual(stage, 3)
        target, stage, phi = env._phase_info(10.6)
        self.assertEqual(stage, 4)
        self.assertAlmostEqual(phi, 2.0)
        np.testing.assert_allclose(target, LEG_FOUR)

    def test_pitch_of_helper(self):
        env = Go2wEnv(task="balance", domain_randomize=False)
        env.reset(seed=0)
        self.assertAlmostEqual(pitch_of(env.data), 0.0, delta=0.02)

    def test_traverse_tasks_shapes(self):
        for task in TRAVERSE_TASKS:
            env = Go2wEnv(task=task, domain_randomize=False)
            obs, info = env.reset(seed=0)
            self.assertEqual(obs.shape, (61,), task)
            self.assertEqual(env.action_space.shape, (6,), task)
            self.assertEqual(info["scenario"], TASK_SCENARIO[task])
            obs, reward, terminated, truncated, info = env.step(np.zeros(6))
            self.assertEqual(obs.shape, (61,), task)
            self.assertTrue(np.isfinite(obs).all())
            self.assertIsInstance(terminated, bool)
            env.close()

    def test_wrap_angle(self):
        self.assertAlmostEqual(wrap_angle(0.5), 0.5)
        self.assertAlmostEqual(wrap_angle(3.5), 3.5 - 2.0 * math.pi)
        self.assertAlmostEqual(wrap_angle(-3.5), -3.5 + 2.0 * math.pi)
        self.assertAlmostEqual(abs(wrap_angle(math.pi)), math.pi)
        self.assertAlmostEqual(wrap_angle(-math.pi), -math.pi)

    def test_path_tangent_angle(self):
        self.assertEqual(path_tangent_angle(2.0, is_curve=False), 0.0)
        x = 1.5
        eps = 1e-6
        num = (curve_center_y(x + eps) - curve_center_y(x - eps)) / (2.0 * eps)
        self.assertAlmostEqual(
            path_tangent_angle(x, True), math.atan2(num, 1.0), places=5
        )
        # 弯道上升段（x=1.2）切线为正
        self.assertGreater(path_tangent_angle(1.2, True), 0.0)

    def test_heading_error(self):
        x = 1.5
        tan = path_tangent_angle(x, True)
        self.assertAlmostEqual(heading_error(x, tan, True), 0.0, places=9)
        self.assertAlmostEqual(
            abs(heading_error(x, tan + math.pi, True)), math.pi, places=9
        )
        self.assertAlmostEqual(heading_error(2.0, 0.3, False), 0.3, places=9)

    def test_progress_rate_obs(self):
        env = Go2wEnv(task="traverse_slope", domain_randomize=False)
        env.reset(seed=0)
        env.data.qpos[0] = 1.2
        env._prev_progress = 1.0
        mujoco.mj_forward(env.model, env.data)
        obs = env._get_obs()
        self.assertAlmostEqual(float(obs[58]), 20.0, places=3)  # (1.2-1.0)/0.01
        env.data.qpos[0] = 1.0
        env._prev_progress = 1.2
        mujoco.mj_forward(env.model, env.data)
        obs = env._get_obs()
        self.assertAlmostEqual(float(obs[58]), -20.0, places=3)
        env.close()

    def test_curve_heading_penalty_relative(self):
        """弯道中 yaw=切线方向时不再受大惩罚，偏 1 rad 时惩罚为 1.5*1^2*0.01。"""
        env = Go2wEnv(task="traverse_curve", domain_randomize=False)
        env.reset(seed=0)
        x = 1.5
        tan = path_tangent_angle(x, True)
        env.data.qpos[0] = x
        env.data.qpos[1] = float(curve_center_y(x))
        env.data.qvel[:] = 0.0
        p = env._progress_metric(x)
        env._prev_progress = p
        env._prev_x = x

        env.data.qpos[3:7] = (math.cos(tan / 2.0), 0.0, 0.0, math.sin(tan / 2.0))
        mujoco.mj_forward(env.model, env.data)
        r_good = env._get_reward(np.zeros(6), False, False)

        env.data.qpos[3:7] = (
            math.cos((tan + 1.0) / 2.0), 0.0, 0.0, math.sin((tan + 1.0) / 2.0),
        )
        mujoco.mj_forward(env.model, env.data)
        r_bad = env._get_reward(np.zeros(6), False, False)

        expected = TRAVERSE_HEADING_PENALTY * (1.0 ** 2) * 0.01
        self.assertAlmostEqual(r_good - r_bad, expected, places=6)
        env.close()

    def test_heading_penalty_parameter(self):
        """heading_penalty 可配置：2.5 时同样偏航角惩罚更大。"""
        env = Go2wEnv(task="traverse_curve", domain_randomize=False,
                      heading_penalty=2.5)
        env.reset(seed=0)
        x = 1.5
        tan = path_tangent_angle(x, True)
        env.data.qpos[0] = x
        env.data.qpos[1] = float(curve_center_y(x))
        env.data.qvel[:] = 0.0
        p = env._progress_metric(x)
        env._prev_progress = p
        env._prev_x = x
        env.data.qpos[3:7] = (
            math.cos((tan + 1.0) / 2.0), 0.0, 0.0, math.sin((tan + 1.0) / 2.0),
        )
        mujoco.mj_forward(env.model, env.data)
        r_bad = env._get_reward(np.zeros(6), False, False)
        env.data.qpos[3:7] = (math.cos(tan / 2.0), 0.0, 0.0, math.sin(tan / 2.0))
        mujoco.mj_forward(env.model, env.data)
        r_good = env._get_reward(np.zeros(6), False, False)
        self.assertAlmostEqual(r_good - r_bad, 2.5 * 0.01, places=6)
        env.close()

    def test_curve_params_and_straight(self):
        # 振幅=0 → 直线：切线 0、中心线 y=0
        self.assertAlmostEqual(float(curve_center_y(1.0, amp=0.0)), 0.0)
        self.assertAlmostEqual(path_tangent_angle(1.0, True, amp=0.0), 0.0)
        env = Go2wEnv(task="traverse_curve", domain_randomize=False,
                      corridor_width=0.8, curve_amplitude=0.15,
                      reward_version="simple")
        env.reset(seed=0)
        # 走廊加宽到 0.8：dev=0.5 不再触发越界终止
        env.data.qpos[0] = 2.0
        env.data.qpos[1] = float(curve_center_y(2.0, amp=0.15)) + 0.5
        mujoco.mj_forward(env.model, env.data)
        self.assertFalse(env._check_termination())
        # 默认走廊 0.40 下 dev=0.5 会终止
        env2 = Go2wEnv(task="traverse_curve", domain_randomize=False)
        env2.reset(seed=0)
        env2.data.qpos[0] = 2.0
        env2.data.qpos[1] = float(curve_center_y(2.0)) + 0.5
        mujoco.mj_forward(env2.model, env2.data)
        self.assertTrue(env2._check_termination())
        env.close()
        env2.close()

    def test_simple_reward(self):
        """课程简洁奖励：无生存分；前进+0.1*dx；到达+100；摔倒-20。"""
        env = Go2wEnv(task="traverse_curve", domain_randomize=False,
                      reward_version="simple")
        env.reset(seed=0)
        p = env._progress_metric(float(env.data.body("base_link").xpos[0]))
        env._prev_progress = p
        r_still = env._get_reward(np.zeros(6), False, False)
        self.assertLess(abs(r_still), 0.05)  # 无生存分
        env.data.qpos[0] = 1.0
        mujoco.mj_forward(env.model, env.data)
        env._prev_progress = 0.9
        env._prev_x = 0.9
        r_move = env._get_reward(np.zeros(6), False, False)
        self.assertGreater(r_move, 0.005)  # 0.1*0.1 前进 - 航向小项
        # 到达 +100
        env._goal_reached = True
        r_goal = env._get_reward(np.zeros(6), False, False)
        self.assertGreaterEqual(r_goal - r_move, 99.0)  # 100 - 前进项 0.01
        env._goal_reached = False
        # 摔倒 -20（非早停）
        r_fall = env._get_reward(np.zeros(6), True, False)
        self.assertLessEqual(r_fall, -19.0)
        env.close()

    def test_v5_reward(self):
        """v5：无生存分、前进+1.0*dx、势能shaping、摔倒/早停-5、到达+100。"""
        env = Go2wEnv(task="traverse_curve", domain_randomize=False,
                      reward_version="v5")
        env.reset(seed=0)
        p = env._progress_metric(float(env.data.body("base_link").xpos[0]))
        env._prev_progress = p
        r_still = env._get_reward(np.zeros(6), False, False)
        self.assertLess(abs(r_still), 0.05)  # 无生存分，静止只余微小惩罚
        # 前进 0.1m（x: 0.9→1.0，弧长约 0.057）：+1.0*dx + shaping → 明显为正
        env.data.qpos[0] = 0.9
        mujoco.mj_forward(env.model, env.data)
        env._prev_progress = env._progress_metric(0.9)
        env.data.qpos[0] = 1.0
        mujoco.mj_forward(env.model, env.data)
        r_move = env._get_reward(np.zeros(6), False, False)
        self.assertGreater(r_move, 0.1)
        # 到达 +100
        env._prev_progress = env._progress_metric(1.0)
        env._goal_reached = True
        r_goal = env._get_reward(np.zeros(6), False, False)
        self.assertGreaterEqual(r_goal - r_move, 95.0)  # 100 减去 r_move 里的前进/shaping
        env._goal_reached = False
        # 摔倒 -5
        r_fall = env._get_reward(np.zeros(6), True, False)
        self.assertLessEqual(r_fall, -4.0)
        self.assertGreater(r_fall, -6.0)
        # 早停 -5
        env._early_stopped = True
        r_early = env._get_reward(np.zeros(6), False, False)
        self.assertLessEqual(r_early, -4.0)
        self.assertGreater(r_early, -6.0)
        env.close()

    def test_v5_reward_parameters(self):
        """v5 摔倒/到达可配置：-10 / +200。"""
        env = Go2wEnv(task="traverse_curve", domain_randomize=False,
                      reward_version="v5", fall_penalty=-10.0, goal_bonus=200.0)
        env.reset(seed=0)
        env._goal_reached = True
        r_goal = env._get_reward(np.zeros(6), False, False)
        env._goal_reached = False
        r_fall = env._get_reward(np.zeros(6), True, False)
        self.assertGreaterEqual(r_goal, 199.0)
        self.assertLessEqual(r_fall, -9.0)
        self.assertGreater(r_fall, -11.0)
        env.close()

    def test_full_chain_simple_stages_and_reward(self):
        env = Go2wEnv(task="full_chain_simple", domain_randomize=False)
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, (44,))
        self.assertEqual(env.action_space.shape, (2,))
        self.assertEqual(info["stage"], 0)
        self.assertEqual(env.max_steps, 1200)  # 12s
        target, stage, phi = env._phase_info(7.5)
        self.assertEqual(stage, 2)
        np.testing.assert_allclose(target, LEG_TWO)
        # 双轮保持 5s -> 成功奖励 +30 并置成功标志
        init_two_wheel_pose(env.model, env.data)
        mujoco.mj_forward(env.model, env.data)
        env._stage = 2
        env._phi = 1.0
        env._prev_phi = 1.0
        env._stable_seconds = HOLD_MIN - 0.02
        env._stable_bonus_given = True  # 隔离测试：只验证成功 +30
        r1 = env._get_reward(np.zeros(2), False, False)
        self.assertFalse(env._simple_success)
        r2 = env._get_reward(np.zeros(2), False, False)
        self.assertTrue(env._simple_success)
        self.assertAlmostEqual(r2 - r1, 30.0, places=6)
        env.close()

    def test_traverse_leg_cmd_mapping(self):
        env = Go2wEnv(task="traverse_slope", domain_randomize=False)
        env.reset(seed=0)
        base = LEG_FOUR.copy()
        target_up = env._leg_target_from_cmds(1.0, 1.0)
        target_split = env._leg_target_from_cmds(1.0, -1.0)
        self.assertNotAlmostEqual(target_up[1], base[1])
        self.assertLess(target_split[1], base[1])
        self.assertGreater(target_split[7], base[7])
        np.testing.assert_allclose(target_up[[0, 3, 6, 9]], base[[0, 3, 6, 9]])
        env.step(np.array([0.0] * 4 + [0.8, -0.5]))
        np.testing.assert_allclose(env._leg_cmd, [0.8, -0.5])
        env.close()

    def test_traverse_speed_penalty(self):
        self.assertEqual(traverse_speed_penalty(0.5), 0.0)
        self.assertEqual(traverse_speed_penalty(0.8), 0.0)
        self.assertLess(traverse_speed_penalty(1.0), 0.0)
        self.assertLess(traverse_speed_penalty(1.5), traverse_speed_penalty(1.0))
        self.assertLess(traverse_speed_penalty(-1.5), traverse_speed_penalty(-1.0))

    def test_traverse_slope_geometry(self):
        env = Go2wEnv(task="traverse_slope", domain_randomize=False)
        env.reset(seed=0)
        # 默认 hfield：高度场激活、盒状斜坡隐藏
        self.assertGreaterEqual(env._scn_terrain_gid, 0)
        self.assertAlmostEqual(env.model.geom_pos[env._scn_terrain_gid][0], 2.5)
        z = env.model.hfield_data[env._hfield_slice]
        self.assertGreater(float(np.max(z)), 0.2)
        self.assertLessEqual(float(np.max(z)), 1.0)
        ramp = env._scn_ramp_id
        self.assertGreater(ramp, -1)
        self.assertAlmostEqual(
            env.model.body_pos[env._scn_ramp_body_id][1], 1000.0
        )
        # 弯道墙应被隐藏
        l0 = env._scn_curve_l_bodies[0]
        self.assertAlmostEqual(env.model.body_pos[l0][1], 1000.0)
        env.close()

    def test_traverse_slope_boxes_geometry(self):
        """terrain='boxes' 时保持旧盒状斜坡几何。"""
        env = Go2wEnv(task="traverse_slope", domain_randomize=False,
                      terrain="boxes")
        env.reset(seed=0)
        ramp = env._scn_ramp_id
        self.assertLess(env.model.body_pos[env._scn_ramp_body_id][1], 0.5)
        # 斜坡 body 应带俯仰旋转（运行时 geom_quat 不生效，靠 body_quat）
        self.assertNotAlmostEqual(
            abs(env.model.body_quat[env._scn_ramp_body_id][2]), 0.0
        )
        # 高度场应被隐藏
        self.assertAlmostEqual(
            env.model.geom_pos[env._scn_terrain_gid][1], 1000.0
        )
        env.close()

    def test_traverse_hfield_reproducible_and_varied(self):
        env1 = Go2wEnv(task="traverse_slope", domain_randomize=False)
        env2 = Go2wEnv(task="traverse_slope", domain_randomize=False)
        env1.reset(seed=42)
        env2.reset(seed=42)
        z1 = env1.model.hfield_data[env1._hfield_slice].copy()
        z2 = env2.model.hfield_data[env2._hfield_slice].copy()
        np.testing.assert_allclose(z1, z2, rtol=0, atol=1e-12)
        env1.reset(seed=43)
        z3 = env1.model.hfield_data[env1._hfield_slice].copy()
        self.assertFalse(np.allclose(z1, z3))
        env1.close()
        env2.close()

    def test_traverse_curve_geometry_and_termination(self):
        env = Go2wEnv(task="traverse_curve", domain_randomize=False)
        env.reset(seed=0)
        self.assertGreater(len(env._scn_curve_l_ids), 0)
        l0 = env._scn_curve_l_bodies[0]
        self.assertAlmostEqual(env.model.body_pos[l0][1], -0.40, delta=0.2)
        ramp = env._scn_ramp_id
        self.assertAlmostEqual(
            env.model.body_pos[env._scn_ramp_body_id][1], 1000.0
        )
        # 弯道场景不激活高度场地形
        self.assertAlmostEqual(
            env.model.geom_pos[env._scn_terrain_gid][1], 1000.0
        )
        # 偏离中心线超过半宽 -> 终止
        env.data.qpos[0] = 2.5
        env.data.qpos[1] = 1.2
        mujoco.mj_forward(env.model, env.data)
        _, _, terminated, _, _ = env.step(np.zeros(6))
        self.assertTrue(terminated)
        env.close()

    def test_curve_walls_rotated_and_straight_blocked(self):
        """墙必须真的旋转（body_quat 生效），直线捷径 y≈0 在弯道峰处必须撞墙，
        而沿中心线仍可无碰撞通过。"""
        env = Go2wEnv(task="traverse_curve", domain_randomize=False)
        env.reset(seed=0)
        l2 = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "scn_curve_l_2")
        xmat = env.data.geom_xmat[l2].reshape(3, 3)
        # 旋转后的墙 xmat 不应是单位阵（绕 z 有转角）
        self.assertGreater(abs(float(xmat[0, 1])), 0.05)

        def hits(x: float, y: float) -> int:
            env.data.qpos[0] = x
            env.data.qpos[1] = y
            env.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            env.data.qvel[:] = 0.0
            mujoco.mj_forward(env.model, env.data)
            n = 0
            for c in env.data.contact:
                if c.dist < 0.02:
                    g0 = env.model.geom(c.geom[0]).name
                    g1 = env.model.geom(c.geom[1]).name
                    if "scn_curve" in (g0 or "") or "scn_curve" in (g1 or ""):
                        n += 1
            return n

        self.assertGreater(hits(1.8, 0.0), 0)  # 直线捷径被挡
        self.assertGreater(hits(3.0, 0.0), 0)
        self.assertEqual(hits(1.8, float(curve_center_y(1.8))), 0)  # 中心线可通行
        env.close()

    def test_make_terrain_properties(self):
        rng = np.random.default_rng(7)
        z = make_terrain(20.0, rng=rng)
        self.assertEqual(z.shape, (TERRAIN_NROW, TERRAIN_NCOL))
        self.assertTrue(np.isfinite(z).all())
        self.assertGreaterEqual(float(z.min()), 0.0)
        self.assertLessEqual(float(z.max()), 1.0)
        # x < 0.8 严格为 0（初始平地段）
        n_flat = int(round(0.8 / 5.0 * TERRAIN_NCOL))
        self.assertAlmostEqual(float(np.max(z[:, :n_flat])), 0.0)
        # 主坡段沿 x 单调不减（允许同高）
        col0 = z[:, int(1.2 / 5.0 * TERRAIN_NCOL)]
        col1 = z[:, int(2.5 / 5.0 * TERRAIN_NCOL)]
        col2 = z[:, int(4.0 / 5.0 * TERRAIN_NCOL)]
        self.assertGreater(float(np.mean(col1)), float(np.mean(col0)))
        self.assertGreater(float(np.mean(col2)), float(np.mean(col1)))
        # y 方向左右对称
        np.testing.assert_allclose(z, z[::-1], rtol=0, atol=1e-12)

    def test_make_terrain_determinism(self):
        z1 = make_terrain(20.0, rng=np.random.default_rng(123))
        z2 = make_terrain(20.0, rng=np.random.default_rng(123))
        z3 = make_terrain(20.0, rng=np.random.default_rng(124))
        np.testing.assert_allclose(z1, z2, rtol=0, atol=1e-12)
        self.assertFalse(np.allclose(z1, z3))

    def test_write_heightfield(self):
        env = Go2wEnv(task="traverse_slope", domain_randomize=False,
                      terrain="boxes")
        env.reset(seed=0)
        z = make_terrain(25.0, rng=np.random.default_rng(9))
        write_heightfield(env.model, env._scn_terrain_hid, z)
        np.testing.assert_allclose(
            env.model.hfield_data[env._hfield_slice], z.reshape(-1),
            rtol=0, atol=1e-6,
        )
        with self.assertRaises(ValueError):
            write_heightfield(env.model, env._scn_terrain_hid, z[:-1])
        env.close()

    def test_traverse_goal_termination(self):
        env = Go2wEnv(task="traverse_slope", domain_randomize=False)
        env.reset(seed=0)
        env.data.qpos[0] = 4.6
        mujoco.mj_forward(env.model, env.data)
        _, reward, terminated, _, info = env.step(np.zeros(6))
        self.assertTrue(terminated)
        self.assertTrue(info["goal"])
        self.assertGreater(reward, 40.0)
        env.close()

    def test_traverse_shaping_direction(self):
        forward = traverse_shaping(1.0, 0.9)
        backward = traverse_shaping(0.9, 1.0)
        self.assertGreater(forward, 0.0)
        self.assertLess(backward, 0.0)
        self.assertLess(traverse_backward_penalty(-0.1), 0.0)
        self.assertAlmostEqual(traverse_backward_penalty(-0.1), -1.5)
        self.assertEqual(traverse_backward_penalty(0.1), 0.0)

    def test_traverse_shaping_total(self):
        """Φ_max=90 时完整走完 4.5m，shaping 累计 ≈ 0.99×90 = 89.1。"""
        total = 0.0
        x = 0.0
        while x < 4.5:
            nxt = min(x + 0.01, 4.5)
            total += traverse_shaping(nxt, x)
            x = nxt
        self.assertAlmostEqual(total, 0.99 * 90.0, places=2)

    def test_traverse_survival_by_progress(self):
        """生存分按进度：静止无生存分，前进 0.2m/步给 0.1×0.2×100。"""
        env = Go2wEnv(task="traverse_slope", domain_randomize=False)
        env.reset(seed=0)
        # 静止：dx=0，奖励里不应有 +0.1 生存分
        p = env._progress_metric(float(env.data.body("base_link").xpos[0]))
        env._prev_progress = p
        r_still = env._get_reward(np.zeros(6), False, False)
        self.assertLess(r_still, 0.05)
        # 前进：dx=0.2 → +2.0 生存分 + shaping 3.76，总计 > 5
        env.data.qpos[0] = 1.2
        mujoco.mj_forward(env.model, env.data)
        env._prev_progress = 1.0
        r_move = env._get_reward(np.zeros(6), False, False)
        self.assertGreater(r_move, 5.0)
        env.close()

    def test_traverse_early_stop_penalty_20(self):
        env = Go2wEnv(task="traverse_slope", domain_randomize=False)
        env.reset(seed=0)
        p = env._progress_metric(float(env.data.body("base_link").xpos[0]))
        env._prev_progress = p
        env._early_stopped = True
        r = env._get_reward(np.zeros(6), False, False)
        self.assertAlmostEqual(r, -20.0, delta=0.5)
        env.close()

    def test_traverse_stalled_helper(self):
        self.assertTrue(traverse_stalled(max_x=0.05, ref_x=0.0, now_t=3.1, ref_t=0.0))
        self.assertFalse(traverse_stalled(max_x=0.05, ref_x=0.0, now_t=2.9, ref_t=0.0))
        self.assertFalse(traverse_stalled(max_x=0.2, ref_x=0.1, now_t=3.1, ref_t=0.0))

    def test_traverse_early_stop(self):
        env = Go2wEnv(task="traverse_slope", domain_randomize=False)
        env.reset(seed=0)
        env.data.qpos[0] = 0.05
        mujoco.mj_forward(env.model, env.data)
        env._max_x = 0.05
        env._progress_ref_x = 0.0
        env._progress_ref_t = 0.0
        env.data.time = 3.1
        env._update_progress_tracking()
        self.assertTrue(env._early_stopped)
        _, reward, terminated, _, info = env.step(np.zeros(6))
        self.assertTrue(terminated)
        self.assertTrue(info["early_stopped"])
        self.assertLess(reward, -5.0)

    def test_traverse_backward_runaway(self):
        env = Go2wEnv(task="traverse_slope", domain_randomize=False)
        env.reset(seed=0)
        env.data.qpos[0] = -1.1
        mujoco.mj_forward(env.model, env.data)
        _, _, terminated, _, _ = env.step(np.zeros(6))
        self.assertTrue(terminated)


if __name__ == "__main__":
    unittest.main()
