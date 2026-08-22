"""项目一冒烟测试：Go2w 模型加载、数值稳定、渲染与双轮自平衡。"""

import math
import pathlib
import unittest

import mujoco
import numpy as np

from mujoco_demos import common
from scripts.demo_go2w import (
    MODEL_PATH,
    PITCH_REF,
    Go2wBalanceController,
    init_pose,
    pitch_of,
)


class Go2wSmokeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = common.load_model(MODEL_PATH)
        cls.controller = Go2wBalanceController(cls.model)

    def test_model_load(self):
        self.assertEqual(self.model.opt.timestep, 0.002)
        self.assertEqual(self.model.nu, 16)
        for name in ("FR_wheel", "FL_wheel", "RR_wheel", "RL_wheel"):
            self.assertGreaterEqual(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name),
                0,
            )

    def test_overview_camera_in_scenes(self):
        root = pathlib.Path(__file__).resolve().parents[1]
        for rel in (
            "models/go2w/go2w_factory_scene.xml",
            "models/go2w/go2w_scenario_scene.xml",
        ):
            model = mujoco.MjModel.from_xml_path(str(root / rel))
            self.assertGreaterEqual(
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "overview"),
                0,
                f"{rel} 缺少 overview 相机",
            )

    def test_simulation_no_nan(self):
        data = mujoco.MjData(self.model)
        init_pose(self.model, data)
        mujoco.mj_forward(self.model, data)
        for _ in range(500):
            self.controller.step(data, data.time)
            mujoco.mj_step(self.model, data)
        self.assertFalse(np.isnan(data.qpos).any())
        self.assertFalse(np.isnan(data.qvel).any())

    def test_render_frame(self):
        frames, _ = common.record(
            self.model,
            0.2,
            30,
            self.controller.step,
            init_fn=lambda data: init_pose(self.model, data),
        )
        self.assertEqual(len(frames), 6)
        self.assertEqual(frames[0].shape, (480, 640, 3))
        self.assertGreater(int((frames[0] != 0).sum()), 0)

    def test_two_wheel_balance_hold(self):
        """完整流程：四轮站稳 -> 收腿 -> 双轮平衡保持 8 秒。"""
        data = mujoco.MjData(self.model)
        init_pose(self.model, data)
        mujoco.mj_forward(self.model, data)
        max_dev = 0.0
        min_front_clear = 9.9
        balance_started = False
        for _ in range(int(10.5 / self.model.opt.timestep)):
            t = data.time
            self.controller.step(data, t)
            mujoco.mj_step(self.model, data)
            if t > 3.0:
                balance_started = True
                dev = abs(pitch_of(data) - PITCH_REF)
                max_dev = max(max_dev, dev)
                clear = min(
                    data.body("FL_wheel_link").xpos[2],
                    data.body("FR_wheel_link").xpos[2],
                ) - 0.086
                min_front_clear = min(min_front_clear, clear)
        self.assertTrue(balance_started)
        self.assertLess(max_dev, 0.1, f"双轮平衡偏差过大: {max_dev:.3f} rad")
        self.assertGreater(min_front_clear, 0.10, f"前轮离地不足: {min_front_clear:.3f} m")
        self.assertFalse(np.isnan(data.qpos).any())


if __name__ == "__main__":
    unittest.main()
