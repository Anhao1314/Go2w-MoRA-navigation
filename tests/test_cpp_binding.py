"""C++ S0 binding and Go2wEnv integration tests."""

from __future__ import annotations

import math
import unittest

import numpy as np

from rl.cpp_s0_controller import CppS0Controller, _import_go2w_py
from rl.go2w_env import Go2wEnv
from rl.low_level_controller import MultiSegmentLowLevelController
from scripts.train_multi_segment_curriculum import make_stage_env


def _binding_available() -> bool:
    try:
        _import_go2w_py()
    except ImportError:
        return False
    return True


BINDING_AVAILABLE = _binding_available()


@unittest.skipUnless(BINDING_AVAILABLE, "go2w_py is not built")
class CppBindingTests(unittest.TestCase):
    def test_total_mass(self) -> None:
        go2w_py = _import_go2w_py()
        controller = go2w_py.S0Controller()
        self.assertAlmostEqual(float(controller.total_mass()), 19.126408, places=6)

    def test_teacher_parity_with_and_without_pre_brake(self) -> None:
        for pre_brake in (False, True):
            env = Go2wEnv(
                task="traverse_curve",
                domain_randomize=False,
                multi_segment=True,
                reward_version="mseg",
                scope=2,
            )
            env.reset(seed=1000)
            python_ctrl = MultiSegmentLowLevelController(
                env.segments, pre_brake=pre_brake
            )
            cpp_ctrl = CppS0Controller(env, pre_brake=pre_brake)
            cpp_ctrl.reset(env)
            max_diff = 0.0
            try:
                for _ in range(40):
                    x, y, yaw = python_ctrl.get_state_from_env(env)
                    cpp_ctrl.get_state_from_env(env)
                    gx, gy = env._segment_goal()
                    dist = math.hypot(x - gx, y - gy)
                    vx, _ = env._body_vel()
                    cmd = {
                        "speed_scale": 1.0,
                        "turn_adjust": 0.0,
                        "segment_idx": env._current_segment(),
                        "dist_to_subgoal": dist,
                        "speed": abs(vx),
                    }
                    a_py = python_ctrl.compute_action(x, y, yaw, cmd)
                    a_cpp = cpp_ctrl.compute_action(x, y, yaw, cmd)
                    max_diff = max(max_diff, float(np.max(np.abs(a_py - a_cpp))))
                    # Advance the raw env with the low-level 6D action.
                    env.step(a_py)
            finally:
                env.close()
            self.assertLess(max_diff, 1e-6, f"pre_brake={pre_brake}")

    def test_multi_segment_episode_matches_scripted(self) -> None:
        results = {}
        for s0 in ("scripted", "cpp"):
            env = make_stage_env(2, False, s0=s0, pre_brake=True)
            env.reset(seed=1000)
            action = np.array([1.0, 0.0], dtype=np.float32)
            steps = 0
            while True:
                obs, _reward, term, trunc, info = env.step(action)
                steps += 1
                if term or trunc or steps >= 4000:
                    break
            self.assertFalse(np.isnan(obs).any())
            results[s0] = (
                steps,
                float(env.base_env.data.time),
                float(info.get("x", 0.0)),
                bool(info.get("goal")),
            )
            env.close()
        self.assertEqual(results["scripted"], results["cpp"])
        self.assertTrue(results["cpp"][3])


if __name__ == "__main__":
    unittest.main()
