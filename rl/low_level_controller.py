"""System 0 低层运动控制器：脚本控制器 + 高层指令（速度倍率/转向调整）。

高层动作空间为 2 维 [speed_scale, turn_adjust]：
  - speed_scale ∈ [0.5, 1.5]：缩放脚本控制器的前进力矩；
  - turn_adjust ∈ [-0.5, 0.5]：在脚本控制器基础 bias 上叠加转向调整。

与 `scripts/gen_demo_trajectory.compute_teacher_action` 完全同源：
高层指令 [1.0, 0.0] 时输出应与教师动作逐元素一致。
"""

from __future__ import annotations

import numpy as np

from scripts.gen_demo_trajectory import compute_teacher_bias, default_params
from scripts.gen_demo_multi_segment import compute_mseg_teacher_bias


class LowLevelController:
    def __init__(
        self,
        lookahead: float = 0.7,
        k_ang: float = 2.5,
        k_dev: float = 0.5,
        forward: float = 0.12,
        bias_clip: float = 1.0,
        leg_cmd: tuple[float, float] = (0.0, 0.0),
    ) -> None:
        self.params = {
            "lookahead": float(lookahead),
            "k_ang": float(k_ang),
            "k_dev": float(k_dev),
            "forward": float(forward),
            "bias_clip": float(bias_clip),
            "rear_only": False,
        }
        self.leg_cmd = (float(leg_cmd[0]), float(leg_cmd[1]))

    def compute_action(
        self,
        x: float,
        y: float,
        yaw: float,
        high_level_cmd: dict | None = None,
    ) -> np.ndarray:
        cmd = high_level_cmd or {}
        speed_scale = float(np.clip(float(cmd.get("speed_scale", 1.0)), 0.5, 1.5))
        turn_adjust = float(np.clip(float(cmd.get("turn_adjust", 0.0)), -0.5, 0.5))

        teacher_bias = compute_teacher_bias(x, y, yaw, self.params)
        actual_bias = float(
            np.clip(
                teacher_bias + turn_adjust,
                -self.params["bias_clip"],
                self.params["bias_clip"],
            )
        )
        actual_forward = float(self.params["forward"]) * speed_scale
        action = np.array(
            [
                actual_forward - actual_bias,  # FR
                actual_forward + actual_bias,  # FL
                actual_forward - actual_bias,  # RR
                actual_forward + actual_bias,  # RL
                self.leg_cmd[0],
                self.leg_cmd[1],
            ],
            dtype=np.float32,
        )
        return np.clip(action, -1.0, 1.0)

    def get_state_from_env(self, env) -> tuple[float, float, float]:
        x = float(env.data.body("base_link").xpos[0])
        y = float(env.data.body("base_link").xpos[1])
        yaw = float(env._yaw_of())
        return x, y, yaw


class MultiSegmentLowLevelController(LowLevelController):
    """多段路径 System 0：教师 bias 按当前段计算。"""

    def __init__(self, segments: list, **kwargs):
        super().__init__(**kwargs)
        self._segments = segments

    def compute_action(
        self,
        x: float,
        y: float,
        yaw: float,
        high_level_cmd: dict | None = None,
    ) -> np.ndarray:
        cmd = high_level_cmd or {}
        seg_idx = int(cmd.get("segment_idx", 0))
        speed_scale = float(np.clip(float(cmd.get("speed_scale", 1.0)), 0.5, 1.5))
        turn_adjust = float(np.clip(float(cmd.get("turn_adjust", 0.0)), -0.5, 0.5))
        dist = cmd.get("dist_to_subgoal")
        speed = cmd.get("speed")
        seg = self._segments[seg_idx]
        if dist is not None and speed is not None:
            if dist <= seg["goal_tolerance"]:
                if abs(speed) < seg["stop_speed_threshold"]:
                    return np.array(
                        [0.0, 0.0, 0.0, 0.0, self.leg_cmd[0], self.leg_cmd[1]],
                        dtype=np.float32,
                    )
                brake = -0.02 * abs(float(speed))
                return np.array(
                    [brake, brake, brake, brake, self.leg_cmd[0], self.leg_cmd[1]],
                    dtype=np.float32,
                )
        teacher_bias = compute_mseg_teacher_bias(
            x, y, yaw, seg_idx, self.params, segments=self._segments
        )
        actual_bias = float(
            np.clip(
                teacher_bias + turn_adjust,
                -self.params["bias_clip"],
                self.params["bias_clip"],
            )
        )
        actual_forward = float(self.params["forward"]) * speed_scale
        action = np.array(
            [
                actual_forward - actual_bias,
                actual_forward + actual_bias,
                actual_forward - actual_bias,
                actual_forward + actual_bias,
                self.leg_cmd[0],
                self.leg_cmd[1],
            ],
            dtype=np.float32,
        )
        return np.clip(action, -1.0, 1.0)


def default_controller() -> LowLevelController:
    p = default_params()
    return LowLevelController(
        lookahead=p["lookahead"],
        k_ang=p["k_ang"],
        k_dev=p["k_dev"],
        forward=p["forward"],
        bias_clip=p["bias_clip"],
    )
