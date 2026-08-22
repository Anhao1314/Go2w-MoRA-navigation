"""项目一：Unitree Go2w 双轮自平衡（窄通道机动 + 启停稳定）。

问题一（窄通道通过性）：四轮模式占用宽、原地机动半径大，无法在产线设备
     之间、窄通道、充电对接口等受限空间通行；收腿进入双轮窄体姿态后可
     显著缩小占位并支持原地转向。
问题二（启停俯仰失稳）：四轮模式下加减速会产生点头/翘尾冲击；双轮平衡
     模式用后轮力矩主动控制俯仰，把“启停冲击”变成“可控姿态”。
被优化的动作：四轮站稳 -> 收腿抬前轮 -> 双轮平衡保持 的复合动作链。

模型来源（BSD-3-Clause）：
https://github.com/unitreerobotics/unitree_mujoco

已知限制：本版本完成“四轮 -> 双轮平衡保持”；“双轮 -> 四轮”恢复动作在
手写控制器下不稳定（腿部过渡会使重心突变），列为后续迭代目标。
"""

from __future__ import annotations

import argparse
import math
import pathlib
import sys

import mujoco
import numpy as np

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mujoco_demos import common

MODEL_PATH = PROJECT_ROOT / "models" / "go2w" / "go2w_factory_scene.xml"
MEDIA_PREFIX = PROJECT_ROOT / "media" / "go2w_two_wheel_balance"

WHEEL_RADIUS = 0.086  # 米
INIT_BASE_Z = 0.436  # 四轮站姿贴地高度
LEG_FOUR = np.array([0.0, 0.67, -1.3] * 4)  # 四轮站姿（官方关节角）
# 双轮站姿（离线搜索得到）：重心落在后轮轴上、腿部静态保持力矩 ≤19% 限幅
LEG_TWO = np.array([0.0, 2.82, -1.75] * 2 + [0.0, 0.09, -2.20] * 2)
PITCH_REF = 0.225  # 双轮姿态的自然平衡角（仿真标定）

KP_LEG, KD_LEG = 300.0, 15.0  # 腿关节高刚度锁定
KP_BAL, KD_BAL, KV_BAL = -100.0, -25.0, 8.0  # 后轮平衡 LQR 型增益
LIFT_DUR = 1.5  # 收腿过渡时长


class Go2wBalanceController:
    """Go2w 双轮自平衡控制器：四轮站稳 -> 收腿 -> 双轮平衡保持。"""

    def __init__(self, model: mujoco.MjModel) -> None:
        self.leg_acts: list[tuple[int, int, int, float]] = []
        self.target_map = {7: 0.0, 8: 0.67, 9: -1.3,
                           11: 0.0, 12: 0.67, 13: -1.3,
                           15: 0.0, 16: 0.67, 17: -1.3,
                           19: 0.0, 20: 0.67, 21: -1.3}
        for name in (
            "FR_hip", "FR_thigh", "FR_calf",
            "FL_hip", "FL_thigh", "FL_calf",
            "RR_hip", "RR_thigh", "RR_calf",
            "RL_hip", "RL_thigh", "RL_calf",
        ):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            jid = model.actuator_trnid[aid, 0]
            qi = model.jnt_qposadr[jid]
            self.leg_acts.append((aid, qi, model.jnt_dofadr[jid], self.target_map[qi]))
        self.leg_range = {aid: model.actuator_ctrlrange[aid].copy() for aid, *_ in self.leg_acts}
        self.wheel_acts: list[tuple[int, int]] = []
        for name in ("RR_wheel", "RL_wheel"):
            aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            jid = model.actuator_trnid[aid, 0]
            self.wheel_acts.append((aid, model.jnt_dofadr[jid]))
        self.wheel_range = {aid: model.actuator_ctrlrange[aid].copy() for aid, _ in self.wheel_acts}

    @staticmethod
    def front_clearance(data: mujoco.MjData) -> float:
        """前轮底部离地高度（米）。"""
        z = min(data.body("FL_wheel_link").xpos[2], data.body("FR_wheel_link").xpos[2])
        return z - WHEEL_RADIUS

    def _leg_targets(self, t: float) -> np.ndarray:
        if t < 1.0:
            return LEG_FOUR
        if t < 1.0 + LIFT_DUR:
            s = 0.5 - 0.5 * math.cos(math.pi * min(1.0, (t - 1.0) / LIFT_DUR))
            return LEG_FOUR + s * (LEG_TWO - LEG_FOUR)
        return LEG_TWO

    def step(self, data: mujoco.MjData, t: float) -> None:
        tgt = self._leg_targets(t)
        for aid, qi, di, _ in self.leg_acts:
            idx = {7: 0, 8: 1, 9: 2, 11: 3, 12: 4, 13: 5,
                   15: 6, 16: 7, 17: 8, 19: 9, 20: 10, 21: 11}[qi]
            qd = tgt[idx]
            tau = KP_LEG * (qd - data.qpos[qi]) + KD_LEG * (0.0 - data.qvel[di])
            lo, hi = self.leg_range[aid]
            data.actuator(aid).ctrl[0] = max(lo, min(hi, tau))

        fl = self.front_clearance(data)
        if fl > 0.02:
            p = math.atan2(-data.body("base_link").xmat[6], data.body("base_link").xmat[0])
            pd = data.qvel[4]
            w = 0.5 * (data.qvel[21] + data.qvel[17])  # 后轮等效角速度
            tau = KP_BAL * (p - PITCH_REF) + KD_BAL * pd + KV_BAL * (0.0 - w * WHEEL_RADIUS)
            tau = max(-30.0, min(30.0, tau))
            for aid, di in self.wheel_acts:
                lo, hi = self.wheel_range[aid]
                data.actuator(aid).ctrl[0] = max(lo, min(hi, tau / 2.0))
        else:
            for aid, di in self.wheel_acts:
                lo, hi = self.wheel_range[aid]
                data.actuator(aid).ctrl[0] = max(lo, min(hi, 8.0 * (0.0 - data.qvel[di])))


def init_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    """初始位姿：四轮站姿、贴地、零速度。"""
    leg_idx = [7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21]
    data.qpos[leg_idx] = LEG_FOUR
    data.qpos[2] = INIT_BASE_Z
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0


def pitch_of(data: mujoco.MjData) -> float:
    """车体俯仰角（绕 y 轴，正值=前倾）。"""
    xmat = data.body("base_link").xmat
    return math.atan2(-xmat[6], xmat[0])


def main() -> None:
    parser = argparse.ArgumentParser(description="Unitree Go2w 双轮自平衡 Demo")
    common.add_common_args(parser)
    parser.set_defaults(duration=13.0)
    args = parser.parse_args()

    model = common.load_model(MODEL_PATH)
    controller = Go2wBalanceController(model)
    init_fn = lambda data: init_pose(model, data)

    if args.gui:
        common.run_gui(model, controller.step, init_fn=init_fn)
        return

    frames, data = common.record(
        model,
        args.duration,
        args.fps,
        controller.step,
        camera="track",
        init_fn=init_fn,
    )
    prefix = pathlib.Path(args.out) if args.out else MEDIA_PREFIX
    gif_path = common.save_gif(frames, prefix.with_suffix(".gif"), args.fps)
    mp4_path = common.save_mp4(frames, prefix.with_suffix(".mp4"), args.fps)
    print(f"已生成: {gif_path}")
    print(f"已生成: {mp4_path}")
    print(f"最终俯仰: {pitch_of(data):+.3f} rad")


if __name__ == "__main__":
    main()
