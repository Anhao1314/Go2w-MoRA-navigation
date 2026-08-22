"""Go2w 双轮自平衡/完整动作链的 Gymnasium 环境。

分层控制：腿关节沿用脚本轨迹 + 高刚度 PD，RL 只输出两个后轮力矩。
物理步长 0.002s，策略频率 100Hz（每 5 个物理步决策一次）。
"""

from __future__ import annotations

import math
import pathlib

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .terrain import TERRAIN_CX, make_terrain, write_heightfield

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "go2w" / "go2w_factory_scene.xml"
MODEL_SCENARIO_PATH = PROJECT_ROOT / "models" / "go2w" / "go2w_scenario_scene.xml"

TRAVERSE_TASKS = ("traverse_flat_slope", "traverse_slope", "traverse_curve")
TASK_SCENARIO = {
    "traverse_flat_slope": "flat_slope",
    "traverse_slope": "slope",
    "traverse_curve": "curve",
}
SCENARIOS = ("flat_slope", "slope", "curve")
GOAL_X = 4.5
SLOPE_X0 = 1.0
SLOPE_LEN = 3.0
SLOPE_THICK = 0.06
TRAVERSE_PHI_MAX = 90.0
TRAVERSE_GOAL_BONUS = 100.0
TRAVERSE_BACKWARD_PENALTY = 15.0
TRAVERSE_HEADING_PENALTY = 1.5
TRAVERSE_LATERAL_PENALTY = 2.0
TRAVERSE_V_MAX = 0.8
TRAVERSE_SPEED_PENALTY = 50.0
STALL_WINDOW = 3.0
STALL_PROGRESS = 0.1
BACKWARD_LIMIT = -1.0

# S 形弯道参数：中心线 y(x)=A*sin(2π(x-X0)/L)，走廊半宽 0.45m
CURVE_X0 = 0.8
CURVE_X1 = 4.5
CURVE_AMP = 0.35
CURVE_WAVE = 3.7
# 走廊半宽：0.40 时直线捷径 y≈0 在峰处必然撞墙（实测 240 接触），
# 沿中心线仍可无碰撞通过（实测 0 接触）
CURVE_HALF_WIDTH = 0.40
CURVE_SEGMENTS = 8

# Balance v2 抗撞击参数（力脉冲作用于基座 + 等效俯仰力矩）
BALANCE_IMPACT_N = (2, 5)          # 每 episode 撞击次数
BALANCE_IMPACT_FORCE = (40.0, 100.0)
BALANCE_IMPACT_DUR = (0.06, 0.12)
BALANCE_IMPACT_GAP = 1.0           # 撞击之间最小间隔（秒）
BALANCE_IMPACT_LEVER = 0.25        # 撞击作用点的等效力臂（m）

WHEEL_RADIUS = 0.086
INIT_BASE_Z = 0.436
TWO_WHEEL_BASE_Z = 0.1721  # 双轮站姿基座高度（仿真标定）
PITCH_REF = 0.225
TORQUE_LIMIT = 15.0
KP_LEG, KD_LEG = 300.0, 15.0
LEG_THIGH_RANGE = -0.45
LEG_CALF_RANGE = 0.8

LEG_FOUR = np.array([0.0, 0.67, -1.3] * 4)
LEG_TWO = np.array([0.0, 2.82, -1.75] * 2 + [0.0, 0.09, -2.20] * 2)
LEG_NAMES = (
    "FR_hip", "FR_thigh", "FR_calf",
    "FL_hip", "FL_thigh", "FL_calf",
    "RR_hip", "RR_thigh", "RR_calf",
    "RL_hip", "RL_thigh", "RL_calf",
)

PHYS_DT = 0.002
POLICY_HZ = 100
PHYS_STEPS_PER_POLICY = round(1.0 / POLICY_HZ / PHYS_DT)  # 5
POLICY_DT = PHYS_DT * PHYS_STEPS_PER_POLICY  # 0.01，每个策略步时长

LIFT_START = 1.0
LIFT_DUR = 1.5
HOLD_MIN = 5.0
LOWER_DUR = 2.5


def init_four_wheel_pose(model: mujoco.MjModel, data: mujoco.MjData,
                         pitch_offset: float = 0.0, z_offset: float = 0.0) -> None:
    """四轮站姿初始位姿，可选俯仰偏移与地面高度偏移（域随机化/斜坡起点）。"""
    data.qpos[:] = 0.0
    data.qpos[2] = INIT_BASE_Z + z_offset
    theta = pitch_offset * 0.5
    data.qpos[3:7] = (math.cos(theta), 0.0, math.sin(theta), 0.0)
    leg_idx = [7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21]
    data.qpos[leg_idx] = LEG_FOUR
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.xfrc_applied[:] = 0.0


def init_two_wheel_pose(model: mujoco.MjModel, data: mujoco.MjData,
                        pitch_offset: float = 0.0) -> None:
    """双轮站姿初始位姿，绕 PITCH_REF 加俯仰偏移。"""
    data.qpos[:] = 0.0
    data.qpos[2] = TWO_WHEEL_BASE_Z
    theta = (PITCH_REF + pitch_offset) * 0.5
    data.qpos[3:7] = (math.cos(theta), 0.0, math.sin(theta), 0.0)
    leg_idx = [7, 8, 9, 11, 12, 13, 15, 16, 17, 19, 20, 21]
    data.qpos[leg_idx] = LEG_TWO
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0
    data.xfrc_applied[:] = 0.0


def pitch_of(data: mujoco.MjData) -> float:
    """车体俯仰角（绕 y 轴，正值=前倾），与现有 demo 保持一致。"""
    xmat = data.body("base_link").xmat
    return math.atan2(-xmat[6], xmat[0])


def traverse_potential(x: float, phi_max: float = TRAVERSE_PHI_MAX,
                       goal_x: float = GOAL_X) -> float:
    """前进位移势能：到终点累计 phi_max。"""
    return phi_max * min(max(x, 0.0), goal_x) / goal_x


def traverse_shaping(x_new: float, x_prev: float, gamma: float = 0.99,
                     phi_max: float = TRAVERSE_PHI_MAX,
                     goal_x: float = GOAL_X) -> float:
    """势能 shaping：前进为正、倒车为负。

    用 γ·(Φ(x')−Φ(x))：完整走完累计 ≈ γ·Φ_max = 0.99×90 ≈ 89.1，
    且不会因 γ<1 的拖尾项惩罚“慢速但持续前进”。
    """
    return gamma * (
        traverse_potential(x_new, phi_max, goal_x=goal_x)
        - traverse_potential(x_prev, phi_max, goal_x=goal_x)
    )


def traverse_backward_penalty(dx: float,
                              per_meter: float = TRAVERSE_BACKWARD_PENALTY) -> float:
    """显式倒车惩罚（每米）。"""
    return -per_meter * max(0.0, -dx)


def traverse_speed_penalty(vx: float, v_max: float = TRAVERSE_V_MAX,
                           coeff: float = TRAVERSE_SPEED_PENALTY,
                           dt: float = 0.01) -> float:
    """超速惩罚：|vx| 超过 v_max 后二次惩罚（每个策略步）。"""
    return -coeff * max(0.0, abs(vx) - v_max) ** 2 * dt


def traverse_stalled(max_x: float, ref_x: float, now_t: float, ref_t: float,
                     window: float = STALL_WINDOW,
                     threshold: float = STALL_PROGRESS) -> bool:
    """最近 window 秒内 max_x 未前进 threshold 米则视为无进展。"""
    return max_x - ref_x < threshold and now_t - ref_t >= window


def curve_center_y(x: float | np.ndarray, x0: float = CURVE_X0,
                   wave: float = CURVE_WAVE, amp: float = CURVE_AMP
                   ) -> float | np.ndarray:
    """S 形弯道中心线的世界 y 坐标（amp=0 时为直线 y=0）。"""
    return amp * np.sin(2.0 * math.pi * (x - x0) / wave)


def wrap_angle(a: float) -> float:
    """角度归一化到 (-π, π]。"""
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def path_tangent_angle(x: float, is_curve: bool = True,
                       amp: float = CURVE_AMP) -> float:
    """路径切线方向（世界 yaw）。斜坡/平地恒为 0；弯道取中心线导数方向。"""
    if not is_curve:
        return 0.0
    slope = 2.0 * math.pi * amp / CURVE_WAVE * math.cos(
        2.0 * math.pi * (x - CURVE_X0) / CURVE_WAVE
    )
    return math.atan2(slope, 1.0)


def heading_error(x: float, yaw: float, is_curve: bool = True,
                  amp: float = CURVE_AMP) -> float:
    """车头 yaw 相对路径切线的航向误差（归一化到 (-π, π]）。"""
    return wrap_angle(yaw - path_tangent_angle(x, is_curve, amp))


def _curve_arc_grid(x0: float = CURVE_X0, x1: float = CURVE_X1,
                    n: int = 400, amp: float = CURVE_AMP
                    ) -> tuple[np.ndarray, np.ndarray]:
    """预计算中心线弧长网格：返回 (x 网格, 累计弧长)。"""
    xs = np.linspace(x0, x1, n)
    dx = xs[1] - xs[0]
    dys = np.gradient(curve_center_y(xs, amp=amp), dx)
    arc = np.concatenate(([0.0], np.cumsum(np.hypot(dx, dys[1:]) * dx)))
    return xs, arc


_ARC_CACHE: dict[float, tuple[np.ndarray, np.ndarray]] = {}


def curve_progress(x: float, amp: float = CURVE_AMP) -> float:
    """把 x 坐标映射为中心线累计弧长（超出范围时截断）。"""
    if amp not in _ARC_CACHE:
        _ARC_CACHE[amp] = _curve_arc_grid(amp=amp)
    xs, arc = _ARC_CACHE[amp]
    xc = float(np.clip(x, CURVE_X0, CURVE_X1))
    return float(np.interp(xc, xs, arc))


_CURVE_XS, _CURVE_ARC = _curve_arc_grid()
CURVE_TOTAL_LEN = float(_CURVE_ARC[-1])
GOAL_ARC_MIN = 0.3
GOAL_ARC_MAX = CURVE_TOTAL_LEN

MULTI_SEGMENT_DEFAULT = [
    {
        "x0": 0.8,
        "x1": 4.3,
        "amp": 0.12,
        "corridor_width": 0.50,
        "goal_tolerance": 0.15,
        "stop_duration": 0.3,
        "stop_speed_threshold": 0.2,
    },
    {
        "x0": 4.3,
        "x1": 10.8,
        "amp": 0.35,
        "corridor_width": 0.40,
        "goal_tolerance": 0.15,
        "stop_duration": 0.3,
        "stop_speed_threshold": 0.2,
    },
]

JUNCTION_DEFAULT = {
    "common_length": 2.2,
    "branch_length": 3.5,
    "branch_amplitude": 0.30,
    "corridor_common": 0.50,
    "corridor_branch": 0.40,
    "decision_x_start": 1.8,
    "decision_x_end": 2.2,
    "common_amplitude": 0.05,
    "goal_tolerance": 0.5,
    "goal_x": 5.5,
    "goal_y": 0.30,
}


def x_for_arc(arc: float, amp: float = CURVE_AMP) -> float:
    """弧长 -> 世界 x 的单调逆映射（用于目标点切线角计算）。"""
    if amp not in _ARC_CACHE:
        _ARC_CACHE[amp] = _curve_arc_grid(amp=amp)
    xs, arcs = _ARC_CACHE[amp]
    a = float(np.clip(arc, 0.0, float(arcs[-1])))
    return float(np.interp(a, arcs, xs))


class Go2wEnv(gym.Env):
    """Go2w 分层控制 RL 环境。

    task="balance"              : 四轮站姿 + 随机撞击干扰，抗撞保持站立（10s）。
    task="full_chain"           : 四轮 -> 收腿 -> 双轮 -> 放腿 -> 四轮（15s）。
    task="full_chain_simple"    : 课程版：四轮 -> 收腿 -> 双轮保持 5s 即成功（不放腿）。
    task="traverse_flat_slope"  : 平地接 25~35° 陡坡上坡穿越。
    task="traverse_slope"       : 15~25° 缓坡上坡穿越。
    task="traverse_curve"       : S 形弯道穿越（需差速转向）。
    domain_randomize=True 时启用质量/摩擦/初始俯仰/推力/观测/控制随机化。
    disturbance=True 时施加确定性的 40N/0.2s 推力（用于评估扰动恢复）。
    scenario 可覆盖 traverse 任务默认场景（用于查看器/评估）。
    terrain="hfield" 时斜坡任务使用运行时生成的高度场地形（平滑坡面+随机凸起），
    terrain="boxes" 时保留旧盒状斜坡几何（回归/对比用）。
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        task: str = "balance",
        domain_randomize: bool = True,
        episode_seconds: float | None = None,
        disturbance: bool = False,
        scenario: str | None = None,
        terrain: str = "hfield",
        heading_penalty: float = TRAVERSE_HEADING_PENALTY,
        corridor_width: float = CURVE_HALF_WIDTH,
        curve_amplitude: float = CURVE_AMP,
        reward_version: str = "v4",
        fall_penalty: float = -5.0,
        goal_bonus: float = TRAVERSE_GOAL_BONUS,
        goal_arc: float | None = None,
        multi_segment: bool = False,
        segments: list | None = None,
        subgoal_switch: bool = False,
        scope: int = 2,
        junction: bool = False,
        junction_config: dict | None = None,
        target_goal: str = "A",
        branch_selected: str | None = None,
    ) -> None:
        super().__init__()
        if task not in ("balance", "full_chain", "full_chain_simple", *TRAVERSE_TASKS):
            raise ValueError(f"未知 task: {task}")
        if terrain not in ("hfield", "boxes"):
            raise ValueError(f"未知 terrain: {terrain}")
        if heading_penalty <= 0.0:
            raise ValueError(f"heading_penalty 必须为正: {heading_penalty}")
        if corridor_width <= 0.0:
            raise ValueError(f"corridor_width 必须为正: {corridor_width}")
        if curve_amplitude < 0.0:
            raise ValueError(f"curve_amplitude 不能为负: {curve_amplitude}")
        if reward_version not in ("v4", "simple", "v5", "mseg", "junction"):
            raise ValueError(f"未知 reward_version: {reward_version}")
        self.task = task
        self.terrain = terrain
        self.heading_penalty = float(heading_penalty)
        self.corridor_width = float(corridor_width)
        self.curve_amplitude = float(curve_amplitude)
        self.reward_version = reward_version
        self.fall_penalty = float(fall_penalty)
        self.goal_bonus = float(goal_bonus)
        self._goal_arc: float | None = (
            float(np.clip(goal_arc, GOAL_ARC_MIN, GOAL_ARC_MAX))
            if goal_arc is not None
            else None
        )
        self.multi_segment = bool(multi_segment)
        self.subgoal_switch = bool(subgoal_switch)
        self.scope = int(scope)
        self.junction = bool(junction)
        self.junction_config = dict(junction_config or JUNCTION_DEFAULT)
        self.target_goal = target_goal if target_goal in ("A", "B") else "A"
        self.branch_selected = (
            branch_selected if branch_selected in ("left", "right") else None
        )
        self.segments = list(segments or MULTI_SEGMENT_DEFAULT)
        self._segment_idx = 0
        self._subgoal_reached_a = False
        self._passed_subgoal = False
        self._subgoal_hold_start: float | None = None
        self._subgoal_hold_ok = False
        self._subgoal_in_range = False
        self._docking_done = False
        self._start_at_subgoal = False
        self._prev_dist_to_subgoal = 0.0
        self._subgoal_bonus_given = False
        self._docking_bonus_given = False
        self._switch_bonus_given = False
        self._final_bonus_given = False
        self._junction_reached_bonus_given = False
        self._junction_decision_bonus_given = False
        self._junction_goal_bonus_given = False
        self._junction_wrong_branch_given = False
        self._prev_junction_x = 0.0
        self.domain_randomize = bool(domain_randomize)
        self.disturbance = bool(disturbance)
        if scenario is not None and scenario not in SCENARIOS:
            raise ValueError(f"未知 scenario: {scenario}")
        self._fixed_scenario = scenario
        self.scenario = scenario or TASK_SCENARIO.get(task, "flat_slope")
        self.episode_seconds = float(
            episode_seconds
            or (
                30.0
                if multi_segment and task == "traverse_curve"
                else 10.0
                if task == "balance"
                else 12.0
                if task == "full_chain_simple"
                else 15.0
            )
        )
        self.max_steps = int(round(self.episode_seconds * POLICY_HZ))
        self.action_dim = 6 if task in TRAVERSE_TASKS else 2
        self.obs_dim = 61 if task in TRAVERSE_TASKS else 44

        model_path = MODEL_SCENARIO_PATH if task in TRAVERSE_TASKS else MODEL_PATH
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        self._base_body_mass = self.model.body_mass.copy()
        self._base_body_inertia = self.model.body_inertia.copy()
        self._base_geom_friction = self.model.geom_friction.copy()
        self._floor_geom_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        self._wheel_geom_ids: list[int] = []
        for body_name in ("FL_wheel_link", "FR_wheel_link", "RL_wheel_link", "RR_wheel_link"):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            for gid in range(self.model.ngeom):
                if self.model.geom_bodyid[gid] == bid:
                    self._wheel_geom_ids.append(gid)

        # 腿关节映射与高刚度 PD 参数
        self.leg_acts: list[tuple[int, int, int]] = []
        self.leg_ranges: dict[int, np.ndarray] = {}
        self.leg_qpos_idx: list[int] = []
        self.leg_qvel_idx: list[int] = []
        self._target_idx: dict[int, int] = {}
        for idx, name in enumerate(LEG_NAMES):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            jid = self.model.actuator_trnid[aid, 0]
            qi = self.model.jnt_qposadr[jid]
            di = self.model.jnt_dofadr[jid]
            self.leg_acts.append((aid, qi, di))
            self.leg_qpos_idx.append(qi)
            self.leg_qvel_idx.append(di)
            self._target_idx[qi] = idx
            self.leg_ranges[aid] = self.model.actuator_ctrlrange[aid].copy()

        self.wheel_aids: list[int] = []
        self.wheel_dofs: list[int] = []
        for name in ("FR_wheel", "FL_wheel", "RR_wheel", "RL_wheel"):
            aid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            jid = self.model.actuator_trnid[aid, 0]
            self.wheel_aids.append(aid)
            self.wheel_dofs.append(self.model.jnt_dofadr[jid])
        self.rear_wheel_aids = [self.wheel_aids[2], self.wheel_aids[3]]
        self.rear_wheel_dofs = [self.wheel_dofs[2], self.wheel_dofs[3]]
        self.front_wheel_aids = [self.wheel_aids[0], self.wheel_aids[1]]
        if task in TRAVERSE_TASKS:
            self.action_wheel_aids = self.wheel_aids
            self.zero_wheel_aids: list[int] = []
            self.obs_wheel_dofs = self.wheel_dofs
        else:
            self.action_wheel_aids = self.rear_wheel_aids
            self.zero_wheel_aids = self.front_wheel_aids
            self.obs_wheel_dofs = self.rear_wheel_dofs

        self.scn_geom_ids: list[int] = []
        self._scn_ramp_id = -1
        self._scn_ramp_flat_id = -1
        self._scn_goal_id = -1
        self._scn_terrain_gid = -1
        self._scn_terrain_hid = -1
        self._hfield_slice: slice | None = None
        self._scn_bump_ids: list[int] = []
        self._scn_speed_ids: list[int] = []
        self._scn_curve_l_ids: list[int] = []
        self._scn_curve_r_ids: list[int] = []
        self._scn_curve_l_bodies: list[int] = []
        self._scn_curve_r_bodies: list[int] = []
        self._scn_mseg_l_ids: list[int] = []
        self._scn_mseg_r_ids: list[int] = []
        self._scn_mseg_l_bodies: list[int] = []
        self._scn_mseg_r_bodies: list[int] = []
        self._scn_ramp_body_id = -1
        self._scn_ramp_flat_body_id = -1
        self.scn_body_ids: list[int] = []
        for i in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i)
            if name and name.startswith("scn_"):
                self.scn_geom_ids.append(i)
            if name == "scn_ramp":
                self._scn_ramp_id = i
            elif name == "scn_ramp_flat":
                self._scn_ramp_flat_id = i
            elif name == "scn_goal":
                self._scn_goal_id = i
            elif name and name.startswith("scn_bump_"):
                self._scn_bump_ids.append(i)
            elif name and name.startswith("scn_speed_"):
                self._scn_speed_ids.append(i)
            elif name and name.startswith("scn_curve_l_"):
                self._scn_curve_l_ids.append(i)
            elif name and name.startswith("scn_curve_r_"):
                self._scn_curve_r_ids.append(i)
            elif name and name.startswith("scn_mseg_l_"):
                self._scn_mseg_l_ids.append(i)
            elif name and name.startswith("scn_mseg_r_"):
                self._scn_mseg_r_ids.append(i)
        # 需要旋转的几何（弯道墙/斜坡）包在 body 里：运行时写 body_pos/body_quat
        # （MuJoCo 3.11 运行时写 geom_quat 不生效，见 go2w_scenario_scene.xml 注释）
        for i in range(CURVE_SEGMENTS):
            for side, target in (("l", self._scn_curve_l_bodies),
                                 ("r", self._scn_curve_r_bodies)):
                bid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"scn_curve_{side}_{i}_body"
                )
                if bid >= 0:
                    target.append(bid)
                    self.scn_body_ids.append(bid)
        for i in range(CURVE_SEGMENTS):
            for side, target in (("l", self._scn_mseg_l_bodies),
                                 ("r", self._scn_mseg_r_bodies)):
                bid = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, f"scn_mseg_{side}_{i}_body"
                )
                if bid >= 0:
                    target.append(bid)
                    self.scn_body_ids.append(bid)
        for name, attr in (("scn_ramp_body", "_scn_ramp_body_id"),
                           ("scn_ramp_flat_body", "_scn_ramp_flat_body_id")):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                setattr(self, attr, bid)
                self.scn_body_ids.append(bid)
        self._scn_terrain_gid = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "scn_terrain"
        )
        if self._scn_terrain_gid >= 0:
            self._scn_terrain_hid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_HFIELD, "scn_terrain"
            )
            if self._scn_terrain_hid >= 0:
                adr = int(self.model.hfield_adr[self._scn_terrain_hid])
                n = int(
                    self.model.hfield_nrow[self._scn_terrain_hid]
                    * self.model.hfield_ncol[self._scn_terrain_hid]
                )
                self._hfield_slice = slice(adr, adr + n)

        self.base_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
        )

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.action_dim,), dtype=np.float32
        )

        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._leg_cmd = np.zeros(2, dtype=np.float32)
        self._steps = 0
        self._episode_count = 0
        self._stage = 0
        self._phi = 0.0
        self._prev_phi = 0.0
        self._lower_started = False
        self._lower_t0 = 0.0
        self._stable_seconds = 0.0
        self._stable_bonus_given = False
        self._recovery_bonus_given = False
        self._simple_success = False
        self._push_force = 0.0
        self._push_remaining = 0
        self._push_t0 = 1e9
        self._impacts: list[dict] = []
        self._traverse_start_z = 0.0
        self._goal_reached = False
        self._early_stopped = False
        self._prev_x = 0.0
        self._prev_progress = 0.0
        self._max_x = 0.0
        self._progress_ref_x = 0.0
        self._progress_ref_t = 0.0
        self._lift_started = False
        self._lift_done = False
        self._lift_t0 = 0.0
        self._lower_t0 = 0.0

    # ------------------------------------------------------------------
    # Gymnasium 接口
    # ------------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if options and "goal_arc" in options:
            self._goal_arc = float(
                np.clip(options["goal_arc"], GOAL_ARC_MIN, GOAL_ARC_MAX)
            )
        if options and "scope" in options:
            self.scope = int(options["scope"])
        if options and "target_goal" in options:
            self.target_goal = (
                options["target_goal"] if options["target_goal"] in ("A", "B") else "A"
            )
        if options and "branch_selected" in options:
            self.branch_selected = (
                options["branch_selected"]
                if options["branch_selected"] in ("left", "right")
                else None
            )
        start_junction = bool(options and options.get("start_at_junction"))
        self._start_at_subgoal = bool(
            options and options.get("start_at_subgoal") == "A"
        )
        self._randomize_model()
        pitch_offset = (
            float(self.np_random.uniform(-0.08, 0.08))
            if self.domain_randomize
            else 0.0
        )
        if self.task in TRAVERSE_TASKS:
            self._traverse_start_z = self._setup_scenario()
            init_four_wheel_pose(
                self.model, self.data, pitch_offset, z_offset=self._traverse_start_z
            )
            self._goal_reached = False
            self._early_stopped = False
            self._segment_idx = 0
            self._subgoal_reached_a = False
            self._passed_subgoal = False
            self._subgoal_hold_start = None
            self._subgoal_hold_ok = False
            self._subgoal_in_range = False
            self._docking_done = False
            self._prev_dist_to_subgoal = 0.0
            self._subgoal_bonus_given = False
            self._docking_bonus_given = False
            self._switch_bonus_given = False
            self._final_bonus_given = False
            self._junction_reached_bonus_given = False
            self._junction_decision_bonus_given = False
            self._junction_goal_bonus_given = False
            self._junction_wrong_branch_given = False
            self._prev_junction_x = 0.0
            if self.junction:
                self.branch_selected = None
                if self.scope == 1:
                    self.branch_selected = "left" if self.np_random.random() < 0.5 else "right"
                if start_junction:
                    self.data.qpos[0] = self.junction_config["decision_x_end"]
                    self.data.qpos[1] = 0.0
                    self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
                    self.data.qvel[:] = 0.0
                    mujoco.mj_forward(self.model, self.data)
                    self._prev_junction_x = float(self.data.qpos[0])
            if self._start_at_subgoal:
                ax, ay = self._segment_goal(0)
                self.data.qpos[0] = ax
                self.data.qpos[1] = ay
                yaw0 = self._segment_tangent_angle(ax, 1)
                self.data.qpos[3:7] = (
                    math.cos(yaw0 / 2.0), 0.0, 0.0, math.sin(yaw0 / 2.0),
                )
                self.data.qvel[:] = 0.0
                mujoco.mj_forward(self.model, self.data)
                self._segment_idx = 1
                self._subgoal_reached_a = True
                self._passed_subgoal = True
                self._docking_done = True
                self._subgoal_bonus_given = True
                self._docking_bonus_given = True
                self._switch_bonus_given = True
                gx, gy = self._segment_goal(1)
                self._prev_dist_to_subgoal = math.hypot(ax - gx, ay - gy)
        elif self.task == "balance":
            # Balance v2：四轮站姿 + 随机撞击干扰
            init_four_wheel_pose(self.model, self.data, pitch_offset)
            self._impacts = self._schedule_impacts()
        else:
            init_four_wheel_pose(self.model, self.data, pitch_offset)

        self._steps = 0
        self._last_action = np.zeros(self.action_dim, dtype=np.float32)
        self._leg_cmd = np.zeros(2, dtype=np.float32)
        self._lower_started = False
        self._lower_t0 = 0.0
        self._stable_seconds = 0.0
        self._stable_bonus_given = False
        self._recovery_bonus_given = False
        self._simple_success = False
        self._push_force = 0.0
        self._push_remaining = 0
        self._push_t0 = 1e9

        if self.domain_randomize and self.np_random.random() < 0.2:
            self._push_force = float(self.np_random.uniform(10.0, 40.0))
            self._push_remaining = int(0.2 / PHYS_DT)
            self._push_t0 = float(self.np_random.uniform(1.0, max(1.0, self.episode_seconds - 1.0)))
        elif self.disturbance:
            self._push_force = 40.0
            self._push_remaining = int(0.2 / PHYS_DT)
            self._push_t0 = 5.0

        mujoco.mj_forward(self.model, self.data)
        if self.task in TRAVERSE_TASKS:
            x = float(self.data.body("base_link").xpos[0])
            self._prev_x = x
            self._prev_progress = self._progress_metric(x)
            self._max_x = x
            self._progress_ref_x = x
            self._progress_ref_t = self.data.time
        _, self._stage, self._phi = self._phase_info(self.data.time)
        self._prev_phi = self._phi
        return self._get_obs(), self._info()

    def set_domain_randomize(self, flag: bool) -> None:
        """运行时切换域随机化开关（用于课程学习）。"""
        self.domain_randomize = bool(flag)

    def step(self, action):
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(-1), -1.0, 1.0)
        if action.shape != (self.action_dim,):
            raise ValueError(f"动作维度应为 {self.action_dim}，收到 {action.shape}")

        if self.task in TRAVERSE_TASKS:
            # 混合控制：前 4 维是四轮力矩，后 2 维是前后腿组姿态命令
            torque = action[:4] * TORQUE_LIMIT
            self._leg_cmd = action[4:6].astype(np.float32)
        else:
            torque = action * TORQUE_LIMIT
        if self.domain_randomize:
            torque = torque + self.np_random.normal(0.0, 0.5, size=torque.shape)
        torque = np.clip(torque, -TORQUE_LIMIT, TORQUE_LIMIT)

        terminated = False
        for _ in range(PHYS_STEPS_PER_POLICY):
            t = self.data.time
            if self._check_termination():
                terminated = True
                break
            force_applied = False
            if self._push_remaining > 0 and t >= self._push_t0:
                self.data.xfrc_applied[self.base_body_id, 0] = self._push_force
                self._push_remaining -= 1
                force_applied = True
            elif self.task == "balance":
                # Balance v2：撞击窗口内施加力脉冲 + 等效俯仰力矩
                for imp in self._impacts:
                    if imp["t0"] <= t < imp["t0"] + imp["dur"]:
                        self.data.xfrc_applied[self.base_body_id, 0] = imp["dir"] * imp["force"]
                        self.data.xfrc_applied[self.base_body_id, 5] = (
                            -imp["dir"] * imp["force"] * BALANCE_IMPACT_LEVER
                        )
                        force_applied = True
                        break
            if not force_applied:
                self.data.xfrc_applied[self.base_body_id] = 0.0

            target, stage, phi = self._phase_info(t)
            self._stage = stage
            self._phi = phi
            if self.task in TRAVERSE_TASKS:
                # 混合腿控：traverse 全部场景由 RL 腿命令覆盖脚本姿态
                target = self._leg_target_from_cmds(float(action[4]), float(action[5]))
            for aid, qi, di in self.leg_acts:
                qd = target[self._target_idx[qi]]
                tau = KP_LEG * (qd - self.data.qpos[qi]) + KD_LEG * (0.0 - self.data.qvel[di])
                lo, hi = self.leg_ranges[aid]
                self.data.ctrl[aid] = max(lo, min(hi, tau))
            for aid in self.zero_wheel_aids:
                self.data.ctrl[aid] = 0.0
            for aid, tau in zip(self.action_wheel_aids, torque):
                self.data.ctrl[aid] = tau

            mujoco.mj_step(self.model, self.data)
            if self._check_termination():
                terminated = True
                break

        self._steps += 1
        if self.multi_segment:
            self._update_subgoals()
        self._goal_reached = self.task in TRAVERSE_TASKS and self._traverse_goal_reached()
        b_arrived = bool(
            self.multi_segment and self._goal_reached and self._segment_idx == 1
        )
        j_arrived = bool(self.junction and self._goal_reached)
        early_goal_trunc = b_arrived or j_arrived
        if self.task == "full_chain_simple" and self._simple_success:
            terminated = True
        elif self._goal_reached and not early_goal_trunc:
            terminated = True
        elif self.task in TRAVERSE_TASKS:
            self._update_progress_tracking()
            if self._early_stopped:
                terminated = True
        truncated = early_goal_trunc or (self._steps >= self.max_steps and not terminated)
        reward = self._get_reward(action, terminated, truncated)
        self._last_action = action.astype(np.float32)
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), self._info()

    # ------------------------------------------------------------------
    # 内部实现
    # ------------------------------------------------------------------
    def _randomize_model(self) -> None:
        """逐 episode 域随机化：质量、惯性、摩擦。"""
        self.model.body_mass[:] = self._base_body_mass
        self.model.body_inertia[:] = self._base_body_inertia
        self.model.geom_friction[:] = self._base_geom_friction
        if not self.domain_randomize:
            return
        k_mass = float(self.np_random.uniform(0.9, 1.1))
        self.model.body_mass *= k_mass
        self.model.body_inertia *= k_mass
        k_fric = float(self.np_random.uniform(0.7, 1.3))
        fric_ids = self._wheel_geom_ids + [self._floor_geom_id]
        if self._scn_terrain_gid >= 0:
            fric_ids = fric_ids + [self._scn_terrain_gid]
        for gid in fric_ids:
            self.model.geom_friction[gid] *= k_fric

    def _setup_scenario(self) -> float:
        """按当前场景激活/随机化几何体，返回机器人起始地面高度偏移。"""
        # 需要旋转的几何先整体挪远（body 级），再由各场景分支激活
        for bid in self.scn_body_ids:
            self.model.body_pos[bid] = (100.0, 1000.0, 100.0)
        curve_geoms = set(self._scn_curve_l_ids + self._scn_curve_r_ids)
        for gid in self.scn_geom_ids:
            if (
                gid != self._scn_goal_id
                and gid != self._scn_terrain_gid
                and gid not in curve_geoms
                and gid != self._scn_ramp_id
                and gid != self._scn_ramp_flat_id
            ):
                # 未使用场景的几何体挪到远处：contype 在 XML 中恒为 1（运行时改不生效），
                # 挪远即可同时隐藏渲染并避免碰撞
                self.model.geom_pos[gid] = (100.0, 1000.0, 100.0)

        scenario = self.scenario
        if self.junction:
            if self._scn_terrain_gid >= 0:
                self.model.geom_pos[self._scn_terrain_gid] = (100.0, 1000.0, 100.0)
            return 0.0
        if self.multi_segment and scenario == "curve":
            if self._scn_terrain_gid >= 0:
                self.model.geom_pos[self._scn_terrain_gid] = (100.0, 1000.0, 100.0)
            n = CURVE_SEGMENTS
            for seg_idx, seg in enumerate(self.segments):
                xs = np.linspace(seg["x0"], seg["x1"], n + 1)
                mids = (xs[:-1] + xs[1:]) / 2.0
                seg_len = (seg["x1"] - seg["x0"]) / n
                length = seg["x1"] - seg["x0"]
                amp = seg["amp"]
                for i in range(n):
                    xc = float(mids[i])
                    yc = float(
                        self._segment_center_y(xc, seg_idx)
                    )
                    slope = amp * 2.0 * math.pi / length * math.cos(
                        2.0 * math.pi * (xc - seg["x0"]) / length
                    )
                    ang = math.atan(slope)
                    nx, ny = -math.sin(ang), math.cos(ang)
                    half = seg_len / 2.0 + 0.03
                    cw = seg["corridor_width"]
                    for side, bodies, geoms in (
                        ("l", self._scn_mseg_l_bodies, self._scn_mseg_l_ids),
                        ("r", self._scn_mseg_r_bodies, self._scn_mseg_r_ids),
                    ):
                        idx = seg_idx * n + i
                        if idx >= len(bodies) or idx >= len(geoms):
                            continue
                        sign = -1.0 if side == "l" else 1.0
                        bid = bodies[idx]
                        gid = geoms[idx]
                        self.model.body_pos[bid] = (
                            xc + sign * nx * cw,
                            yc + sign * ny * cw,
                            0.25,
                        )
                        self.model.body_quat[bid] = (
                            math.cos(ang / 2.0), 0.0, 0.0, math.sin(ang / 2.0),
                        )
                        self.model.geom_size[gid] = (half, 0.05, 0.25)
            return 0.0

        if scenario in ("flat_slope", "slope"):
            if scenario == "flat_slope":
                angle = float(self.np_random.uniform(25.0, 35.0)) if self.domain_randomize else 30.0
            else:
                angle = float(self.np_random.uniform(15.0, 25.0)) if self.domain_randomize else 20.0

            if self.terrain == "hfield" and self._scn_terrain_gid >= 0 and self._hfield_slice is not None:
                # 运行时高度场地形：平滑坡面 + 随机凸起，逐 episode 重新生成
                z = make_terrain(angle, rng=self.np_random)
                write_heightfield(self.model, self._scn_terrain_hid, z)
                self.model.geom_pos[self._scn_terrain_gid] = (TERRAIN_CX, 0.0, 0.0)
                return 0.0

            # 旧盒状斜坡路径（terrain="boxes"）：隐藏高度场，恢复盒体几何
            if self._scn_terrain_gid >= 0:
                self.model.geom_pos[self._scn_terrain_gid] = (100.0, 1000.0, 100.0)
            th = math.radians(angle)
            # 统一为上坡：平地(0~1m) -> 斜坡(1~4m) -> 坡顶平地(4~4.5m)
            rise = SLOPE_LEN * math.tan(th)
            h0 = 0.0
            cx = SLOPE_X0 + SLOPE_LEN / 2.0
            cz = rise / 2.0 - SLOPE_THICK / (2.0 * math.cos(th))
            self.model.body_pos[self._scn_ramp_body_id] = (cx, 0.0, cz)
            self.model.body_quat[self._scn_ramp_body_id] = (
                math.cos(-th / 2.0), 0.0, math.sin(-th / 2.0), 0.0,
            )
            self.model.geom_size[self._scn_ramp_id] = (
                SLOPE_LEN / 2.0, 1.0, SLOPE_THICK / 2.0,
            )

            center_x = (SLOPE_X0 + SLOPE_LEN + GOAL_X) / 2.0
            half_len = (GOAL_X - SLOPE_X0 - SLOPE_LEN) / 2.0
            self.model.body_pos[self._scn_ramp_flat_body_id] = (
                center_x, 0.0, rise - SLOPE_THICK / 2.0,
            )
            self.model.body_quat[self._scn_ramp_flat_body_id] = (1.0, 0.0, 0.0, 0.0)
            self.model.geom_size[self._scn_ramp_flat_id] = (
                half_len, 1.0, SLOPE_THICK / 2.0,
            )
            return h0

        if scenario == "curve":
            if self._scn_terrain_gid >= 0:
                self.model.geom_pos[self._scn_terrain_gid] = (100.0, 1000.0, 100.0)
            n = CURVE_SEGMENTS
            xs = np.linspace(CURVE_X0, CURVE_X1, n + 1)
            mids = (xs[:-1] + xs[1:]) / 2.0
            seg_len = (CURVE_X1 - CURVE_X0) / n
            for i in range(n):
                xc = float(mids[i])
                yc = curve_center_y(xc)
                slope = 2.0 * math.pi * CURVE_AMP / CURVE_WAVE * math.cos(
                    2.0 * math.pi * (xc - CURVE_X0) / CURVE_WAVE
                )
                ang = math.atan(slope)
                nx, ny = -math.sin(ang), math.cos(ang)  # 中心线法向
                half = seg_len / 2.0 + 0.03  # 墙段重叠增大，防止段间缝隙
                for body_ids, sign in ((self._scn_curve_l_bodies, -1.0),
                                       (self._scn_curve_r_bodies, +1.0)):
                    bid = body_ids[i]
                    self.model.body_pos[bid] = (
                        xc + sign * nx * self.corridor_width,
                        yc + sign * ny * self.corridor_width,
                        0.25,
                    )
                    self.model.body_quat[bid] = (
                        math.cos(ang / 2.0), 0.0, 0.0, math.sin(ang / 2.0),
                    )
                    gid = self._scn_curve_l_ids[i] if sign < 0 else self._scn_curve_r_ids[i]
                    self.model.geom_size[gid] = (half, 0.05, 0.25)
            return 0.0

        if self._scn_terrain_gid >= 0:
            self.model.geom_pos[self._scn_terrain_gid] = (100.0, 1000.0, 100.0)
        return 0.0

    def _traverse_goal_reached(self) -> bool:
        if self.junction:
            if self.scope == 0:
                return self._junction_reached()
            if self.branch_selected is None:
                return False
            x = float(self.data.body("base_link").xpos[0])
            y = float(self.data.body("base_link").xpos[1])
            gx, gy = self._junction_goal()
            return (
                math.hypot(x - gx, y - gy)
                <= self.junction_config["goal_tolerance"]
            )
        if self.multi_segment:
            return bool(self._goal_reached)
        x = float(self.data.body("base_link").xpos[0])
        if self.scenario == "curve":
            if abs(self.curve_amplitude - CURVE_AMP) < 1e-9:
                # 默认振幅：维持原有弧长判定，行为完全不变
                return curve_progress(x) >= self._goal_arc_len() - 0.05
            # 课程学习阶段：按世界 x 判到达（不同振幅弧长不同）
            return x >= GOAL_X - 0.05
        return x >= GOAL_X

    def _update_progress_tracking(self) -> None:
        x = float(self.data.body("base_link").xpos[0])
        p = self._progress_metric(x)
        self._max_x = max(self._max_x, p)
        if self._max_x - self._progress_ref_x >= STALL_PROGRESS:
            self._progress_ref_x = self._max_x
            self._progress_ref_t = self.data.time
        self._early_stopped = (
            traverse_stalled(
                self._max_x, self._progress_ref_x, self.data.time, self._progress_ref_t
            )
            and p < self._goal_arc_len() - 0.5
        )

    def _body_vel(self) -> tuple[float, float]:
        """基座车体系 x/y 方向线速度（m/s）。"""
        world_vel = self.data.cvel[self.base_body_id, 0:3]
        xmat = self.data.body("base_link").xmat
        vx = float(np.dot(world_vel, xmat[0:3]))
        vy = float(np.dot(world_vel, xmat[3:6]))
        return vx, vy

    def _yaw_of(self) -> float:
        q = self.data.xquat[self.base_body_id]
        w, x, y, z = q
        return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    def _phase_info(self, t: float):
        """返回 (腿目标关节角, 阶段, 进度 phi)。"""
        if self.task == "balance":
            # Balance v2：四轮站姿，腿保持 LEG_FOUR 脚本姿态
            return LEG_FOUR, 0, 0.0
        if self.task in TRAVERSE_TASKS:
            # traverse 任务腿由 RL 混合控制（step 中覆盖目标）
            return LEG_FOUR, 0, 0.0
        if t < LIFT_START:
            return LEG_FOUR, 0, 0.0
        if t < LIFT_START + LIFT_DUR:
            s = 0.5 - 0.5 * math.cos(math.pi * (t - LIFT_START) / LIFT_DUR)
            return LEG_FOUR + s * (LEG_TWO - LEG_FOUR), 1, s

        if self.task == "full_chain_simple":
            # 课程版：收腿后一直保持双轮，不放腿恢复四轮
            return LEG_TWO, 2, 1.0

        if not self._lower_started:
            pitch = pitch_of(self.data)
            if t - (LIFT_START + LIFT_DUR) >= HOLD_MIN and abs(pitch - PITCH_REF) < 0.08:
                self._lower_started = True
                self._lower_t0 = t
            else:
                return LEG_TWO, 2, 1.0

        s_low = 0.5 - 0.5 * math.cos(
            math.pi * min(1.0, (t - self._lower_t0) / LOWER_DUR)
        )
        if s_low <= 0.0:
            return LEG_TWO, 2, 1.0
        if s_low >= 1.0:
            return LEG_FOUR, 4, 2.0
        return LEG_TWO + s_low * (LEG_FOUR - LEG_TWO), 3, 1.0 + s_low

    def _leg_target_from_cmds(self, front_cmd: float, rear_cmd: float) -> np.ndarray:
        """把前后腿组命令映射成 12 维腿目标关节角（hip 保持 0）。

        腿组划分（与 LEG_NAMES 一致）：
          front = FR(0-2) + FL(3-5)，rear = RR(6-8) + RL(9-11)；
        命令 +1 = 升高基座、-1 = 蹲低（符号由标定探针确认后固定）。
        """
        base = LEG_FOUR.copy()
        for idx in (1, 4):      # FR_thigh, FL_thigh
            base[idx] += front_cmd * LEG_THIGH_RANGE
        for idx in (2, 5):      # FR_calf, FL_calf
            base[idx] += front_cmd * LEG_CALF_RANGE
        for idx in (7, 10):     # RR_thigh, RL_thigh
            base[idx] += rear_cmd * LEG_THIGH_RANGE
        for idx in (8, 11):     # RR_calf, RL_calf
            base[idx] += rear_cmd * LEG_CALF_RANGE
        return base

    def _pitch_ref(self, stage: int) -> float:
        if self.task == "balance":
            return 0.0  # Balance v2 四轮站立：俯仰基准为水平
        return PITCH_REF if stage in (1, 2) else 0.0

    def _schedule_impacts(self) -> list[dict]:
        """生成 2~4 次撞击：沿 ±x 的力脉冲 + 等效俯仰力矩，间隔 ≥1s。"""
        impacts: list[dict] = []
        t = 1.0
        n = int(self.np_random.integers(BALANCE_IMPACT_N[0], BALANCE_IMPACT_N[1]))
        for _ in range(n):
            t0 = t + float(self.np_random.uniform(1.0, 2.5))
            impacts.append(
                {
                    "t0": t0,
                    "dur": float(self.np_random.uniform(*BALANCE_IMPACT_DUR)),
                    "dir": float(self.np_random.choice([-1.0, 1.0])),
                    "force": float(self.np_random.uniform(*BALANCE_IMPACT_FORCE)),
                }
            )
            t = t0 + impacts[-1]["dur"] + BALANCE_IMPACT_GAP
        return impacts

    def _progress_metric(self, x: float) -> float:
        """traverse 进度指标：弯道用弧长，其余用世界 x。"""
        if self.junction:
            return x
        if self.multi_segment:
            return x
        if self.scenario == "curve":
            return curve_progress(x, amp=self.curve_amplitude)
        return x

    def _current_segment(self) -> int:
        return int(self._segment_idx)

    def _segment_center_y(self, x: float, idx: int | None = None) -> float:
        idx = self._segment_idx if idx is None else idx
        seg = self.segments[idx]
        x0, length, amp = seg["x0"], seg["x1"] - seg["x0"], seg["amp"]
        if idx == 0:
            return float(amp * math.sin(2.0 * math.pi * (x - x0) / length))
        ya = float(
            self.segments[0]["amp"]
            * math.sin(2.0 * math.pi * (self.segments[0]["x1"] - self.segments[0]["x0"])
                        / (self.segments[0]["x1"] - self.segments[0]["x0"]))
        )
        return ya + float(amp * math.sin(2.0 * math.pi * (x - x0) / length))

    def _segment_tangent_angle(self, x: float, idx: int | None = None) -> float:
        idx = self._segment_idx if idx is None else idx
        seg = self.segments[idx]
        x0, length, amp = seg["x0"], seg["x1"] - seg["x0"], seg["amp"]
        slope = amp * 2.0 * math.pi / length * math.cos(
            2.0 * math.pi * (x - x0) / length
        )
        return float(math.atan(slope))

    def _segment_deviation(self, x: float, y: float, idx: int | None = None) -> float:
        return float(y - self._segment_center_y(x, idx))

    def _segment_goal(self, idx: int | None = None) -> tuple[float, float]:
        idx = self._segment_idx if idx is None else idx
        seg = self.segments[idx]
        gx = float(seg["x1"])
        gy = self._segment_center_y(gx, idx)
        return gx, gy

    def _check_subgoal_reached(
        self, x: float, y: float, vx: float
    ) -> bool:
        seg = self.segments[self._segment_idx]
        gx, gy = self._segment_goal()
        dist = math.hypot(x - gx, y - gy)
        speed = abs(float(vx))
        if dist <= seg["goal_tolerance"] and speed < seg["stop_speed_threshold"]:
            self._subgoal_in_range = True
            if self._subgoal_hold_start is None:
                self._subgoal_hold_start = self.data.time
            self._subgoal_hold_ok = (
                self.data.time - self._subgoal_hold_start >= seg["stop_duration"]
            )
        else:
            self._subgoal_in_range = False
            self._subgoal_hold_start = None
            self._subgoal_hold_ok = False
        return bool(self._subgoal_hold_ok)

    def _switch_to_next_segment(self) -> None:
        self._segment_idx = min(1, self._segment_idx + 1)
        self._subgoal_hold_start = None
        self._subgoal_hold_ok = False
        if self._segment_idx >= 1:
            self._subgoal_reached_a = True
            self._passed_subgoal = True

    def _update_subgoals(self) -> bool:
        """检查子目标；A 到达自动切段，B 到达返回 True（任务完成）。"""
        if not self.multi_segment:
            return False
        x = float(self.data.body("base_link").xpos[0])
        y = float(self.data.body("base_link").xpos[1])
        vx, _ = self._body_vel()
        seg = self.segments[self._segment_idx]
        gx, gy = self._segment_goal()
        self._subgoal_in_range = (
            math.hypot(x - gx, y - gy) <= seg["goal_tolerance"]
        )
        if self._segment_idx == 1:
            # B 点“到达”即终止（当步视为精确停止完成，同步发 +100）
            if self._subgoal_in_range:
                self._docking_done = True
                self._goal_reached = True
                return True
            return False
        if self._check_subgoal_reached(x, y, vx):
            self._docking_done = True
            self._subgoal_reached_a = True
            self._passed_subgoal = True
            if self.scope == 0:
                self._goal_reached = True
                return True
            self._switch_to_next_segment()
            return True
        return False

    # ------------------------------------------------------------------
    # Junction（岔路口）
    # ------------------------------------------------------------------
    def _junction_center_y(self, x: float, branch: str | None = None) -> float:
        cfg = self.junction_config
        branch = branch or self.branch_selected
        if x <= cfg["decision_x_end"]:
            return float(
                cfg["common_amplitude"]
                * math.sin(math.pi * x / cfg["decision_x_end"])
            )
        if branch is None:
            return 0.0
        sign = 1.0 if branch == "left" else -1.0
        xb = max(x, cfg["decision_x_end"])
        gx = cfg["goal_x"]
        gy = cfg["goal_y"]
        if xb >= gx:
            return sign * gy
        return sign * gy * math.sin(
            math.pi * (xb - cfg["decision_x_end"]) / (gx - cfg["decision_x_end"])
        )

    def _junction_tangent_angle(
        self, x: float, branch: str | None = None
    ) -> float:
        cfg = self.junction_config
        branch = branch or self.branch_selected
        if x <= cfg["decision_x_end"]:
            slope = (
                cfg["common_amplitude"]
                * math.pi
                / cfg["decision_x_end"]
                * math.cos(math.pi * x / cfg["decision_x_end"])
            )
            return float(math.atan(slope))
        if branch is None:
            return 0.0
        sign = 1.0 if branch == "left" else -1.0
        xb = max(x, cfg["decision_x_end"])
        gx = cfg["goal_x"]
        gy = cfg["goal_y"]
        if xb >= gx:
            return 0.0
        slope = sign * gy * math.pi / (gx - cfg["decision_x_end"]) * math.cos(
            math.pi * (xb - cfg["decision_x_end"]) / (gx - cfg["decision_x_end"])
        )
        return float(math.atan(slope))

    def _junction_deviation(
        self, x: float, y: float, branch: str | None = None
    ) -> float:
        return float(y - self._junction_center_y(x, branch))

    def _junction_goal(self) -> tuple[float, float]:
        gy = self.junction_config["goal_y"]
        if self.target_goal == "A":
            return self.junction_config["goal_x"], gy
        return self.junction_config["goal_x"], -gy

    def get_target_goal(self) -> str:
        return self.target_goal

    def get_branch_selected(self) -> str | None:
        return self.branch_selected

    def get_distance_to_junction(self) -> float:
        x = float(self.data.body("base_link").xpos[0])
        return max(0.0, self.junction_config["decision_x_end"] - x)

    def get_distance_to_goal(self) -> float:
        x = float(self.data.body("base_link").xpos[0])
        y = float(self.data.body("base_link").xpos[1])
        gx, gy = self._junction_goal()
        return float(math.hypot(x - gx, y - gy))

    def get_system2_decision_flag(self) -> float:
        return 1.0 if self.branch_selected is not None else 0.0

    def set_branch_selected(self, branch: str) -> None:
        if branch in ("left", "right") and self.branch_selected is None:
            self.branch_selected = branch

    def _junction_reached(self) -> bool:
        x = float(self.data.body("base_link").xpos[0])
        return x >= self.junction_config["decision_x_end"]

    def _goal_arc_len(self) -> float:
        """当前振幅下终点处进度（弯道=弧长，其余=GOAL_X）。"""
        if self.junction:
            return 5.5
        if self.multi_segment:
            return float(self.segments[self._segment_idx]["x1"])
        if self.scenario == "curve":
            if self._goal_arc is not None:
                return self._goal_arc
            return curve_progress(GOAL_X, amp=self.curve_amplitude)
        return GOAL_X

    def get_goal_arc(self) -> float:
        """返回当前 episode 的目标弧长。"""
        return float(self._goal_arc if self._goal_arc is not None else self._goal_arc_len())

    def get_remaining_arc(self) -> float:
        """返回剩余弧长：goal_arc - 当前进度。"""
        x = float(self.data.body("base_link").xpos[0])
        return max(0.0, self.get_goal_arc() - self._progress_metric(x))

    def get_target_heading(self) -> float:
        """返回目标点（goal_arc 对应位置）的切线角。"""
        goal_x = x_for_arc(self.get_goal_arc(), amp=self.curve_amplitude)
        return float(path_tangent_angle(goal_x, is_curve=True, amp=self.curve_amplitude))

    def _curve_deviation(self, x: float, y: float) -> float:
        """弯道场景相对中心线的横向偏差。"""
        if self.multi_segment:
            return self._segment_deviation(x, y)
        return float(y - curve_center_y(x, amp=self.curve_amplitude))

    def _front_clearance(self) -> float:
        z = min(
            self.data.body("FL_wheel_link").xpos[2],
            self.data.body("FR_wheel_link").xpos[2],
        )
        return float(z - WHEEL_RADIUS)

    def _check_termination(self) -> bool:
        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            return True
        if self.task in TRAVERSE_TASKS and self.scenario == "curve":
            x = float(self.data.body("base_link").xpos[0])
            y = float(self.data.body("base_link").xpos[1])
            if self.junction:
                cfg = self.junction_config
                width = (
                    cfg["corridor_common"]
                    if x <= cfg["decision_x_end"]
                    else cfg["corridor_branch"]
                )
                if 0.0 <= x <= 5.5 and abs(self._junction_deviation(x, y)) > width:
                    return True
            elif self.multi_segment:
                seg = self.segments[self._segment_idx]
                if (
                    seg["x0"] - 0.2 <= x <= seg["x1"] + 0.2
                    and abs(self._segment_deviation(x, y)) > seg["corridor_width"]
                ):
                    return True
            else:
                if CURVE_X0 <= x <= CURVE_X1 and abs(self._curve_deviation(x, y)) > self.corridor_width:
                    return True
        if self.task in TRAVERSE_TASKS:
            if float(self.data.body("base_link").xpos[0]) < BACKWARD_LIMIT:
                return True
        pitch = pitch_of(self.data)
        if abs(pitch - self._pitch_ref(self._stage)) > 0.4:
            return True
        if self.data.body("base_link").xpos[2] < 0.10:
            return True
        if self._stage == 2 and self._front_clearance() < 0.0:
            return True
        return False

    def _get_obs(self) -> np.ndarray:
        pitch = pitch_of(self.data)
        pitch_rate = float(self.data.qvel[4])
        quat = self.data.xquat[self.base_body_id].copy()
        angvel = self.data.qvel[3:6].copy()
        base_z = float(self.data.body("base_link").xpos[2])
        clearance = self._front_clearance()
        wheel_vel = np.array(
            [self.data.qvel[di] for di in self.obs_wheel_dofs], dtype=np.float64
        )
        leg_qpos = self.data.qpos[self.leg_qpos_idx].copy()
        leg_qvel = self.data.qvel[self.leg_qvel_idx].copy()

        onehot = np.zeros(4, dtype=np.float64)
        if self._stage in (2,):
            onehot[2] = 1.0
        elif self._stage in (0, 4):
            onehot[0] = 1.0
        elif self._stage == 1:
            onehot[1] = 1.0
        elif self._stage == 3:
            onehot[3] = 1.0
        progress = np.array([self._phi / 2.0], dtype=np.float64)

        if self.domain_randomize:
            pitch += float(self.np_random.normal(0.0, 0.01))
            pitch_rate += float(self.np_random.normal(0.0, 0.05))
            angvel += self.np_random.normal(0.0, 0.05, size=3)
            wheel_vel += self.np_random.normal(0.0, 0.05, size=wheel_vel.shape)

        if self.task not in TRAVERSE_TASKS:
            obs = np.concatenate(
                [
                    [pitch, pitch_rate],
                    quat,
                    angvel,
                    [base_z, clearance],
                    wheel_vel,
                    leg_qpos,
                    leg_qvel,
                    onehot,
                    progress,
                    self._last_action,
                ]
            )
            return obs.astype(np.float32)

        vx, vy = self._body_vel()
        yaw = self._yaw_of()
        x = float(self.data.body("base_link").xpos[0])
        y = float(self.data.body("base_link").xpos[1])
        is_curve = self.scenario == "curve"
        herr = heading_error(x, yaw, is_curve=is_curve, amp=self.curve_amplitude)
        p = self._progress_metric(x)
        progress_rate = (p - self._prev_progress) / POLICY_DT
        if self.multi_segment:
            tan = self._segment_tangent_angle(x)
            herr = wrap_angle(yaw - tan)
            nav = [
                vx, vy,
                math.sin(yaw), math.cos(yaw),
                math.sin(herr), math.cos(herr),
                progress_rate,
                self._segment_deviation(x, y),
                self._goal_arc_len() - x,
            ]
        elif self.scenario == "curve":
            nav = [
                vx, vy,
                math.sin(yaw), math.cos(yaw),
                math.sin(herr), math.cos(herr),
                progress_rate,
                self._curve_deviation(x, y),
                self._goal_arc_len() - p,
            ]
        else:
            nav = [
                vx, vy,
                math.sin(yaw), math.cos(yaw),
                math.sin(herr), math.cos(herr),
                progress_rate,
                y,
                GOAL_X - x,
            ]
        obs = np.concatenate(
            [
                [pitch, pitch_rate],
                quat,
                angvel,
                [base_z, clearance],
                wheel_vel,
                leg_qpos,
                leg_qvel,
                onehot,
                progress,
                self._leg_cmd,
                self._last_action,
                nav,
            ]
        )
        return obs.astype(np.float32)

    def _get_reward(self, action: np.ndarray, terminated: bool, truncated: bool) -> float:
        pitch = pitch_of(self.data)
        pitch_rate = float(self.data.qvel[4])
        err = pitch - self._pitch_ref(self._stage)
        # traverse 生存分改为按进度增量给分（在 traverse 分支内计算），
        # 这里先置 0，避免“站着也给分”；balance 保持 0.1（站住是目标）
        survival = (
            0.0
            if self.task in TRAVERSE_TASKS
            else (0.1 if self.task == "balance" else 1.0)
        )
        r = (
            survival
            - 2.0 * err * err
            - 0.2 * pitch_rate * pitch_rate
            - 0.01 * float(np.dot(action, action))
            - 0.02 * float(np.dot(action - self._last_action, action - self._last_action))
        )

        if self.task == "balance":
            if terminated and not truncated:
                r -= 20.0
        elif self.task == "full_chain":
            r += 2.0 * max(0.0, self._phi - self._prev_phi)
            clearance = self._front_clearance()
            if self._stage == 2 and clearance > 0.10 and abs(err) < 0.1:
                self._stable_seconds += 0.01
                if not self._stable_bonus_given and self._stable_seconds >= 2.0:
                    r += 10.0
                    self._stable_bonus_given = True
            else:
                self._stable_seconds = 0.0
            if (
                self._stage == 4
                and self._phi >= 2.0
                and abs(pitch) < 0.05
                and not self._recovery_bonus_given
            ):
                r += 30.0
                self._recovery_bonus_given = True
            if terminated and not truncated:
                r -= 20.0
        elif self.task == "full_chain_simple":
            clearance = self._front_clearance()
            if self._stage == 2 and clearance > 0.10 and abs(err) < 0.1:
                self._stable_seconds += 0.01
                if not self._stable_bonus_given and self._stable_seconds >= 2.0:
                    r += 10.0
                    self._stable_bonus_given = True
                if not self._simple_success and self._stable_seconds >= HOLD_MIN:
                    r += 30.0
                    self._simple_success = True
            else:
                self._stable_seconds = 0.0
            if terminated and not truncated and not self._simple_success:
                r -= 20.0
        elif self.task in TRAVERSE_TASKS:
            x = float(self.data.body("base_link").xpos[0])
            y = float(self.data.body("base_link").xpos[1])
            vx, _ = self._body_vel()
            p = self._progress_metric(x)
            dx = p - self._prev_progress
            dx_x = x - self._prev_x  # 世界 x 位移（米），简洁奖励用
            herr = heading_error(
                x, self._yaw_of(), is_curve=self.scenario == "curve",
                amp=self.curve_amplitude,
            )
            if self.reward_version == "simple":
                # 课程学习简洁奖励：目标+100 / 摔倒-20 / 前进+0.1*dx / 航向-1.5*herr²
                r += 0.1 * max(0.0, dx_x)
                r -= 1.5 * herr * herr * 0.01
                if self._goal_reached:
                    r += TRAVERSE_GOAL_BONUS
                elif terminated and not truncated and not self._early_stopped:
                    r -= 20.0
            elif self.reward_version == "v5":
                # v5：摔倒/早停 -5，前进 +1.0*dx（进度增量），
                # 势能 shaping γ(Φ(p')-Φ(p))，traverse 无生存分。
                r += 1.0 * max(0.0, dx)
                r += traverse_shaping(
                    p, self._prev_progress, goal_x=self._goal_arc_len()
                )
                r += traverse_backward_penalty(dx)
                r += traverse_speed_penalty(vx)
                r -= self.heading_penalty * herr * herr * 0.01
                if self.scenario == "curve":
                    dev = self._curve_deviation(x, y)
                    r -= TRAVERSE_LATERAL_PENALTY * (
                        dev / self.corridor_width
                    ) ** 2 * 0.01
                if self._goal_reached:
                    r += self.goal_bonus
                elif self._early_stopped or (terminated and not truncated):
                    r += self.fall_penalty
            elif self.reward_version == "mseg":
                seg = self.segments[self._segment_idx]
                gx, gy = self._segment_goal()
                dist = math.hypot(x - gx, y - gy)
                prev = self._prev_dist_to_subgoal
                r += 0.1 * max(0.0, prev - dist)
                yaw = self._yaw_of()
                target_ang = math.atan2(gy - y, gx - x)
                herr_sub = wrap_angle(yaw - target_ang)
                r -= 1.5 * herr_sub * herr_sub * 0.01
                dev = self._segment_deviation(x, y)
                r -= 2.0 * (dev / seg["corridor_width"]) ** 2 * 0.01
                if self._subgoal_in_range and not self._subgoal_bonus_given:
                    r += 50.0
                    self._subgoal_bonus_given = True
                if self._docking_done and not self._docking_bonus_given:
                    r += 100.0
                    self._docking_bonus_given = True
                if self._passed_subgoal:
                    r -= 1.0 * max(0.0, x - gx)
                if (
                    self._passed_subgoal
                    and self.scope == 2
                    and not self._start_at_subgoal
                    and not self._switch_bonus_given
                ):
                    r += 10.0
                    self._switch_bonus_given = True
                if (
                    self._goal_reached
                    and self._segment_idx == 1
                    and not self._final_bonus_given
                ):
                    r += 100.0
                    self._final_bonus_given = True
                if not self._goal_reached:
                    if self._early_stopped:
                        r -= 5.0
                    elif terminated and not truncated:
                        r -= 20.0
                self._prev_dist_to_subgoal = dist
            elif self.reward_version == "junction":
                cfg = self.junction_config
                x = float(self.data.body("base_link").xpos[0])
                y = float(self.data.body("base_link").xpos[1])
                r += 0.1 * max(0.0, x - self._prev_junction_x)
                branch = self.branch_selected
                tan = self._junction_tangent_angle(x, branch)
                herr = wrap_angle(self._yaw_of() - tan)
                r -= 1.5 * herr * herr * 0.01
                width = (
                    cfg["corridor_common"]
                    if x <= cfg["decision_x_end"]
                    else cfg["corridor_branch"]
                )
                dev = self._junction_deviation(x, y, branch)
                r -= 2.0 * (dev / width) ** 2 * 0.01
                if self._junction_reached() and not self._junction_reached_bonus_given:
                    r += 20.0
                    self._junction_reached_bonus_given = True
                if branch is not None and not self._junction_decision_bonus_given:
                    correct = (branch == "left") == (self.target_goal == "A")
                    if correct:
                        r += 30.0
                    else:
                        r -= 30.0
                        self._junction_wrong_branch_given = True
                    self._junction_decision_bonus_given = True
                if (
                    self._goal_reached
                    and self.scope > 0
                    and not self._junction_goal_bonus_given
                ):
                    r += 100.0
                    self._junction_goal_bonus_given = True
                if not self._goal_reached:
                    if self._early_stopped:
                        r -= 5.0
                    elif terminated and not truncated:
                        r -= 20.0
                self._prev_junction_x = x
            else:
                # v4（默认）：进度生存分 + shaping90 + 早停-20 + 航向 + 横向
                r += 0.1 * max(0.0, dx * 100.0)
                r += traverse_shaping(
                    p, self._prev_progress, goal_x=self._goal_arc_len()
                )
                r += traverse_backward_penalty(dx)
                r += traverse_speed_penalty(vx)
                r -= self.heading_penalty * herr * herr * 0.01
                if self.scenario == "curve":
                    dev = self._curve_deviation(x, y)
                    r -= TRAVERSE_LATERAL_PENALTY * (
                        dev / self.corridor_width
                    ) ** 2 * 0.01
                if self._goal_reached:
                    r += TRAVERSE_GOAL_BONUS
                elif self._early_stopped:
                    r -= 20.0
                elif terminated and not truncated:
                    r -= 20.0
            self._prev_progress = p
            self._prev_x = x

        self._prev_phi = self._phi
        return float(r)

    def _info(self) -> dict:
        pitch = pitch_of(self.data)
        vx, _ = self._body_vel()
        x = float(self.data.body("base_link").xpos[0])
        return {
            "pitch": float(pitch),
            "pitch_ref": float(self._pitch_ref(self._stage)),
            "pitch_err": float(pitch - self._pitch_ref(self._stage)),
            "pitch_rate": float(self.data.qvel[4]),
            "front_clearance": self._front_clearance(),
            "base_z": float(self.data.body("base_link").xpos[2]),
            "stage": self._stage,
            "segment": self._segment_idx if self.multi_segment else None,
            "subgoal_a_reached": bool(self._subgoal_reached_a),
            "passed_subgoal": bool(self._passed_subgoal) if self.multi_segment else None,
            "docking_timer": (
                max(0.0, self.data.time - self._subgoal_hold_start)
                if self._subgoal_hold_start is not None
                else 0.0
            ) if self.multi_segment else None,
            "docking_done": bool(self._docking_done) if self.multi_segment else None,
            "scope": self.scope if self.multi_segment else None,
            "junction": bool(self.junction),
            "target_goal": self.target_goal if self.junction else None,
            "branch_selected": self.branch_selected if self.junction else None,
            "system2_flag": (
                self.get_system2_decision_flag() if self.junction else None
            ),
            "phi": self._phi,
            "scenario": self.scenario,
            "x": x,
            "progress": self._progress_metric(x),
            "y": float(self.data.body("base_link").xpos[1]),
            "vx": vx,
            "max_x": float(getattr(self, "_max_x", 0.0)),
            "early_stopped": bool(self._early_stopped),
            "goal": bool(self._goal_reached),
            "reward": 0.0,
            "nan": bool(np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any()),
        }
