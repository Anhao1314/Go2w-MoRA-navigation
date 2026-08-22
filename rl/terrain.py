"""运行时高度场地形生成（Perlin 高度场思路，BSD-3-Clause）。

思路移植自 westonrobot/unitree_mujoco 的 terrain_tool/terrain_generator.py
（AddPerlinHeighField：以 Perlin 噪声生成高度场）。本项目不做 PNG 文件交换，
而是逐 episode 在 numpy 中生成归一化高度矩阵，直接写入 MuJoCo 的
``model.hfield_data``（运行时数据必须自行归一化到 [0,1]）。

License: BSD 3-Clause License (详见 docs/THIRD_PARTY_NOTICES.md)
"""

from __future__ import annotations

import math

import numpy as np

# 高度场世界覆盖范围（与 models/go2w/go2w_scenario_scene.xml 的
# <hfield name="scn_terrain"> 保持一致）：
#   世界 x ∈ [TERRAIN_X_MIN, TERRAIN_X_MAX]，y ∈ [-TERRAIN_Y_HALF, TERRAIN_Y_HALF]
TERRAIN_X_MIN = 0.0
TERRAIN_X_MAX = 5.0
TERRAIN_Y_HALF = 1.25
TERRAIN_CX = (TERRAIN_X_MIN + TERRAIN_X_MAX) / 2.0  # 2.5，hfield geom 的 x 位置

# 高度场参数（与 XML 中 nrow/ncol 一致）
TERRAIN_NROW = 64
TERRAIN_NCOL = 128

# 坡面参数（与 go2w_env 中的 course 几何保持一致）
TERRAIN_X0 = 1.0       # 上坡起点
TERRAIN_SLOPE_LEN = 3.0
TERRAIN_GOAL_X = 4.5

# 凸起参数：3~6 个高斯凸包 + 小幅 Perlin 型噪声
BUMP_COUNT = (3, 7)
BUMP_AMP = (0.03, 0.06)
BUMP_SIGMA = (0.25, 0.5)
BUMP_X_RANGE = (1.2, 4.4)
BUMP_Y_RANGE = (-0.8, 0.8)
NOISE_AMP = 0.02
NOISE_COMPONENTS = 3
NOISE_FREQ = (0.8, 2.0)  # 周期/m，对应波长 0.5~1.25m

# 初始平地段：x < 0.8 严格为 0，0.8~1.05 平滑淡入凸起/噪声
FLAT_FADE_START = 0.8
FLAT_FADE_END = 1.05


def _smoothstep(t: np.ndarray) -> np.ndarray:
    """0/1 平滑过渡（3t^2 - 2t^3）。"""
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def make_terrain(
    angle_deg: float,
    x0: float = TERRAIN_X0,
    slope_len: float = TERRAIN_SLOPE_LEN,
    goal_x: float = TERRAIN_GOAL_X,
    width_half: float = TERRAIN_Y_HALF,
    nrow: int = TERRAIN_NROW,
    ncol: int = TERRAIN_NCOL,
    rng: np.random.Generator | None = None,
    bump_amp_max: float = 0.06,
) -> np.ndarray:
    """生成归一化 [0,1] 的高度场矩阵，形状 (nrow, ncol)。

    高度 = 基准坡面（x<x0 平地 -> x0~x0+slope_len 线性上坡 -> 之后平台）
    + 3~6 个高斯凸包（y 方向左右镜像，规避 MuJoCo 行序翻转歧义）
    + 小幅 Perlin 型噪声（波长 >= 0.5m，满足高度场接触稳定性）。

    列索引 j 对应世界 x，行索引 i 对应世界 y；x < 0.8m 严格为 0。
    返回数据可直接写入 ``model.hfield_data``（MuJoCo 运行时数据需归一化）。
    """
    rng = rng if rng is not None else np.random.default_rng()
    if not np.isfinite(angle_deg) or not (0.0 <= angle_deg < 90.0):
        raise ValueError(f"非法坡角: {angle_deg}")
    if nrow <= 1 or ncol <= 1:
        raise ValueError("高度场至少需要 2x2 网格")
    if bump_amp_max <= 0.0:
        raise ValueError("bump_amp_max 必须为正")

    xs = np.linspace(TERRAIN_X_MIN, TERRAIN_X_MAX, ncol)  # 世界 x
    ys = np.linspace(-width_half, width_half, nrow)       # 世界 y
    xx, yy = np.meshgrid(xs, ys)  # shape (nrow, ncol)

    # 基准坡面：x < x0 -> 0，x0~x0+len 线性上升，之后保持最高点
    rise = slope_len * math.tan(math.radians(angle_deg))
    t = np.clip((xx - x0) / slope_len, 0.0, 1.0)
    base = rise * t

    # 高斯凸包（y 镜像保证左右对称）
    bumps = np.zeros_like(xx)
    for _ in range(int(rng.integers(*BUMP_COUNT))):
        cx = float(rng.uniform(*BUMP_X_RANGE))
        cy = float(rng.uniform(*BUMP_Y_RANGE))
        amp = float(rng.uniform(BUMP_AMP[0], min(bump_amp_max, BUMP_AMP[1])))
        sigma = float(rng.uniform(*BUMP_SIGMA))
        d = (xx - cx) ** 2 + (yy - cy) ** 2
        bumps += amp * np.exp(-d / (2.0 * sigma * sigma))
        if abs(cy) > 1e-6:
            d_m = (xx - cx) ** 2 + (yy + cy) ** 2
            bumps += amp * np.exp(-d_m / (2.0 * sigma * sigma))

    # 小幅 Perlin 型噪声：多分量正弦乘积
    noise = np.zeros_like(xx)
    for _ in range(NOISE_COMPONENTS):
        fx = float(rng.uniform(*NOISE_FREQ))
        fy = float(rng.uniform(*NOISE_FREQ))
        px = float(rng.uniform(0.0, 2.0 * math.pi))
        noise += (
            np.sin(2.0 * math.pi * fx * xx + px)
            * np.cos(2.0 * math.pi * fy * yy)  # cos 为偶函数，保证 y 左右对称
        )
    noise *= NOISE_AMP / NOISE_COMPONENTS

    # 初始平地段平滑淡入：x < 0.8 严格为 0，保证起始位姿干净
    fade = _smoothstep((xs - FLAT_FADE_START) / (FLAT_FADE_END - FLAT_FADE_START))
    z = base + fade[None, :] * (bumps + noise)
    z[:, xs < FLAT_FADE_START] = 0.0
    z = np.maximum(z, 0.0)

    # 归一化：除以“坡顶高度 + 最大凸起 + 余量”，并夹到 [0,1]
    z_max = rise + bump_amp_max + NOISE_AMP + 0.03
    z_norm = np.clip(z / z_max, 0.0, 1.0)
    if not np.isfinite(z_norm).all():
        raise ValueError("地形生成出现非有限值")
    return z_norm


def write_heightfield(
    model, hfield_id: int, z: np.ndarray
) -> None:
    """把归一化高度矩阵写入 MuJoCo 模型的 hfield_data 对应切片。"""
    z = np.asarray(z, dtype=np.float64)
    adr = int(model.hfield_adr[hfield_id])
    n = int(model.hfield_nrow[hfield_id] * model.hfield_ncol[hfield_id])
    if z.size != n:
        raise ValueError(
            f"高度数据尺寸 {z.size} 与 hfield {hfield_id} 容量 {n} 不一致"
        )
    if not np.isfinite(z).all():
        raise ValueError("高度数据包含非有限值")
    model.hfield_data[adr : adr + n] = z.reshape(-1)
