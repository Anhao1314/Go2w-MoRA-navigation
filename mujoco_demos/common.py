"""共享工具：模型加载、无头录制、GIF/MP4 输出与 GUI 查看器。"""

from __future__ import annotations

import argparse
import pathlib

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np
from PIL import Image


def load_model(path: str | pathlib.Path) -> mujoco.MjModel:
    """从 XML 路径加载 MuJoCo 模型。"""
    return mujoco.MjModel.from_xml_path(str(path))


def record(
    model: mujoco.MjModel,
    duration: float,
    fps: int,
    control_fn,
    *,
    width: int = 640,
    height: int = 480,
    camera: str | None = None,
    init_fn=None,
) -> tuple[list[np.ndarray], mujoco.MjData]:
    """无头运行仿真并逐帧渲染。

    control_fn(data, t) 会在每个仿真步被调用，用于更新 data.ctrl。
    init_fn(data) 在仿真开始前调用，用于设置初始位姿。
    返回 (帧列表, 最终 MjData)。
    """
    data = mujoco.MjData(model)
    if init_fn is not None:
        init_fn(data)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height, width)
    n_frames = int(round(duration * fps))
    steps_per_frame = max(1, round(1.0 / fps / model.opt.timestep))
    frames: list[np.ndarray] = []

    for _ in range(n_frames):
        for _ in range(steps_per_frame):
            control_fn(data, data.time)
            mujoco.mj_step(model, data)
        if camera is None:
            renderer.update_scene(data)
        else:
            renderer.update_scene(data, camera=camera)
        frames.append(renderer.render().copy())

    return frames, data


def run_gui(model: mujoco.MjModel, control_fn, init_fn=None) -> None:
    """在 MuJoCo 原生查看器中实时运行仿真；按 R 复位到初始位姿。"""
    data = mujoco.MjData(model)
    if init_fn is not None:
        init_fn(data)
    mujoco.mj_forward(model, data)

    reset_requested = False

    def on_key(key: int) -> None:
        nonlocal reset_requested
        if key == ord("R"):
            reset_requested = True

    with mujoco.viewer.launch_passive(model, data, key_callback=on_key) as viewer:
        while viewer.is_running():
            if reset_requested:
                mujoco.mj_resetData(model, data)
                if init_fn is not None:
                    init_fn(data)
                reset_requested = False
                print("已复位到初始位置")
            control_fn(data, data.time)
            mujoco.mj_step(model, data)
            viewer.sync()


def save_gif(
    frames: list[np.ndarray],
    path,
    fps: int,
    *,
    max_width: int = 480,
    max_fps: int = 30,
) -> pathlib.Path:
    """把帧序列保存为 GIF（README 内嵌用，自动降采样控制体积）。"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    step = max(1, round(fps / max_fps))
    out_fps = fps / step
    images = []
    for frame in frames[::step]:
        image = Image.fromarray(frame)
        if image.width > max_width:
            height = round(image.height * max_width / image.width)
            image = image.resize((max_width, height), Image.Resampling.LANCZOS)
        images.append(image.quantize(colors=128, method=Image.Quantize.FASTOCTREE))
    images[0].save(
        path,
        save_all=True,
        append_images=images[1:],
        duration=int(round(1000.0 / out_fps)),
        loop=0,
        optimize=True,
    )
    return path


def save_mp4(frames: list[np.ndarray], path, fps: int) -> pathlib.Path:
    """把帧序列保存为 MP4（H.264）。"""
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(path, fps=fps, codec="libx264", quality=8)
    for frame in frames:
        writer.append_data(frame)
    writer.close()
    return path


def add_common_args(parser: argparse.ArgumentParser) -> None:
    """给 Demo 脚本添加统一的命令行参数。"""
    parser.add_argument("--duration", type=float, default=5.0, help="仿真时长（秒）")
    parser.add_argument("--fps", type=int, default=60, help="录制帧率")
    parser.add_argument("--out", type=str, default=None, help="输出文件前缀，默认 media/<demo 名>")
    parser.add_argument("--gui", action="store_true", help="打开实时查看器而非录制")
