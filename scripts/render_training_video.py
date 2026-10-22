#!/usr/bin/env python3
"""Render a training-progress MP4 from the three-seed training log.

Pure CPU work: replays the log (no simulator, no GPU) and draws the learning
curves stage by stage.  Use it after or during training to produce a
shareable record of how each seed progressed.

Each frame shows:
- reward curve per seed (partial curve up to the current frame)
- velocity-tracking error curves
- a stage panel with seed, iteration, elapsed/ETA and the latest metrics

Usage::

    python scripts/render_training_video.py \
        --log "C:/Users/Administrator/.zcode/workspace/default/m1_training.log" \
        --out media/training/m1_training_progress.mp4

Exit codes: 0 = video written, 1 = no data / render failure.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import imageio.v2 as imageio  # noqa: E402

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT = PROJECT_ROOT / "media" / "training" / "m1_training_progress.mp4"

ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
SEED_START = re.compile(r"=== SEED (\d+) START")
SEED_DONE = re.compile(r"=== SEED (\d+) DONE rc=(\d+)")
ITERATION = re.compile(r"Learning iteration (\d+)/(\d+)")
MEAN_REWARD = re.compile(r"Mean reward:\s*(-?[\d.]+)")
ERR_XY = re.compile(r"error_vel_xy:\s*(-?[\d.]+)")
ERR_YAW = re.compile(r"error_vel_yaw:\s*(-?[\d.]+)")

SEED_COLORS = ["#4c8dff", "#3ddc97", "#f5a524", "#ff5c5c"]


def parse_log(path: pathlib.Path) -> dict[int, list[dict[str, float]]]:
    """Extract one record per iteration: iteration, reward, err_xy, err_yaw."""
    seeds: dict[int, list[dict[str, float]]] = {}
    current: int | None = None
    pending: dict[str, float] = {}

    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            line = ANSI.sub("", raw).strip()
            if not line:
                continue
            match = SEED_START.search(line)
            if match:
                current = int(match.group(1))
                seeds.setdefault(current, [])
                pending = {}
                continue
            if current is None:
                continue
            match = ITERATION.search(line)
            if match:
                pending["iteration"] = float(match.group(1))
                pending["max_iteration"] = float(match.group(2))
                continue
            match = MEAN_REWARD.search(line)
            if match:
                pending["reward"] = float(match.group(1))
                continue
            match = ERR_XY.search(line)
            if match:
                pending["err_xy"] = float(match.group(1))
                continue
            match = ERR_YAW.search(line)
            if match:
                pending["err_yaw"] = float(match.group(1))
                if all(key in pending for key in ("iteration", "reward", "err_xy", "err_yaw")):
                    seeds[current].append(dict(pending))
                pending = {}
    return {seed: rows for seed, rows in seeds.items() if rows}


def build_frames(data: dict[int, list[dict[str, float]]]) -> tuple[list[tuple[int, int, float]], float]:
    """Flatten per-seed records into (seed, index, elapsed_fraction)."""
    lengths = {seed: len(rows) for seed, rows in data.items()}
    total = sum(lengths.values())
    frames: list[tuple[int, int, float]] = []
    running = 0
    for seed in sorted(data):
        for index in range(lengths[seed]):
            running += 1
            frames.append((seed, index, running / max(1, total)))
    return frames, float(total)


def render(
    data: dict[int, list[dict[str, float]]],
    out: pathlib.Path,
    *,
    fps: int = 20,
    stride: int = 5,
) -> pathlib.Path:
    frames, _total = build_frames(data)
    frames = frames[:: max(1, stride)]

    seeds = sorted(data)
    iterations = {s: np.array([r["iteration"] for r in data[s]]) for s in seeds}
    rewards = {s: np.array([r["reward"] for r in data[s]]) for s in seeds}
    err_xy = {s: np.array([r["err_xy"] for r in data[s]]) for s in seeds}
    err_yaw = {s: np.array([r["err_yaw"] for r in data[s]]) for s in seeds}

    reward_all = np.concatenate([rewards[s] for s in seeds])
    xy_all = np.concatenate([err_xy[s] for s in seeds])
    yaw_all = np.concatenate([err_yaw[s] for s in seeds])
    iter_all = np.concatenate([iterations[s] for s in seeds])

    plt.rcParams.update({"figure.facecolor": "#0f1115", "axes.facecolor": "#171a21"})
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 7.2), dpi=100)
    ax_reward, ax_xy = axes[0, 0], axes[0, 1]
    ax_yaw, ax_info = axes[1, 0], axes[1, 1]

    def style(ax, title: str, ylabel: str) -> None:
        ax.set_facecolor("#171a21")
        ax.set_title(title, color="#e6e9ef", fontsize=12, loc="left")
        ax.set_xlabel("iteration", color="#8b93a7", fontsize=9)
        ax.set_ylabel(ylabel, color="#8b93a7", fontsize=9)
        ax.tick_params(colors="#8b93a7", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#262b36")
        ax.grid(color="#262b36", linewidth=0.6, alpha=0.7)

    style(ax_reward, "Mean reward", "reward")
    style(ax_xy, "Velocity tracking error (xy)", "m/s")
    style(ax_yaw, "Yaw-rate tracking error", "rad/s")
    for ax, values in ((ax_reward, reward_all), (ax_xy, xy_all), (ax_yaw, yaw_all)):
        ax.set_xlim(float(iter_all.min()), float(iter_all.max()) + 1)
        pad = (float(values.max()) - float(values.min())) * 0.08 + 1e-3
        ax.set_ylim(float(values.min()) - pad, float(values.max()) + pad)

    lines = {}
    for offset, seed in enumerate(seeds):
        color = SEED_COLORS[offset % len(SEED_COLORS)]
        lines[seed] = {
            "reward": ax_reward.plot([], [], color=color, lw=1.4, label=f"seed {seed}")[0],
            "xy": ax_xy.plot([], [], color=color, lw=1.4)[0],
            "yaw": ax_yaw.plot([], [], color=color, lw=1.4)[0],
        }
    ax_reward.legend(loc="lower right", fontsize=8, facecolor="#171a21", edgecolor="#262b36",
                     labelcolor="#e6e9ef")

    ax_info.axis("off")
    stage_text = ax_info.text(
        0.02, 0.92, "", transform=ax_info.transAxes, va="top", ha="left",
        fontsize=12, color="#e6e9ef", family="monospace", linespacing=1.7,
    )
    progress_text = ax_info.text(
        0.02, 0.16, "", transform=ax_info.transAxes, va="top", ha="left",
        fontsize=10, color="#8b93a7", family="monospace",
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(
        out, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",
        macro_block_size=1,
    )
    try:
        for seed, index, fraction in frames:
            for other in seeds:
                upto = len(data[other]) if other < seed else (index + 1 if other == seed else 0)
                row = lines[other]
                row["reward"].set_data(iterations[other][:upto], rewards[other][:upto])
                row["xy"].set_data(iterations[other][:upto], err_xy[other][:upto])
                row["yaw"].set_data(iterations[other][:upto], err_yaw[other][:upto])
            current = data[seed][index]
            stage_text.set_text(
                f"stage   : seed {seed}  ({seeds.index(seed) + 1}/{len(seeds)})\n"
                f"iter    : {int(current['iteration'])}/{int(current.get('max_iteration', 1500))}\n"
                f"reward  : {current['reward']:.2f}\n"
                f"err_xy  : {current['err_xy']:.4f} m/s\n"
                f"err_yaw : {current['err_yaw']:.4f} rad/s"
            )
            progress_text.set_text(f"overall progress: {fraction * 100:.1f}%")
            fig.canvas.draw()
            frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
            writer.append_data(frame)
    finally:
        writer.close()
        plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default=DEFAULT_OUT)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--stride", type=int, default=5, help="log records per frame")
    args = parser.parse_args(argv)

    data = parse_log(args.log)
    if not data:
        print(f"[render] no iteration records found in {args.log}")
        return 1
    total = sum(len(rows) for rows in data.values())
    print(f"[render] seeds={sorted(data)} records={total}")
    out = render(data, args.out, fps=args.fps, stride=args.stride)
    size_mb = out.stat().st_size / 1e6
    print(f"[render] wrote {out} ({size_mb:.2f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
