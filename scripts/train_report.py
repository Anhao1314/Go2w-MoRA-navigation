#!/usr/bin/env python3
"""Generate a formal training report for one run directory.

Reads three sources and writes one Markdown report under ``reports/training/``:

1. ``run_config.json`` -- git SHA, dirty flag, dependency versions, command line
2. TensorBoard scalars -- reward terms, terminations, losses, throughput
3. ``evaluation.json`` -- frozen-seed evaluation, when it exists

The report is deliberately audit-shaped: every number names its source, and
an automatic anomaly section flags the failure modes this project has actually
hit (reward not rising, a reward term staying exactly zero, episodes running to
the time limit, evaluation success at zero, non-finite values).

Usage::

    python scripts/train_report.py --run-dir logs/rsl_rl/go2_loco_sequence/<stamp>
    python scripts/train_report.py --run-dir <dir> --out reports/training/<name>.md

Exit codes: 0 = written, 1 = no run directory / no scalars.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import statistics
import sys
import time
from typing import Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_OUT_DIR = PROJECT_ROOT / "reports" / "training"

# scalars worth showing in the process table, in report order
PROCESS_TAGS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Metrics/base_velocity/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw",
    "Episode_Termination/time_out",
    "Episode_Termination/base_contact",
    "Policy/mean_noise_std",
    "Loss/value_function",
    "Loss/entropy",
    "Perf/total_fps",
)


def load_scalars(run_dir: pathlib.Path) -> dict[str, list[tuple[int, float]]]:
    """Read every scalar tag from the run's TensorBoard events."""
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except Exception as exc:  # pragma: no cover - depends on environment
        raise SystemExit(f"tensorboard is required to read run scalars: {exc}")

    accumulator = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    series: dict[str, list[tuple[int, float]]] = {}
    for tag in accumulator.Tags().get("scalars", []):
        series[tag] = [(event.step, float(event.value)) for event in accumulator.Scalars(tag)]
    return series


def _trend(values: list[float]) -> str:
    """Classify a scalar series as rising, falling or flat over its second half."""
    if len(values) < 10:
        return "too short"
    half = len(values) // 2
    early = statistics.fmean(values[:half])
    late = statistics.fmean(values[half:])
    scale = max(abs(early), abs(late), 1e-9)
    change = (late - early) / scale
    if change > 0.10:
        return "rising"
    if change < -0.10:
        return "falling"
    return "flat"


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "first": values[0],
        "last": values[-1],
        "min": min(values),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def detect_anomalies(
    series: dict[str, list[tuple[int, float]]], evaluation: dict[str, Any] | None
) -> list[str]:
    """Flag failure modes this project has actually encountered."""
    alarms: list[str] = []

    reward = [value for _, value in series.get("Train/mean_reward", [])]
    if reward:
        if reward[-1] < reward[0]:
            alarms.append(
                f"Mean reward did not rise: first {reward[0]:.3f} -> last {reward[-1]:.3f}."
            )
        if _trend(reward) == "flat":
            alarms.append("Reward is flat over the second half of training.")

    length = [value for _, value in series.get("Train/mean_episode_length", [])]
    if length:
        cap = max(length)
        if length[-1] >= 0.95 * cap and cap > 100:
            alarms.append(
                f"Mean episode length reached {length[-1]:.0f} steps (cap {cap:.0f}). "
                "For a sustained task (velocity tracking) this is expected; for a staged "
                "task it means the stages were not completed."
            )

    # terms that legitimately stay zero in a healthy run (limits never hit)
    zero_is_fine = {"Episode_Reward/dof_pos_limits"}
    for tag, rows in series.items():
        if not tag.startswith("Episode_Reward/") or tag in zero_is_fine:
            continue
        values = [value for _, value in rows]
        if values and max(abs(v) for v in values) < 1e-12:
            alarms.append(f"Reward term `{tag}` stayed exactly zero for the whole run.")

    for tag, rows in series.items():
        for step, value in rows:
            if value != value or value in (float("inf"), float("-inf")):
                alarms.append(f"Non-finite value in `{tag}` at step {step}; series truncated below.")
                break

    if evaluation is not None:
        summary = evaluation.get("summary", {})
        if "success_rate" in summary and summary["success_rate"] == 0.0:
            alarms.append("Evaluation success_rate is 0.0: no episode completed the task.")
        if summary.get("fall_rate") == 1.0:
            alarms.append("Evaluation fall_rate is 1.0: every episode ended in a fall.")

    return alarms


def build_report(run_dir: pathlib.Path, series: dict[str, list[tuple[int, float]]]) -> str:
    config_path = run_dir / "run_config.json"
    config: dict[str, Any] = {}
    if config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    eval_path = run_dir / "evaluation.json"
    evaluation: dict[str, Any] | None = None
    if eval_path.is_file():
        evaluation = json.loads(eval_path.read_text(encoding="utf-8"))

    git = config.get("git", {}) if config else {}
    deps = config.get("dependencies", {}) if config else {}
    argv = config.get("argv", []) if config else []

    lines: list[str] = []
    add = lines.append

    add(f"# Training report: `{run_dir.name}`")
    add("")
    add(f"- Generated: {time.strftime('%Y-%m-%dT%H:%M:%S%z')}")
    add(f"- Run directory: `{run_dir.as_posix()}`")
    add(f"- Status: {'evidence bundle finalized' if config else 'NO run_config.json (unfinished or interrupted)'}")
    add("")

    add("## 1. Run identity")
    add("")
    add("| Field | Value |")
    add("| --- | --- |")
    add(f"| Task | `{config.get('task', 'unknown')}` |")
    add(f"| Seed | {config.get('seed', 'unknown')} |")
    add(f"| Git commit | `{str(git.get('commit', 'unknown'))[:12]}` |")
    add(f"| Branch | `{git.get('branch', 'unknown')}` |")
    add(f"| Working tree dirty | {git.get('dirty', 'unknown')} |")
    add(f"| Started | {config.get('started_at', 'unknown')} |")
    add(f"| Command line | `{' '.join(str(a) for a in argv)}` |")
    add("")
    add("Dependencies of record:")
    add("")
    add("| Package | Version |")
    add("| --- | --- |")
    for name, version in deps.items():
        add(f"| {name} | {version} |")
    add("")

    add("## 2. Training process")
    add("")
    add("Source: TensorBoard scalars in the run directory.")
    add("")
    add("| Metric | First | Last | Min | Max | Trend |")
    add("| --- | --- | --- | --- | --- | --- |")
    for tag in PROCESS_TAGS:
        rows = series.get(tag)
        if not rows:
            continue
        values = [value for _, value in rows]
        stats = _summary(values)
        add(
            f"| `{tag}` | {stats['first']:.4g} | {stats['last']:.4g} | "
            f"{stats['min']:.4g} | {stats['max']:.4g} | {_trend(values)} |"
        )
    add("")

    reward_tags = sorted(tag for tag in series if tag.startswith("Episode_Reward/"))
    if reward_tags:
        add("Reward terms (first -> last):")
        add("")
        add("| Term | First | Last |")
        add("| --- | --- | --- |")
        for tag in reward_tags:
            values = [value for _, value in series[tag]]
            add(f"| `{tag.removeprefix('Episode_Reward/')}` | {values[0]:.4g} | {values[-1]:.4g} |")
        add("")

    add("## 3. Evaluation")
    add("")
    if evaluation is None:
        add("No `evaluation.json` in this run directory: the run has not been evaluated on the")
        add("frozen seed set yet. This is not a result, and must not be reported as one.")
    else:
        summary = evaluation.get("summary", {})
        add("| Metric | Value |")
        add("| --- | --- |")
        for key in (
            "episodes",
            "success_rate",
            "vx_rmse_mean",
            "vx_rmse_std",
            "wz_rmse_mean",
            "wz_rmse_std",
            "fall_rate",
            "mean_steps_to_finish",
            "method",
            "wall_time_s",
        ):
            if key in summary:
                add(f"| `{key}` | {summary[key]} |")
        add("")
        stage_rate = summary.get("stage_reach_rate")
        if stage_rate:
            add("Stage reach rate (fraction of episodes reaching at least this stage):")
            add("")
            add("| Stage | Rate |")
            add("| --- | --- |")
            for stage, rate in stage_rate.items():
                add(f"| {stage} | {rate:.3f} |")
            add("")

    add("## 4. Anomalies and warnings")
    add("")
    alarms = detect_anomalies(series, evaluation)
    if alarms:
        for alarm in alarms:
            add(f"- ⚠️ {alarm}")
    else:
        add("None detected by the automated checks.")
    add("")

    add("## 5. Evidence")
    add("")
    for name in ("run_config.json", "sha256_manifest.json", "evaluation.json", "params/agent.yaml", "params/env.yaml"):
        path = run_dir / name
        add(f"- `{name}`: {'present' if path.is_file() else 'missing'}")
    add("")

    add("## 6. Reading this report")
    add("")
    add("- Numbers above come from the run's own logs; none are recomputed by hand.")
    add("- A run without `evaluation.json` has **no measured task performance** -- do not quote")
    add("  training reward as a result.")
    add("- Anomalies are rules, not judgements: each flagged line names the exact evidence.")
    add("")
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--out", type=pathlib.Path, default=None)
    args = parser.parse_args(argv)

    run_dir = args.run_dir
    if not run_dir.is_dir():
        print(f"[report] run directory not found: {run_dir}")
        return 1

    series = load_scalars(run_dir)
    if not series:
        print(f"[report] no TensorBoard scalars under {run_dir}")
        return 1

    text = build_report(run_dir, series)
    out = args.out or (DEFAULT_OUT_DIR / f"{run_dir.name}.md")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    out.with_suffix(".md.sha256").write_text(f"{digest}  {out.name}\n", encoding="ascii")

    print(f"[report] wrote {out} ({len(series)} scalar tags, sha256 {digest[:12]})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
