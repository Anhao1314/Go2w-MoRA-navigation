"""训练完成后自动生成指标报告与训练曲线（零第三方绘图依赖）。

用法：
  python rl/report.py --task balance --seed 00
  python rl/report.py --task traverse --seed 00
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import time

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.go2w_env import TRAVERSE_TASKS  # noqa: E402

RUNS_DIR = PROJECT_ROOT / "rl" / "runs"
REPORTS_DIR = PROJECT_ROOT / "reports"

THRESHOLDS = {
    "balance": {"max_dev": 0.1, "min_clear": 0.10},
    "full_chain": {"max_dev": 0.1, "min_clear": 0.10, "settle_seconds": 2.0},
}
for _task in TRAVERSE_TASKS:
    THRESHOLDS[_task] = {"success_rate": 0.6, "overall_success_rate": 0.8}


def metrics_csv_path(task: str, seed: int) -> pathlib.Path:
    return REPORTS_DIR / task / f"seed{seed:02d}" / "metrics.csv"


def run_eval(task: str, seed: int, episodes: int) -> pathlib.Path:
    out = metrics_csv_path(task, seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    model = RUNS_DIR / task / f"seed{seed:02d}" / "best_model.zip"
    norm = RUNS_DIR / task / f"seed{seed:02d}" / "best_vec_normalize.pkl"
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "rl" / "eval.py"),
        "--task", task,
        "--model", str(model),
        "--vec-norm", str(norm),
        "--episodes", str(episodes),
        "--report", str(out),
    ]
    if task in TRAVERSE_TASKS:
        cmd += ["--scenario", "all"]
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, timeout=1800)
    return out


def read_metrics(path: pathlib.Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def build_verdict(task: str, rows: list[dict[str, str]]) -> tuple[bool, list[str]]:
    checks: list[str] = []
    ok = True

    def add(name: str, passed: bool, detail: str) -> None:
        nonlocal ok
        ok = ok and passed
        checks.append(f"{'✅' if passed else '❌'} {name}: {detail}")

    if task == "balance":
        row = rows[0] if rows else {}
        add("最大俯仰偏差", float(row.get("max_dev", 9)) <= THRESHOLDS["balance"]["max_dev"],
            f"{row.get('max_dev', 'N/A')} ≤ {THRESHOLDS['balance']['max_dev']}")
        add("最小前轮离地", float(row.get("min_clear", -9)) >= THRESHOLDS["balance"]["min_clear"],
            f"{row.get('min_clear', 'N/A')} ≥ {THRESHOLDS['balance']['min_clear']}")
    elif task == "full_chain":
        row = rows[0] if rows else {}
        add("四轮恢复成功", row.get("recovered") == "True", str(row.get("recovered")))
        add("恢复后稳定时长", float(row.get("settle_seconds", -9)) >= 2.0,
            f"{row.get('settle_seconds', 'N/A')} ≥ 2.0s")
        add("双轮保持", float(row.get("dual_hold", -9)) >= 5.0,
            f"{row.get('dual_hold', 'N/A')} ≥ 5.0s")
    else:  # traverse 三任务
        rates = [float(r.get("success_rate", 0)) for r in rows if r.get("success_rate")]
        overall = sum(rates) / len(rates) if rates else 0.0
        add("整体成功率", overall >= THRESHOLDS[task]["overall_success_rate"],
            f"{overall:.0%} ≥ {THRESHOLDS[task]['overall_success_rate']:.0%}")
        for r in rows:
            label = r.get("label", "?")
            rate = float(r.get("success_rate", 0))
            add(f"场景成功率 {label}", rate >= THRESHOLDS[task]["success_rate"],
                f"{rate:.0%} ≥ {THRESHOLDS[task]['success_rate']:.0%}")
    return ok, checks


def read_curve(task: str, seed: int) -> list[tuple[float, float]]:
    path = RUNS_DIR / task / f"seed{seed:02d}" / "eval_log.csv"
    points: list[tuple[float, float]] = []
    if not path.exists():
        return points
    with open(path, encoding="utf-8") as f:
        for line in f.readlines()[1:]:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                try:
                    points.append((float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
    return points


def make_curve_svg(points: list[tuple[float, float]], title: str) -> str:
    width, height, pad = 800, 240, 40
    if not points:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">' \
               f'<text x="{pad}" y="120">暂无数据</text></svg>'
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1

    def px(x: float) -> float:
        return pad + (x - x0) / (x1 - x0) * (width - 2 * pad)

    def py(y: float) -> float:
        return height - pad - (y - y0) / (y1 - y0) * (height - 2 * pad)

    line = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in points)
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
        f'<rect width="{width}" height="{height}" fill="#111" />'
        f'<text x="{pad}" y="20" fill="#eee" font-size="14">{title}</text>'
        f'<polyline points="{line}" fill="none" stroke="#4fc3f7" stroke-width="2" />'
        f'<text x="{pad}" y="{height - 10}" fill="#999" font-size="11">{x0/1e6:.1f}M</text>'
        f'<text x="{width - 2*pad - 40}" y="{height - 10}" fill="#999" font-size="11">{x1/1e6:.1f}M</text>'
        f'</svg>'
    )


def make_lines_svg(
    series: list[tuple[str, list[tuple[float, float]], str]],
    title: str,
    y_min: float | None = None,
    y_max: float | None = None,
) -> str:
    """多系列折线图，series 元素为 (图例标签, [(x, y), ...], 颜色)。"""
    width, height, pad = 800, 240, 40
    all_points = [p for _, pts, _ in series for p in pts]
    if not all_points:
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
            f'<text x="{pad}" y="120" fill="#999">暂无数据</text></svg>'
        )
    xs = [p[0] for p in all_points]
    ys = [p[1] for p in all_points]
    x0, x1 = min(xs), max(xs)
    y0 = y_min if y_min is not None else min(ys)
    y1 = y_max if y_max is not None else max(ys)
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1

    def px(x: float) -> float:
        return pad + (x - x0) / (x1 - x0) * (width - 2 * pad)

    def py(y: float) -> float:
        return height - pad - (y - y0) / (y1 - y0) * (height - 2 * pad)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#111" />',
        f'<text x="{pad}" y="20" fill="#eee" font-size="14">{title}</text>',
    ]
    for label, pts, color in series:
        line = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in pts)
        parts.append(
            f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2" />'
        )
    legend_x = pad
    for label, _pts, color in series:
        parts.append(
            f'<text x="{legend_x:.0f}" y="{height - 10}" fill="{color}" '
            f'font-size="11">{label}</text>'
        )
        legend_x += len(label) * 12 + 24
    parts.append("</svg>")
    return "".join(parts)


def make_bar_svg(labels: list[str], values: list[float], title: str) -> str:
    width, pad = 640, 40
    row_h = 40
    height = max(140, pad + len(labels) * row_h + 20)
    max_value = max(values + [1.0])
    bar_max = width - 2 * pad - 140
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#111" />',
        f'<text x="{pad}" y="20" fill="#eee" font-size="14">{title}</text>',
    ]
    for i, (label, value) in enumerate(zip(labels, values)):
        y = pad + i * row_h
        bar_w = max(2.0, value / max_value * bar_max)
        parts.append(f'<rect x="{pad}" y="{y}" width="{bar_w:.1f}" height="18" fill="#4fc3f7" />')
        parts.append(f'<text x="{pad + bar_w + 6}" y="{y + 14}" fill="#d7dae0" font-size="12">{value:.0%}</text>')
        parts.append(f'<text x="{pad}" y="{y - 6}" fill="#8a93a5" font-size="11">{label}</text>')
    parts.append("</svg>")
    return "".join(parts)


def write_report(task: str, seed: int, ok: bool, checks: list[str],
                 rows: list[dict[str, str]]) -> pathlib.Path:
    out_dir = REPORTS_DIR / task / f"seed{seed:02d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        f"# {task} seed{seed:02d} 训练报告",
        "",
        f"生成时间：{time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"**结论：{'✅ 通过' if ok else '❌ 未通过'}**",
        "",
        "## 验收检查",
        "",
    ]
    lines += [f"- {c}" for c in checks]
    lines += ["", "## 指标", "", "| 指标 | 值 |", "|---|---|"]
    if rows:
        keys = list(rows[0].keys())
        for key in keys:
            vals = " / ".join(r.get(key, "") for r in rows)
            lines.append(f"| {key} | {vals} |")
    else:
        lines.append("| 无指标数据 | — |")
    lines += [
        "",
        "## 训练曲线",
        "",
        "![curve](curve.svg)",
        "",
    ]
    path = out_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="训练完成自动报告")
    parser.add_argument(
        "--task",
        choices=["balance", "full_chain", *TRAVERSE_TASKS],
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--skip-eval", action="store_true",
                        help="跳过评估，使用已有 metrics.csv")
    args = parser.parse_args()

    if not args.skip_eval:
        run_eval(args.task, args.seed, args.episodes)
    rows = read_metrics(metrics_csv_path(args.task, args.seed))
    ok, checks = build_verdict(args.task, rows)
    report_path = write_report(args.task, args.seed, ok, checks, rows)
    svg = make_curve_svg(
        read_curve(args.task, args.seed),
        f"{args.task} seed{args.seed:02d} 训练曲线",
    )
    (report_path.parent / "curve.svg").write_text(svg, encoding="utf-8")
    if args.task in TRAVERSE_TASKS and rows:
        scenarios = [
            {
                "label": row.get("label", ""),
                "success_rate": float(row.get("success_rate", 0)),
                "backward_dist": float(row.get("backward_dist", 0) or 0),
            }
            for row in rows
        ]
        bar_svg = make_bar_svg(
            [s["label"] for s in scenarios],
            [s["success_rate"] for s in scenarios],
            "场景成功率",
        )
        (report_path.parent / "scenarios.svg").write_text(bar_svg, encoding="utf-8")
        summary = {
            "task": args.task,
            "seed": args.seed,
            "verdict": "pass" if ok else "fail",
            "checks": checks,
            "scenarios": scenarios,
        }
        (report_path.parent / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with open(report_path, "a", encoding="utf-8") as f:
            f.write("\n## 场景成功率\n\n![场景成功率](scenarios.svg)\n")
    print(f"报告已生成: {report_path}")
    print("结论:", "通过" if ok else "未通过")


if __name__ == "__main__":
    main()
