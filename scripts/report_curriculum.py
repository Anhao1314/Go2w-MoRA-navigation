"""traverse_curve 课程学习（直道→大弯→中弯→目标弯）验收报告生成。

读取各阶段 eval_log，用 stage4 best 在目标环境（0.4m / 0.35）跑 20 episode
正式验收（每 episode 全新环境，防 MuJoCo warmstart 假成功），输出：
  - reports/traverse_curve_curriculum/seed00/{report.md,metrics.csv,summary.json}
  - data/demo_trajectories/curve_curriculum_report.md

用法：python scripts/report_curriculum.py [--episodes 20] [--skip-eval]
"""

from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time
from typing import Any

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.go2w_env import Go2wEnv  # noqa: E402
from rl.eval import aggregate, load_policy, run_rl_episode, summarize  # noqa: E402

CURRICULUM_ROOT = (
    PROJECT_ROOT / "rl" / "runs" / "traverse_curve_curriculum" / "seed00"
)
REPORT_DIR = PROJECT_ROOT / "reports" / "traverse_curve_curriculum" / "seed00"
DEMO_REPORT = PROJECT_ROOT / "data" / "demo_trajectories" / "curve_curriculum_report.md"

STAGES = [
    {"name": "stage1_straight", "corridor": 1.0, "amp": 0.0, "threshold": 0.8},
    {"name": "stage2_big_curve", "corridor": 0.8, "amp": 0.15, "threshold": 0.4},
    {"name": "stage3_mid_curve", "corridor": 0.6, "amp": 0.25, "threshold": 0.5},
    {"name": "stage4_target", "corridor": 0.4, "amp": 0.35, "threshold": 0.6},
]

METRICS_HEADER = [
    "label", "task", "max_dev", "min_clear", "dual_hold", "recovered",
    "settle_seconds", "success", "success_rate", "distance", "time_to_goal",
    "falls", "total_reward", "mean_base_reward", "nan",
]


def read_eval_csv(path: pathlib.Path) -> list[dict[str, float]]:
    """读取 eval_log.csv，返回数值行列表；文件缺失/损坏返回空列表。"""
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    try:
        with open(path, encoding="utf-8", errors="replace", newline="") as f:
            for rec in csv.DictReader(f):
                try:
                    rows.append(
                        {
                            "timesteps": float(rec["timesteps"]),
                            "mean_reward": float(rec["mean_reward"]),
                            "std_reward": float(rec["std_reward"]),
                            "mean_ep_len": float(rec["mean_ep_len"]),
                        }
                    )
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return []
    return rows


def stage_summaries(
    root: pathlib.Path = CURRICULUM_ROOT,
    stages: list[dict] | None = None,
) -> list[dict[str, Any]]:
    """汇总各阶段：评估点数、末次奖励/ep_len、模型存在性。"""
    stages = stages or STAGES
    out: list[dict[str, Any]] = []
    for spec in stages:
        stage_dir = root / spec["name"]
        rows = read_eval_csv(stage_dir / "eval_log.csv")
        last = rows[-1] if rows else None
        out.append(
            {
                "name": spec["name"],
                "corridor": spec["corridor"],
                "amp": spec["amp"],
                "threshold": spec["threshold"],
                "eval_points": len(rows),
                "last_reward": last["mean_reward"] if last else None,
                "last_ep_len": last["mean_ep_len"] if last else None,
                "best_model": (stage_dir / "best_model.zip").exists(),
                "final_model": (stage_dir / "final_model.zip").exists(),
                "completed": (
                    (stage_dir / "final_model.zip").exists()
                    and (stage_dir / ".completed").exists()
                ),
            }
        )
    return out


def run_acceptance(
    model_path: pathlib.Path,
    norm_path: pathlib.Path,
    episodes: int = 20,
) -> dict[str, Any]:
    """目标环境 20 episode 正式验收（每 episode 全新 Go2wEnv）。"""
    model, _, vec_norm = load_policy(str(model_path), str(norm_path), "traverse_curve")
    summaries = []
    for ep in range(episodes):
        env = Go2wEnv(task="traverse_curve", domain_randomize=False)
        log, _ = run_rl_episode(model, env, vec_norm)
        summaries.append(summarize(log, "traverse_curve"))
        env.close()
    return aggregate(summaries)


def metrics_row(agg: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": "RL PPO (curve)",
        "task": agg.get("task", "traverse_curve"),
        "max_dev": agg.get("max_dev"),
        "min_clear": agg.get("min_clear"),
        "dual_hold": agg.get("dual_hold"),
        "recovered": agg.get("recovered"),
        "settle_seconds": agg.get("settle_seconds"),
        "success": agg.get("success"),
        "success_rate": agg.get("success_rate"),
        "distance": agg.get("distance"),
        "time_to_goal": agg.get("time_to_goal"),
        "falls": agg.get("falls"),
        "total_reward": agg.get("total_reward"),
        "mean_base_reward": agg.get("mean_base_reward"),
        "nan": agg.get("nan"),
    }


def build_report_text(
    stages: list[dict[str, Any]],
    agg: dict[str, Any],
    generated: str,
) -> str:
    sr = float(agg.get("success_rate") or 0.0)
    ok = sr >= 0.6
    stage_lines = [
        "| 阶段 | 走廊 | 振幅 | 阈值 | 评估点数 | 末次奖励 | 末次ep_len | 完成 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in stages:
        reward_txt = "--" if s["last_reward"] is None else f"{s['last_reward']:.2f}"
        ep_txt = "--" if s["last_ep_len"] is None else f"{s['last_ep_len']:.0f}"
        stage_lines.append(
            f"| {s['name']} | {s['corridor']} | {s['amp']} | {s['threshold']} "
            f"| {s['eval_points']} | "
            f"{reward_txt} | {ep_txt} | "
            f"{'✅' if s['completed'] else '❌'} |"
        )
    return "\n".join(
        [
            "# traverse_curve 课程学习训练报告",
            "",
            f"生成时间：{generated}",
            "",
            f"**结论：{'✅ 通过' if ok else '❌ 未通过'}**",
            "",
            "## 验收检查（目标环境 0.4m / 0.35）",
            "",
            f"- {'✅' if ok else '❌'} 场景成功率: {sr * 100:.0f}% ≥ 60%",
            "",
            "## 指标",
            "",
            "| 指标 | 值 |",
            "|---|---|",
            f"| success_rate | {sr} |",
            f"| distance | {agg.get('distance')} |",
            f"| time_to_goal | {agg.get('time_to_goal')} |",
            f"| falls | {agg.get('falls')} |",
            f"| total_reward | {agg.get('total_reward')} |",
            f"| mean_base_reward | {agg.get('mean_base_reward')} |",
            f"| max_dev | {agg.get('max_dev')} |",
            f"| nan | {agg.get('nan')} |",
            "",
            "## 各阶段结果",
            "",
            "\n".join(stage_lines),
            "",
            "## 失败根因",
            "",
            "最终 best 在目标环境表现为“站桩”：ep_len≈301（3 秒早停），"
            "max_progress=0.00m；阶段 1 直道同样走不出 0.00m。"
            "simple 奖励的前进分量（0.1×dx）远小于动作/姿态惩罚，"
            "站桩成为局部最优；纯 PPO 从零未能发现稳定前进动作。",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="生成课程学习验收报告")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="跳过验收，使用已有 metrics.csv（用于幂等重跑）",
    )
    args = parser.parse_args()

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_path = REPORT_DIR / "metrics.csv"
    if args.skip_eval and metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        agg: dict[str, Any] = {
            "task": "traverse_curve",
            "success_rate": float(rows[0]["success_rate"]),
            "success": rows[0]["success"] == "True",
            "distance": rows[0]["distance"],
            "time_to_goal": rows[0]["time_to_goal"],
            "falls": rows[0]["falls"],
            "total_reward": rows[0]["total_reward"],
            "mean_base_reward": rows[0]["mean_base_reward"],
            "max_dev": rows[0]["max_dev"],
            "nan": rows[0]["nan"],
        }
    else:
        model = CURRICULUM_ROOT / "stage4_target_best.zip"
        norm = CURRICULUM_ROOT / "stage4_target_best_vec_normalize.pkl"
        if not model.exists() or not norm.exists():
            raise SystemExit(f"缺少 stage4 模型/norm: {model} / {norm}")
        agg = run_acceptance(model, norm, args.episodes)

    row = metrics_row(agg)
    with open(metrics_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_HEADER)
        writer.writeheader()
        writer.writerow(row)

    verdict = "pass" if float(agg.get("success_rate") or 0.0) >= 0.6 else "fail"
    (REPORT_DIR / "summary.json").write_text(
        json.dumps(
            {
                "task": "traverse_curve_curriculum",
                "seed": 0,
                "verdict": verdict,
                "success_rate": agg.get("success_rate"),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    generated = time.strftime("%Y-%m-%d %H:%M:%S")
    stages = stage_summaries()
    text = build_report_text(stages, agg, generated)
    (REPORT_DIR / "report.md").write_text(text, encoding="utf-8")
    DEMO_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEMO_REPORT.write_text(text, encoding="utf-8")
    print(f"验收: success_rate={agg.get('success_rate')} verdict={verdict}")
    print(f"报告: {REPORT_DIR / 'report.md'}")
    print(f"汇总: {DEMO_REPORT}")


if __name__ == "__main__":
    main()
