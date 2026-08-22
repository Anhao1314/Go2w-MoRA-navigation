"""traverse_curve 课程学习：直道 → 大弯 → 中弯 → 目标弯。

每阶段最多 1M 步（简单奖励），用上一阶段 best 初始化下一阶段；
阶段结束后 5 episode 评估，达标或到步数上限都进入下一阶段；
最终在目标环境（0.40m/0.35）做 20 episode 验收并输出报告。

用法：python scripts/train_curve_curriculum.py
"""

from __future__ import annotations

import csv
import json
import pathlib
import shutil
import subprocess
import sys
import time

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl.go2w_env import Go2wEnv

OUT = PROJECT_ROOT / "data" / "demo_trajectories"
RUN_ROOT = PROJECT_ROOT / "rl" / "runs" / "traverse_curve_curriculum" / "seed00"
PY = sys.executable

STAGES = [
    {"name": "stage1_straight", "corridor": 1.0, "amp": 0.0, "threshold": 0.8},
    {"name": "stage2_big_curve", "corridor": 0.8, "amp": 0.15, "threshold": 0.4},
    {"name": "stage3_mid_curve", "corridor": 0.6, "amp": 0.25, "threshold": 0.5},
    {"name": "stage4_target", "corridor": 0.4, "amp": 0.35, "threshold": 0.6},
]
ENVS = 4
STAGE_STEPS = 1_000_000
LR = 3e-4


def eval_stage(corridor: float, amp: float, model_path: pathlib.Path,
               norm_path: pathlib.Path, episodes: int = 5) -> dict:
    """用阶段环境评估 5 episode（每 episode 全新环境，防 warmstart 假成功）。"""
    proto = Go2wEnv(task="traverse_curve", domain_randomize=False,
                    corridor_width=corridor, curve_amplitude=amp,
                    reward_version="simple")
    vec = VecNormalize.load(str(norm_path), DummyVecEnv([lambda: proto]))
    vec.training = False
    model = PPO.load(str(model_path), device="cpu")
    suc = 0
    dists: list[float] = []
    lens: list[int] = []
    for ep in range(episodes):
        e = Go2wEnv(task="traverse_curve", domain_randomize=False,
                    corridor_width=corridor, curve_amplitude=amp,
                    reward_version="simple")
        o, info = e.reset(seed=ep)
        mx = 0.0
        steps = 0
        while True:
            on = vec.normalize_obs(o)
            a, _ = model.predict(on, deterministic=True)
            o, _r, term, trunc, info = e.step(a)
            steps += 1
            mx = max(mx, float(info.get("progress", info.get("x", 0.0))))
            if term or trunc:
                break
        lens.append(steps)
        dists.append(mx)
        if info.get("goal"):
            suc += 1
        e.close()
    return {"success": suc, "success_rate": suc / episodes,
            "mean_dist": float(np.mean(dists)), "mean_ep_len": float(np.mean(lens))}


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    prev_best = None
    prev_norm = None

    for idx, stage in enumerate(STAGES):
        name = stage["name"]
        run_dir = RUN_ROOT / name
        print(f"\n===== 阶段 {idx + 1}/{len(STAGES)}: {name} "
              f"(走廊 {stage['corridor']}m, 振幅 {stage['amp']}) =====", flush=True)
        cmd = [
            PY, "rl/train.py", "--task", "traverse_curve", "--seed", "0",
            "--total-steps", str(STAGE_STEPS), "--envs", str(ENVS),
            "--corridor-width", str(stage["corridor"]),
            "--curve-amplitude", str(stage["amp"]),
            "--reward-version", "simple",
            "--learning-rate", str(LR),
            "--run-dir", str(run_dir),
        ]
        if prev_best is not None:
            cmd += ["--init-from", str(prev_best)]
        t0 = time.time()
        subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True)
        print(f"阶段训练完成，耗时 {(time.time()-t0)/60:.1f} 分钟", flush=True)

        model_path = run_dir / "best_model.zip"
        norm_path = run_dir / "best_vec_normalize.pkl"
        if not model_path.exists():
            # 训练期未存 best（可能 ep_len 不达标），用 final 兜底
            model_path = run_dir / "final_model.zip"
            norm_path = run_dir / "final_vec_normalize.pkl"
        stage_best = RUN_ROOT / f"{name}_best.zip"
        stage_norm = RUN_ROOT / f"{name}_best_vec_normalize.pkl"
        shutil.copy2(model_path, stage_best)
        shutil.copy2(norm_path, stage_norm)
        prev_best, prev_norm = stage_best, stage_norm

        ev = eval_stage(stage["corridor"], stage["amp"], stage_best, stage_norm)
        records.append({**stage, **ev})
        ok = ev["success_rate"] >= stage["threshold"]
        print(f"阶段评估: 成功率 {ev['success_rate']:.0%} "
              f"({ev['success']}/5), 平均距离 {ev['mean_dist']:.2f}m, "
              f"ep_len {ev['mean_ep_len']:.0f} → {'达标' if ok else '未达标(强制进入下一阶段)'}",
              flush=True)

    # 最终 20 episode 验收（目标环境）
    print("\n===== 最终验收：目标环境 0.40m / 0.35 =====", flush=True)
    if prev_best is None or prev_norm is None:
        raise SystemExit("缺少上一阶段模型，无法执行最终验收")
    final = eval_stage(0.40, 0.35, prev_best, prev_norm, episodes=20)
    print("最终验收:", final, flush=True)

    report = [
        "# traverse_curve 课程学习报告\n",
        "## 配置",
        f"- 奖励：simple（到达+100 / 摔倒-20 / 前进+0.1×dx / 航向-1.5×herr²）",
        f"- 每阶段 {STAGE_STEPS/1e6:.1f}M 步，{ENVS} envs，lr={LR}",
        "- 阶段间用上一阶段 best 初始化\n",
        "## 各阶段结果\n",
        "| 阶段 | 走廊 | 振幅 | 阈值 | 成功率 | 平均距离 | 平均ep_len | 达标 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in records:
        report.append(
            f"| {r['name']} | {r['corridor']} | {r['amp']} | {r['threshold']} "
            f"| {r['success_rate']:.0%} | {r['mean_dist']:.2f} | {r['mean_ep_len']:.0f} "
            f"| {'✅' if r['success_rate'] >= r['threshold'] else '❌'} |"
        )
    report += [
        "\n## 最终验收（20 episode，目标环境）\n",
        f"- 成功率：{final['success_rate']:.0%}（{final['success']}/20）",
        f"- 平均距离：{final['mean_dist']:.2f} m",
        f"- 平均 ep_len：{final['mean_ep_len']:.0f}\n",
        f"结论：{'✅ 课程学习成功' if final['success_rate'] >= 0.6 else '❌ 未达 60% 验收标准，需分析'}",
    ]
    (OUT / "curve_curriculum_report.md").write_text("\n".join(report), encoding="utf-8")

    config = {
        "stages": [
            {"name": s["name"], "corridor_width": s["corridor"],
             "curve_amplitude": s["amp"], "threshold": s["threshold"]}
            for s in STAGES
        ],
        "stage_steps": STAGE_STEPS, "envs": ENVS, "lr": LR,
        "reward_version": "simple",
        "stage_results": records, "final_eval": final,
        "model_dir": str(RUN_ROOT),
    }
    (OUT / "curriculum_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("\n报告: data/demo_trajectories/curve_curriculum_report.md")
    print("配置: data/demo_trajectories/curriculum_config.json")


if __name__ == "__main__":
    main()
