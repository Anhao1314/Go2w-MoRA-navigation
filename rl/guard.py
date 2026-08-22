"""双流水线训练守护：崩溃自动续训 + 完成后自动报告 + 自动 git 触发。

流水线：
  phase12: balance seed0/1/2 -> full_chain seed0/1/2
  phase3 : traverse seed0/1/2
每条流水线同时最多运行一个 seed；两条流水线可并行。

用法：
  python rl/guard.py --dry-run    # 打印调度计划，不启动任何进程
  python rl/guard.py --once       # 单次巡检
  python rl/guard.py              # 守护循环（systemd 使用）
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
import shutil
import subprocess
import sys
import time

import psutil

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

RUNS_DIR = PROJECT_ROOT / "rl" / "runs"
GUARD_DIR = RUNS_DIR / "_guard"
STATE_PATH = GUARD_DIR / "state.json"
LOG_DIR = GUARD_DIR / "logs"
TARGET = 8_000_000
ENVS_PER_TASK = 4
BALANCE_CURRICULUM_STEPS = 2_000_000
CHECK_INTERVAL = 10
MAX_ATTEMPTS = 3
RETRY_DELAY = 60
RESUME_MAX_AGE_HOURS = 24
RESUME_MIN_STEPS = 50_000


@dataclasses.dataclass
class Run:
    task: str
    seed: int

    @property
    def key(self) -> str:
        return f"{self.task}/seed{self.seed:02d}"

    @property
    def seed_dir(self) -> pathlib.Path:
        return RUNS_DIR / self.task / f"seed{self.seed:02d}"

    @property
    def best_model(self) -> pathlib.Path:
        return self.seed_dir / "best_model.zip"

    @property
    def best_norm(self) -> pathlib.Path:
        return self.seed_dir / "best_vec_normalize.pkl"

    @property
    def final_model(self) -> pathlib.Path:
        return self.seed_dir / "final_model.zip"

    @property
    def eval_log(self) -> pathlib.Path:
        return self.seed_dir / "eval_log.csv"

    @property
    def completed_marker(self) -> pathlib.Path:
        return self.seed_dir / ".completed"


PIPELINES: dict[str, list[Run]] = {
    "phase12": [
        Run("balance", 0),
        Run("balance", 1),
        Run("balance", 2),
        Run("full_chain", 0),
        Run("full_chain", 1),
        Run("full_chain", 2),
    ],
    "phase3a": [
        Run("traverse_flat_slope", 0),
        Run("traverse_flat_slope", 1),
        Run("traverse_flat_slope", 2),
    ],
    "phase3b": [
        Run("traverse_slope", 0),
        Run("traverse_slope", 1),
        Run("traverse_slope", 2),
    ],
    "phase3c": [
        Run("traverse_curve", 0),
        Run("traverse_curve", 1),
        Run("traverse_curve", 2),
    ],
}

RERUNS: dict[str, int] = {}

# run 状态中的 "external" 表示该 seed 由外部机器（如云服务器）负责训练。
# 本机守护不会启动它，但会在 rsync 回传 final_model.zip + .completed 后，
# 通过 tick() 里的 is_completed 检测自动标记完成并生成报告。


def last_eval_steps(run: Run) -> float:
    try:
        lines = [line.strip() for line in run.eval_log.open(encoding="utf-8") if line.strip()]
        if len(lines) < 2:
            return 0.0
        return float(lines[-1].split(",")[0])
    except (OSError, ValueError, IndexError):
        return 0.0


def is_completed(run: Run) -> bool:
    return run.final_model.exists() and run.completed_marker.exists()


def resume_candidate(run: Run) -> bool:
    """仅当 best 是近期真实断点时才续训，避免旧冒烟产物污染。"""
    if not run.best_model.exists():
        return False
    age_hours = (time.time() - run.best_model.stat().st_mtime) / 3600.0
    return age_hours <= RESUME_MAX_AGE_HOURS and last_eval_steps(run) >= RESUME_MIN_STEPS


def build_command(run: Run, fresh: bool = False) -> list[str]:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "rl" / "train.py"),
        "--task", run.task,
        "--total-steps", str(TARGET),
        "--seed", str(run.seed),
        "--envs", str(ENVS_PER_TASK),
    ]
    if run.task == "balance":
        cmd += ["--curriculum-steps", str(BALANCE_CURRICULUM_STEPS)]
    if fresh:
        cmd += ["--fresh"]
        return cmd
    if run.task == "full_chain":
        balance_best = RUNS_DIR / "balance" / f"seed{run.seed:02d}" / "best_model.zip"
        if resume_candidate(run):
            cmd += ["--resume-from", str(run.best_model), "--resume-norm", str(run.best_norm)]
        elif balance_best.exists():
            cmd += ["--init-from", str(balance_best)]
    elif resume_candidate(run):
        cmd += ["--resume-from", str(run.best_model), "--resume-norm", str(run.best_norm)]
    return cmd


def next_pending(state: dict, runs: list[Run]) -> Run | None:
    """返回流水线里第一个待启动/待重试的任务。"""
    for run in runs:
        if state["runs"][run.key].get("status") in ("pending", "restarting"):
            return run
    return None


def should_retry(attempts: int) -> bool:
    return attempts <= MAX_ATTEMPTS


def default_state() -> dict:
    state = {
        "pipeline": {"phase12": None, "phase3a": None, "phase3b": None, "phase3c": None},
        "runs": {
            run.key: {
                "status": "pending",
                "attempts": 0,
                "pid": None,
                "started_at": None,
                "next_retry_at": None,
                "last_error": None,
            }
            for pipeline in PIPELINES.values()
            for run in pipeline
        },
    }
    for key, rec in state["runs"].items():
        rec["reruns_left"] = RERUNS.get(key, 0)
        rec["fresh"] = False
    return state


def load_state() -> dict:
    if STATE_PATH.exists():
        try:
            data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
            for key, rec in data.get("runs", {}).items():
                rec.setdefault("reruns_left", RERUNS.get(key, 0))
                rec.setdefault("fresh", False)
            return data
        except (OSError, json.JSONDecodeError):
            pass
    return default_state()


def save_state(state: dict) -> None:
    GUARD_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def running_processes() -> dict[str, list[int]]:
    """扫描正在运行的 (task,seed) -> pid 列表。"""
    found: dict[str, list[int]] = {}
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = proc.info.get("cmdline") or []
        except (psutil.AccessDenied, psutil.ZombieProcess):
            continue
        if not any("rl/train.py" in c for c in cmd):
            continue
        task = None
        seed = None
        for i, token in enumerate(cmd):
            if token == "--task" and i + 1 < len(cmd):
                task = cmd[i + 1]
            elif token == "--seed" and i + 1 < len(cmd):
                seed = cmd[i + 1]
        if task and seed is not None:
            key = f"{task}/seed{int(seed):02d}"
            found.setdefault(key, []).append(proc.info["pid"])
    return found


def launch_run(run: Run, state: dict) -> bool:
    rec = state["runs"][run.key]
    GUARD_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{run.key.replace('/', '_')}.log"
    cmd = build_command(run, fresh=bool(rec.get("fresh")))
    rec["attempts"] = int(rec.get("attempts", 0)) + 1
    try:
        with open(log_path, "a", encoding="utf-8") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=str(PROJECT_ROOT),
                start_new_session=True,
            )
    except OSError as exc:
        rec["status"] = "failed"
        rec["last_error"] = str(exc)
        return False
    rec["status"] = "running"
    rec["pid"] = proc.pid
    rec["started_at"] = time.time()
    rec["next_retry_at"] = None
    return True


def run_report(run: Run) -> None:
    try:
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "rl" / "report.py"),
                "--task", run.task,
                "--seed", str(run.seed),
            ],
            cwd=str(PROJECT_ROOT),
            timeout=900,
            capture_output=True,
        )
    except (OSError, subprocess.TimeoutExpired):
        pass
    try:
        (PROJECT_ROOT / "rl" / ".autosync.trigger").touch()
    except OSError:
        pass


def notify(title: str, msg: str) -> None:
    if shutil.which("notify-send"):
        try:
            subprocess.run(["notify-send", "-u", "normal", title, msg], timeout=5)
        except Exception:
            pass
    print(f"[告警] {title}: {msg}")


def tick(state: dict, dry_run: bool = False) -> None:
    running = running_processes()
    for run in (r for pipeline in PIPELINES.values() for r in pipeline):
        rec = state["runs"][run.key]
        if is_completed(run) and rec.get("status") != "completed":
            if int(rec.get("reruns_left", 0)) > 0:
                rec["reruns_left"] = int(rec["reruns_left"]) - 1
                rec["status"] = "pending"
                rec["attempts"] = 0
                rec["pid"] = None
                rec["fresh"] = True
                rec["last_error"] = None
                try:
                    run.completed_marker.unlink(missing_ok=True)
                except OSError:
                    pass
                for pipe_name, pipe_runs in PIPELINES.items():
                    if any(r.key == run.key for r in pipe_runs):
                        state["pipeline"][pipe_name] = None
                if not dry_run:
                    notify("RL 训练守护", f"{run.key} 完成，准备从零重跑（碰撞修复）")
            else:
                rec["status"] = "completed"
                rec["pid"] = None
                if not dry_run:
                    run_report(run)

    for pipe_name, runs in PIPELINES.items():
        current_key = state["pipeline"].get(pipe_name)
        current = None
        if current_key:
            current = next((r for r in runs if r.key == current_key), None)

        # 接管已在运行的同流水线进程，避免重复启动
        if current is None:
            for run in runs:
                if running.get(run.key) and state["runs"][run.key].get("status") not in (
                    "completed", "failed",
                ):
                    rec = state["runs"][run.key]
                    rec["status"] = "running"
                    rec["pid"] = min(running[run.key])
                    state["pipeline"][pipe_name] = run.key
                    current = run
                    break

        if current is not None:
            rec = state["runs"][current.key]
            if rec.get("status") == "completed":
                state["pipeline"][pipe_name] = None
                current = None
            elif running.get(current.key):
                rec["pid"] = min(running[current.key])
                continue  # 正常运行
            else:
                # 进程消失
                if is_completed(current):
                    rec["status"] = "completed"
                    state["pipeline"][pipe_name] = None
                    if not dry_run:
                        run_report(current)
                    continue
                if rec.get("status") == "restarting":
                    if time.time() < float(rec.get("next_retry_at", 0)):
                        continue  # 还在退避等待
                    if int(rec.get("attempts", 0)) >= MAX_ATTEMPTS:
                        rec["status"] = "failed"
                        rec["last_error"] = f"崩溃超过 {MAX_ATTEMPTS} 次"
                        state["pipeline"][pipe_name] = None
                        if not dry_run:
                            notify("RL 训练守护", f"{current.key} 训练失败，已停止重试")
                        continue
                    if not dry_run:
                        if launch_run(current, state):
                            continue
                        state["pipeline"][pipe_name] = None
                    continue
                rec["attempts"] = int(rec.get("attempts", 0)) + 1
                rec["pid"] = None
                if rec["attempts"] > MAX_ATTEMPTS:
                    rec["status"] = "failed"
                    rec["last_error"] = f"崩溃超过 {MAX_ATTEMPTS} 次"
                    state["pipeline"][pipe_name] = None
                    if not dry_run:
                        notify("RL 训练守护", f"{current.key} 训练失败，已停止重试")
                    continue
                rec["status"] = "restarting"
                rec["next_retry_at"] = time.time() + RETRY_DELAY
                if not dry_run:
                    notify("RL 训练守护", f"{current.key} 进程退出，{RETRY_DELAY}s 后自动续训")
                continue

        # 当前无运行：处理 restarting 或启动下一个 pending
        if current is None and current_key:
            rec = state["runs"][current_key]
            if rec.get("status") == "restarting":
                restart_run = next((r for r in runs if r.key == current_key), None)
                if dry_run or time.time() >= float(rec.get("next_retry_at", 0)):
                    if dry_run:
                        print(f"[dry-run] 将续训 {current_key}")
                    elif restart_run is not None:
                        launch_run(restart_run, state)
                continue

        # 启动下一个 pending
        for run in runs:
            rec = state["runs"][run.key]
            if rec.get("status") in ("completed", "failed"):
                continue
            if rec.get("status") == "pending" or rec.get("status") == "restarting":
                if dry_run:
                    if running.get(run.key):
                        print(f"[dry-run] {run.key} 已在运行，跳过")
                    else:
                        print(f"[dry-run] 将启动 {run.key}: {' '.join(build_command(run))}")
                        state["pipeline"][pipe_name] = run.key
                else:
                    if launch_run(run, state):
                        state["pipeline"][pipe_name] = run.key
                break


def main() -> None:
    parser = argparse.ArgumentParser(description="RL 双流水线训练守护")
    parser.add_argument("--dry-run", action="store_true", help="只打印调度计划")
    parser.add_argument("--once", action="store_true", help="只巡检一次")
    args = parser.parse_args()

    state = load_state()
    if args.dry_run:
        running = running_processes()
        print("当前运行:", running)
        for pipe_name, runs in PIPELINES.items():
            print(f"[{pipe_name}]")
            for run in runs:
                status = state["runs"][run.key].get("status")
                if status == "external":
                    owner = state["runs"][run.key].get("owner", "cloud")
                    print(f"  {run.key}: external（由 {owner} 负责，本机跳过）")
                    continue
                done = "完成" if is_completed(run) else "待跑"
                print(f"  {run.key}: {status} ({done})")
        tick(state, dry_run=True)
        return

    if args.once:
        tick(state)
        save_state(state)
        return

    while True:
        tick(state)
        save_state(state)
        time.sleep(CHECK_INTERVAL)


if __name__ == "__main__":
    main()
