"""零依赖本地 Web 面板：后台状态快照 + 每 seed 详情 + 实时 SVG 曲线 + 报告渲染。

用法：
  python rl/webpanel.py --port 8787
"""

from __future__ import annotations

import argparse
import html
import json
import os
import pathlib
import re
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import psutil

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rl import report as report_mod
from rl import viewer_launcher
from rl.go2w_env import TRAVERSE_TASKS
from rl.monitor import TASKS, build_snapshot, parse_eval_csv, read_tensorboard
from rl.task_info import TASK_INFO, task_info
from scripts import token_dashboard

WEB_DIR = PROJECT_ROOT / "rl" / "web"
REPORTS_DIR = PROJECT_ROOT / "reports"
MEDIA_DIR = PROJECT_ROOT / "media"
GUARD_STATE_PATH = PROJECT_ROOT / "rl" / "runs" / "_guard" / "state.json"
GUARD_DIR = PROJECT_ROOT / "rl" / "runs" / "_guard"
SNAPSHOT_HISTORY_PATH = GUARD_DIR / "snapshots.jsonl"
CURRICULUM_ROOT = PROJECT_ROOT / "rl" / "runs" / "traverse_curve_curriculum" / "seed00"
CURRICULUM_OUT = PROJECT_ROOT / "data" / "demo_trajectories"
CURRICULUM_REPORT_PATH = CURRICULUM_OUT / "curve_curriculum_report.md"
CURRICULUM_CONFIG_PATH = CURRICULUM_OUT / "curriculum_config.json"
CURRICULUM_STAGE_STEPS = 1_000_000
CURRICULUM_STAGES = [
    {"name": "stage1_straight", "corridor": 1.0, "amp": 0.0, "threshold": 0.8},
    {"name": "stage2_big_curve", "corridor": 0.8, "amp": 0.15, "threshold": 0.4},
    {"name": "stage3_mid_curve", "corridor": 0.6, "amp": 0.25, "threshold": 0.5},
    {"name": "stage4_target", "corridor": 0.4, "amp": 0.35, "threshold": 0.6},
]
INDEX_HTML = WEB_DIR / "index.html"
DETAIL_HTML = WEB_DIR / "detail.html"
REFRESH_INTERVAL = 3.0
HISTORY_INTERVAL = 30.0
HISTORY_CAP = 5000
DEMO_RE = re.compile(r"^rl_(\w+)_(seed\d{2})\.(mp4|gif)$")

_lock = threading.Lock()
_snapshot: dict = {}
_alerts_seen: dict[str, dict] = {}
TOKEN: str = ""
REMOTE: bool = False

LOGIN_HTML = """<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>登录</title><style>
body{background:#0f1117;color:#d7dae0;font-family:Consolas,Menlo,monospace;
display:flex;justify-content:center;align-items:center;height:100vh;margin:0}
form{background:#171a23;border:1px solid #2a2f3a;border-radius:8px;padding:24px}
input{background:#0f1117;color:#d7dae0;border:1px solid #2a2f3a;border-radius:6px;
padding:8px;font-size:15px}
button{background:#4fc3f7;color:#0f1117;border:0;border-radius:6px;padding:8px 16px;
font-size:15px;cursor:pointer;margin-left:8px}
</style></head><body>
<form method="post" action="/login">
<div>请输入访问密码</div><br>
<input name="token" type="password" autofocus>
<button type="submit">进入</button>
</form></body></html>"""


def _authorized(handler: Any) -> bool:
    if not TOKEN:
        return True
    query = parse_qs(urlparse(handler.path).query)
    if query.get("token", [None])[0] == TOKEN:
        return True
    if handler.headers.get("Authorization") == f"Bearer {TOKEN}":
        return True
    cookie = handler.headers.get("Cookie", "")
    if f"go2w_token={TOKEN}" in cookie:
        return True
    return False


def guard_state() -> dict:
    data = {}
    try:
        data = json.loads(GUARD_STATE_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        pass
    active = False
    for proc in psutil.process_iter(["cmdline"]):
        try:
            cmd = proc.info.get("cmdline") or []
        except (psutil.AccessDenied, psutil.ZombieProcess):
            continue
        if any("rl/guard.py" in c for c in cmd):
            active = True
            break
    return {
        "active": active,
        "pipeline": data.get("pipeline", {}),
        "runs": {
            key: {
                "status": rec.get("status"),
                "attempts": rec.get("attempts"),
                "last_error": rec.get("last_error"),
            }
            for key, rec in data.get("runs", {}).items()
        },
    }


def read_curriculum_stages(
    root: pathlib.Path = CURRICULUM_ROOT,
    stages: list[dict] | None = None,
) -> list[dict]:
    """读取课程学习各阶段的 eval_log 状态（轻量，不读 TensorBoard）。"""
    stages = stages or CURRICULUM_STAGES
    out: list[dict] = []
    for spec in stages:
        stage_dir = root / spec["name"]
        rows, _ = parse_eval_csv(stage_dir / "eval_log.csv")
        last = rows[-1] if rows else None
        mtime: float | None = None
        try:
            if stage_dir.exists():
                mtime = stage_dir.stat().st_mtime
                for p in stage_dir.rglob("*"):
                    if p.is_file():
                        try:
                            cur = p.stat().st_mtime
                            mtime = cur if mtime is None else max(mtime, cur)
                        except OSError:
                            pass
        except OSError:
            mtime = None
        out.append(
            {
                "name": spec["name"],
                "corridor": spec["corridor"],
                "amp": spec["amp"],
                "threshold": spec["threshold"],
                "exists": stage_dir.exists(),
                "timesteps": float(last.timesteps) if last else 0.0,
                "reward": float(last.mean_reward) if last else None,
                "ep_len": float(last.ep_len) if last else None,
                "best_model": (stage_dir / "best_model.zip").exists(),
                "mtime": mtime,
            }
        )
    return out


def curriculum_process() -> tuple[bool, int | None, str | None]:
    """检测课程学习主脚本与当前阶段（来自 rl/train.py --run-dir）。"""
    running = False
    runner_pid: int | None = None
    current_stage: str | None = None
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = proc.info.get("cmdline") or []
        except (psutil.AccessDenied, psutil.ZombieProcess):
            continue
        joined = " ".join(cmd)
        if "scripts/train_curve_curriculum.py" in joined:
            running = True
            runner_pid = int(proc.info["pid"])
        if "rl/train.py" in joined and "traverse_curve_curriculum/seed00/" in joined:
            for i, token in enumerate(cmd):
                if token == "--run-dir" and i + 1 < len(cmd):
                    run_dir = pathlib.Path(cmd[i + 1])
                    if run_dir.name.startswith("stage"):
                        current_stage = run_dir.name
                    break
    return running, runner_pid, current_stage


def build_curriculum_status(
    stage_data: list[dict],
    running: bool,
    current_stage: str | None,
    config: dict | None = None,
    report: bool = False,
    stage_steps: int = CURRICULUM_STAGE_STEPS,
    total_target: int | None = None,
    eta_seconds: float | None = None,
) -> dict:
    """纯函数：根据阶段数据与运行状态生成面板展示结构。"""
    if total_target is None:
        total_target = stage_steps * len(stage_data)
    current_idx = next(
        (i for i, st in enumerate(stage_data) if st["name"] == current_stage),
        None,
    )
    for i, st in enumerate(stage_data):
        if running and current_idx is not None:
            if i < current_idx:
                status = "completed" if st["exists"] else "pending"
            elif i == current_idx:
                status = "running"
            else:
                status = "pending"
        elif config is not None:
            status = "completed" if st["exists"] else "pending"
        else:
            status = "stopped" if st["exists"] else "pending"
        st["status"] = status
    total_steps = float(sum(st["timesteps"] for st in stage_data))
    result: dict = {
        "running": running,
        "runner_pid": None,
        "current_stage": current_stage,
        "completed": config is not None,
        "cloud_active": False,
        "stage_steps": stage_steps,
        "stages": stage_data,
        "total_steps": total_steps,
        "total_target": float(total_target),
        "eta_seconds": eta_seconds,
        "report": report,
        "config": config,
    }
    return result


def curriculum_status(
    root: pathlib.Path = CURRICULUM_ROOT,
    out_dir: pathlib.Path = CURRICULUM_OUT,
) -> dict:
    """课程学习实时状态：阶段进度、奖励、当前阶段、ETA、完成配置。"""
    running, runner_pid, current_stage = curriculum_process()
    stages = read_curriculum_stages(root)
    config: dict | None = None
    try:
        config = json.loads(CURRICULUM_CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        config = None
    report = CURRICULUM_REPORT_PATH.exists()
    eta_seconds: float | None = None
    if running and current_stage:
        tb = read_tensorboard(root / current_stage)
        if tb:
            for st in stages:
                if st["name"] == current_stage:
                    st["timesteps"] = max(st["timesteps"], float(tb["timesteps"]))
                    if st["reward"] is None:
                        st["reward"] = tb["reward"]
                    if st["ep_len"] is None:
                        st["ep_len"] = tb["ep_len"]
            speed = tb.get("speed")
            if speed and speed > 0:
                total_target = CURRICULUM_STAGE_STEPS * len(CURRICULUM_STAGES)
                total_steps = float(sum(st["timesteps"] for st in stages))
                eta_seconds = max(0.0, (total_target - total_steps) / speed)
    status = build_curriculum_status(
        stages,
        running,
        current_stage,
        config=config,
        report=report,
        eta_seconds=eta_seconds,
    )
    status["runner_pid"] = runner_pid
    if not running and status["config"] is None:
        now = time.time()
        newest = max(
            (st for st in status["stages"] if st.get("mtime") is not None),
            key=lambda st: float(st["mtime"]),
            default=None,
        )
        if newest is not None and now - float(newest["mtime"]) < 180:
            status["cloud_active"] = True
            status["current_stage"] = newest["name"]
            idx = next(
                (i for i, st in enumerate(status["stages"]) if st["name"] == newest["name"]),
                None,
            )
            if idx is not None:
                for i, st in enumerate(status["stages"]):
                    if i < idx:
                        st["status"] = "completed" if st["exists"] else "pending"
                    elif i == idx:
                        st["status"] = "running"
                    else:
                        st["status"] = "pending"
    return status


def discover_custom_roots() -> list[pathlib.Path]:
    """自动发现非标准自定义训练任务（新任务无需改面板代码）。"""
    from rl.monitor import TASKS as _STANDARD_TASKS

    standard = set(_STANDARD_TASKS) | {"traverse_curve_curriculum"}
    runs_dir = PROJECT_ROOT / "rl" / "runs"
    roots: list[pathlib.Path] = []
    if not runs_dir.is_dir():
        return roots
    for d in sorted(runs_dir.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        if d.name in standard:
            continue
        roots.append(d)
    return roots


def high_level_process() -> tuple[bool, int | None, pathlib.Path | None]:
    """检测任意 scripts/train*.py 训练脚本与当前 run-dir/run-root。"""
    running = False
    pid: int | None = None
    run_dir: pathlib.Path | None = None
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmd = proc.info.get("cmdline") or []
        except (psutil.AccessDenied, psutil.ZombieProcess):
            continue
        joined = " ".join(cmd)
        if not any("scripts/train" in c and c.endswith(".py") for c in cmd):
            continue
        running = True
        pid = int(proc.info["pid"])
        for i, token in enumerate(cmd):
            if token in ("--run-dir", "--run-root") and i + 1 < len(cmd):
                run_dir = pathlib.Path(cmd[i + 1])
                break
    return running, pid, run_dir


def read_high_level_runs(
    roots: list[pathlib.Path] | None = None,
) -> list[dict]:
    """扫描高层训练 run 目录，返回每个 seed 的最新状态。"""
    roots = roots or discover_custom_roots()
    runs: list[dict] = []
    for task_dir in roots:
        if not task_dir.is_dir():
            continue
        for seed_dir in sorted(task_dir.glob("seed*")):
            if not seed_dir.is_dir():
                continue
            for run_dir in [seed_dir] + sorted(seed_dir.glob("stage*")):
                if not run_dir.is_dir():
                    continue
                if not (run_dir / "eval_log.csv").exists() and not (
                    run_dir / "best_model.zip"
                ).exists():
                    continue
                rows, _ = parse_eval_csv(run_dir / "eval_log.csv")
                last = rows[-1] if rows else None
                last_extra: dict[str, float] = {}
                eval_path = run_dir / "eval_log.csv"
                if eval_path.exists():
                    try:
                        lines = eval_path.read_text(encoding="utf-8").splitlines()
                        if len(lines) > 1:
                            header = lines[0].split(",")
                            vals = lines[-1].split(",")
                            mapping = dict(zip(header, vals))
                            for key in (
                                "fixed_success_rate",
                                "fixed_mean_x",
                                "random_success_rate",
                                "random_mean_x",
                            ):
                                if key in mapping:
                                    last_extra[key] = float(mapping[key])
                    except (OSError, ValueError):
                        pass
                her_stats: dict = {}
                her_path = run_dir / "her_log.csv"
                if her_path.exists():
                    try:
                        her_lines = her_path.read_text(encoding="utf-8").splitlines()
                        if len(her_lines) > 1:
                            parts = her_lines[-1].split(",")
                            her_stats = {
                                "timesteps": int(float(parts[0])),
                                "buffer_size": int(float(parts[1])),
                                "loss": float(parts[2]),
                                "positive_ratio": float(parts[3]),
                            }
                    except (OSError, IndexError, ValueError):
                        pass
                early_stop = ""
                stop_path = run_dir / "early_stop.txt"
                if stop_path.exists():
                    early_stop = stop_path.read_text(encoding="utf-8").strip()
                live: dict = {}
                live_path = run_dir / "live_status.json"
                if live_path.exists():
                    try:
                        live = json.loads(live_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError, TypeError):
                        live = {}
                timesteps = float(last.timesteps) if last else 0.0
                mean_reward = last.mean_reward if last else None
                mean_ep_len = last.ep_len if last else None
                if isinstance(live.get("timesteps"), (int, float)):
                    timesteps = float(live["timesteps"])
                    if isinstance(live.get("mean_reward"), (int, float)):
                        mean_reward = float(live["mean_reward"])
                    if isinstance(live.get("mean_ep_len"), (int, float)):
                        mean_ep_len = float(live["mean_ep_len"])
                runs.append(
                    {
                        "task": task_dir.name,
                        "seed": run_dir.name,
                        "path": str(run_dir),
                        "timesteps": timesteps,
                        "mean_reward": mean_reward,
                        "mean_ep_len": mean_ep_len,
                        "fixed_success_rate": last_extra.get("fixed_success_rate"),
                        "fixed_mean_x": last_extra.get("fixed_mean_x"),
                        "random_success_rate": last_extra.get("random_success_rate"),
                        "random_mean_x": last_extra.get("random_mean_x"),
                        "early_stop": early_stop,
                        "completed": (run_dir / ".completed").exists(),
                        "best_model": (run_dir / "best_model.zip").exists(),
                        "her": her_stats,
                        "live_stage": live.get("stage"),
                        "live_stage_timesteps": (
                            float(live["stage_timesteps"])
                            if isinstance(live.get("stage_timesteps"), (int, float))
                            else None
                        ),
                        "live_stage_target": (
                            float(live["stage_target"])
                            if isinstance(live.get("stage_target"), (int, float))
                            else None
                        ),
                        "live_total_target": (
                            float(live["total_target"])
                            if isinstance(live.get("total_target"), (int, float))
                            else None
                        ),
                        "live_fresh": (
                            bool(live)
                            and time.time() - live_path.stat().st_mtime < 180.0
                        ),
                        "mtime": run_dir.stat().st_mtime if run_dir.exists() else None,
                    }
                )
    return runs


def high_level_status() -> dict:
    """高层训练实时状态：进程 + 各 run 最新评估。"""
    running, pid, run_dir = high_level_process()
    runs = read_high_level_runs()
    current = (
        pathlib.Path(run_dir).resolve()
        if run_dir is not None
        else None
    )
    for r in runs:
        candidate = pathlib.Path(r["path"]).resolve()
        running_match = bool(
            running
            and current is not None
            and (
                current == candidate
                or current in candidate.parents
                or candidate in current.parents
            )
        )
        live_match = bool(
            running
            and r.get("live_fresh")
            and not r["completed"]
        )
        r["status"] = (
            "running"
            if (running_match or live_match) and not r["completed"]
            else "completed" if r["completed"] else "stopped"
        )
    return {
        "running": running,
        "pid": pid,
        "current_run_dir": str(run_dir) if run_dir else None,
        "runs": runs,
    }


def dedupe_alerts(
    alerts: list[dict],
    seen: dict[str, dict],
    now: float,
    cap: int = 50,
) -> tuple[list[dict], list[dict]]:
    for alert in alerts:
        alert_id = f"{alert['kind']}:{alert['key']}"
        if alert_id not in seen:
            seen[alert_id] = {
                "kind": alert["kind"],
                "key": alert["key"],
                "message": alert["message"],
                "first_seen": now,
                "last_seen": now,
            }
        else:
            seen[alert_id]["last_seen"] = now
            seen[alert_id]["message"] = alert["message"]
    active_ids = {f"{a['kind']}:{a['key']}" for a in alerts}
    current = [
        {"id": aid, **value}
        for aid, value in seen.items()
        if aid in active_ids
    ]
    history = list(seen.values())[-cap:]
    return current, history


def refresh_loop() -> None:
    global _snapshot
    last_history = 0.0
    while True:
        try:
            snap = build_snapshot(remote=REMOTE)
            now = time.time()
            if now - last_history >= HISTORY_INTERVAL:
                append_snapshot_history(snap)
                last_history = now
            current, history = dedupe_alerts(snap["alerts"], _alerts_seen, now)
            snap["current_alerts"] = current
            snap["alert_history"] = history
            snap["guard"] = guard_state()
            snap["cloud"] = cloud_sync_health()
            snap["curriculum"] = curriculum_status()
            snap["high_level"] = high_level_status()
            with _lock:
                _snapshot = snap
        except Exception:
            pass
        time.sleep(REFRESH_INTERVAL)


def token_loop() -> None:
    """后台线程：监听 Codex 日志，维护 token 使用记录。"""
    token_dashboard.load_history()
    while True:
        try:
            token_dashboard.refresh_once()
        except Exception:
            pass
        time.sleep(5.0)


def get_state() -> dict:
    with _lock:
        if _snapshot:
            return _snapshot
    return {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": [],
            "tasks": [], "alerts": [], "current_alerts": [],
            "alert_history": [], "guard": guard_state(), "error": "快照尚未生成"}


def valid_seed(seed: str) -> bool:
    return bool(re.fullmatch(r"seed\d{2}", seed))


def seed_report_links(task: str, seed: str) -> dict:
    links = {
        "report": f"/reports/{task}/{seed}/",
        "curve": f"/curve/{task}/{seed}",
    }
    if task in TRAVERSE_TASKS:
        links["scenarios"] = f"/scenarios/{task}/{seed}"
    return links


def append_snapshot_history(
    snapshot: dict,
    path: pathlib.Path = SNAPSHOT_HISTORY_PATH,
    cap: int = HISTORY_CAP,
) -> int:
    line = json.dumps(
        {
            "time": time.time(),
            "resources": snapshot.get("resources", {}),
            "seeds": [
                {
                    "key": s["key"],
                    "timesteps": s["timesteps"],
                    "reward": (
                        s.get("eval_reward")
                        if s.get("eval_reward") is not None
                        else s.get("tb_reward")
                    ),
                }
                for s in snapshot.get("seeds", [])
            ],
        },
        ensure_ascii=False,
    )
    lines = [line]
    try:
        lines = path.read_text(encoding="utf-8").splitlines() + [line]
    except OSError:
        pass
    if len(lines) > cap:
        lines = lines[-cap:]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def read_resource_history(
    path: pathlib.Path = SNAPSHOT_HISTORY_PATH,
    window_seconds: float = 7200.0,
) -> list[tuple[float, dict]]:
    """读取快照历史中的资源记录，返回 [(time, resources), ...]，仅保留最近窗口。"""
    rows: list[tuple[float, dict]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return rows
    cutoff = time.time() - window_seconds
    for line in lines[-HISTORY_CAP:]:
        try:
            row = json.loads(line)
            ts = float(row.get("time", 0.0))
            res = row.get("resources")
            if res and ts >= cutoff:
                rows.append((ts, res))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return rows


CLOUD_SYNC_LOG = GUARD_DIR / "cloud_sync.log"


def cloud_sync_health(path: pathlib.Path = CLOUD_SYNC_LOG) -> dict:
    """被动解析 cloud_sync.log，返回云端回传健康状态。"""
    if not path.exists():
        return {
            "exists": False,
            "last_mtime": None,
            "age_seconds": None,
            "ok": False,
            "fail": False,
            "stale": True,
            "detail": "无回传日志",
        }
    mtime = path.stat().st_mtime
    age = max(0.0, time.time() - mtime)
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        text = ""
    tail = text[-500:]
    tail_lower = tail.lower()
    fail = any(
        k in tail_lower
        for k in ("rsync error", "connection refused", "connection timed out",
                  "permission denied", "no such file")
    )
    ok = "完成。" in tail and not fail
    return {
        "exists": True,
        "last_mtime": mtime,
        "age_seconds": age,
        "ok": ok,
        "fail": fail,
        "stale": age > 120,
        "detail": "成功" if ok else ("失败" if fail else "未知"),
    }


def read_snapshot_history(
    seed_key: str,
    path: pathlib.Path = SNAPSHOT_HISTORY_PATH,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return points
    for line in lines[-HISTORY_CAP:]:
        try:
            row = json.loads(line)
            for seed in row.get("seeds", []):
                if seed.get("key") == seed_key and seed.get("reward") is not None:
                    points.append((float(seed["timesteps"]), float(seed["reward"])))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return points


def tail_file(path: pathlib.Path, n: int = 100) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return "(无日志)"
    return "\n".join(lines[-n:])


def valid_task_seed(task: str, seed: str) -> bool:
    return task in TASKS and valid_seed(seed)


def render_markdown(text: str) -> str:
    lines = text.splitlines()
    out: list[str] = []
    i = 0

    def inline(line: str) -> str:
        line = line.replace("![", "\x00IMG\x00")
        line = html.escape(line)
        line = re.sub(
            r"\x00IMG\x00([^]]*)\]\(([^)]*)\)",
            r'<img src="\2" alt="\1" style="max-width:100%">',
            line,
        )
        line = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", line)
        return line

    while i < len(lines):
        line = lines[i]
        if line.startswith("|"):
            table = []
            while i < len(lines) and lines[i].startswith("|"):
                table.append(lines[i])
                i += 1
            rows = [r.strip().strip("|").split("|") for r in table]
            rows = [r for r in rows if not all(re.fullmatch(r":?-{2,}:?", c.strip()) for c in r)]
            out.append("<table>")
            for idx, row in enumerate(rows):
                tag = "th" if idx == 0 else "td"
                out.append("<tr>" + "".join(f"<{tag}>{inline(c.strip())}</{tag}>" for c in row) + "</tr>")
            out.append("</table>")
            continue
        if line.startswith("### "):
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("## "):
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("# "):
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("- "):
            items = []
            while i < len(lines) and lines[i].startswith("- "):
                items.append(f"<li>{inline(lines[i][2:])}</li>")
                i += 1
            out.append("<ul>" + "".join(items) + "</ul>")
            continue
        elif line.strip() == "":
            pass
        else:
            out.append(f"<p>{inline(line)}</p>")
        i += 1
    return "\n".join(out)


def report_html(task: str, seed: str) -> bytes | None:
    md_path = REPORTS_DIR / task / seed / "report.md"
    try:
        md = md_path.read_text(encoding="utf-8")
    except OSError:
        return None
    body = render_markdown(md)
    page = (
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<title>报告</title>"
        "<style>body{background:#0f1117;color:#d7dae0;font-family:Consolas,monospace;"
        "max-width:900px;margin:30px auto;padding:0 20px}"
        "table{border-collapse:collapse}td,th{border:1px solid #2a2f3a;padding:6px 10px}"
        "a{color:#4fc3f7}</style></head><body>"
        f"<p><a href='/'>&larr; 返回面板</a></p>{body}</body></html>"
    )
    return page.encode("utf-8")


def curriculum_report_html(path: pathlib.Path = CURRICULUM_REPORT_PATH) -> bytes | None:
    try:
        md = path.read_text(encoding="utf-8")
    except OSError:
        return None
    body = render_markdown(md)
    page = (
        "<!DOCTYPE html><html lang='zh-CN'><head><meta charset='utf-8'>"
        "<title>课程学习报告</title>"
        "<style>body{background:#0f1117;color:#d7dae0;font-family:Consolas,monospace;"
        "max-width:900px;margin:30px auto;padding:0 20px}"
        "table{border-collapse:collapse}td,th{border:1px solid #2a2f3a;padding:6px 10px}"
        "a{color:#4fc3f7}</style></head><body>"
        f"<p><a href='/'>&larr; 返回面板</a></p>{body}</body></html>"
    )
    return page.encode("utf-8")


def safe_report_file(rel: str) -> pathlib.Path | None:
    target = (REPORTS_DIR / rel).resolve()
    try:
        target.relative_to(REPORTS_DIR.resolve())
    except ValueError:
        return None
    return target if target.is_file() else None


def safe_media_file(rel: str) -> pathlib.Path | None:
    target = (MEDIA_DIR / rel).resolve()
    try:
        target.relative_to(MEDIA_DIR.resolve())
    except ValueError:
        return None
    return target if target.is_file() else None


def list_demos(media_dir: pathlib.Path = MEDIA_DIR) -> list[dict]:
    """扫描演示视频：文件名 rl_<task>_<seed>.mp4（配套 .gif）。"""
    items: list[dict] = []
    try:
        files = sorted(media_dir.glob("rl_*.mp4"))
    except OSError:
        return items
    for p in files:
        m = DEMO_RE.match(p.name)
        if not m:
            continue
        gif = p.with_suffix(".gif")
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        items.append(
            {
                "task": m.group(1),
                "seed": m.group(2),
                "mp4": f"/media/{p.name}",
                "gif": f"/media/{gif.name}" if gif.is_file() else None,
                "mtime": mtime,
            }
        )
    return items


def seed_demos(task: str, seed: str,
               media_dir: pathlib.Path = MEDIA_DIR) -> dict:
    """返回单个任务/seed 的演示视频地址（不存在则为 None）。"""
    mp4 = media_dir / f"rl_{task}_{seed}.mp4"
    gif = media_dir / f"rl_{task}_{seed}.gif"
    return {
        "mp4": f"/media/{mp4.name}" if mp4.is_file() else None,
        "gif": f"/media/{gif.name}" if gif.is_file() else None,
    }


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args: object, **kwargs: object) -> None:
        pass

    def _send(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        if not _authorized(self):
            self._send(200, LOGIN_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return

        if path == "/":
            if INDEX_HTML.exists():
                self._send(200, INDEX_HTML.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(404, b"index.html missing", "text/plain")
            return

        if path == "/api/state":
            state = {**get_state(), "task_info": TASK_INFO}
            self._send(200, json.dumps(state, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
            return

        if path == "/curriculum/report":
            page = curriculum_report_html()
            if page is not None:
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._send(404, b"curriculum report not found", "text/plain")
            return

        if path == "/api/viewers":
            items = []
            for task in TASKS:
                for seed in ("seed00", "seed01", "seed02"):
                    key = f"{task}/{seed}"
                    info = viewer_launcher.viewer_info(task, seed)
                    items.append({"key": key, **info})
            self._send(200, json.dumps(items, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
            return

        if path == "/reports":
            items = []
            for md in sorted(REPORTS_DIR.glob("*/seed*/report.md")):
                task = md.parent.parent.name
                seed = md.parent.name
                items.append({
                    "task": task,
                    "seed": seed,
                    "report": f"/reports/{task}/{seed}/",
                    "curve": f"/curve/{task}/{seed}",
                    "scenarios": f"/scenarios/{task}/{seed}" if task in TRAVERSE_TASKS else None,
                })
            self._send(200, json.dumps(items, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
            return

        if path == "/api/demos":
            self._send(200, json.dumps(list_demos(), ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
            return

        m = re.fullmatch(r"/seed/(\w+)/(seed\d{2})", path)
        if m:
            task, seed = m.group(1), m.group(2)
            if task in TASKS and DETAIL_HTML.exists():
                self._send(200, DETAIL_HTML.read_bytes(), "text/html; charset=utf-8")
                return

        m = re.fullmatch(r"/api/seed/(\w+)/(seed\d{2})", path)
        if m:
            task, seed = m.group(1), m.group(2)
            if task not in TASKS or not valid_seed(seed):
                self._send(404, b"not found", "text/plain")
                return
            state = get_state()
            seed_state = next((s for s in state["seeds"] if s["key"] == f"{task}/{seed}"), None)
            guard_rec = state["guard"]["runs"].get(f"{task}/{seed}")
            payload = {
                "task": task,
                "seed": seed,
                "seed_state": seed_state,
                "guard": guard_rec,
                "links": seed_report_links(task, seed),
                "viewer": viewer_launcher.viewer_info(task, seed),
                "task_info": task_info(task),
                "demos": seed_demos(task, seed),
            }
            self._send(200, json.dumps(payload, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
            return

        m = re.fullmatch(r"/curve/(\w+)/(seed\d{2})", path)
        if m:
            task, seed = m.group(1), m.group(2)
            if task not in TASKS or not valid_seed(seed):
                self._send(404, b"not found", "text/plain")
                return
            points = report_mod.read_curve(task, int(seed[4:]))
            svg = report_mod.make_curve_svg(points, f"{task} {seed} 训练曲线")
            self._send(200, svg.encode("utf-8"), "image/svg+xml")
            return

        m = re.fullmatch(r"/history/(\w+)/(seed\d{2})", path)
        if m:
            task, seed = m.group(1), m.group(2)
            if task not in TASKS or not valid_seed(seed):
                self._send(404, b"not found", "text/plain")
                return
            points = read_snapshot_history(f"{task}/{seed}")
            svg = report_mod.make_curve_svg(points, f"{task} {seed} 持久化历史")
            self._send(200, svg.encode("utf-8"), "image/svg+xml")
            return

        m = re.fullmatch(r"/api/guard/log/(\w+)/(seed\d{2})", path)
        if m:
            task, seed = m.group(1), m.group(2)
            log_path = GUARD_DIR / "logs" / f"{task}_{seed}.log"
            payload = json.dumps({"log": tail_file(log_path)}, ensure_ascii=False).encode()
            self._send(200, payload, "application/json; charset=utf-8")
            return

        m = re.fullmatch(r"/scenarios/(\w+)/(seed\d{2})", path)
        if m:
            task, seed = m.group(1), m.group(2)
            scenario_page = WEB_DIR / "scenarios.html"
            if task in TRAVERSE_TASKS and scenario_page.exists():
                self._send(200, scenario_page.read_bytes(), "text/html; charset=utf-8")
                return

        m = re.fullmatch(r"/api/scenarios/(\w+)/(seed\d{2})", path)
        if m:
            task, seed = m.group(1), m.group(2)
            rows = report_mod.read_metrics(REPORTS_DIR / task / seed / "metrics.csv")
            payload = json.dumps({
                "task": task,
                "seed": seed,
                "scenarios": rows,
                "chart": f"/scenarios-chart/{task}/{seed}",
            }, ensure_ascii=False).encode()
            self._send(200, payload, "application/json; charset=utf-8")
            return

        m = re.fullmatch(r"/scenarios-chart/(\w+)/(seed\d{2})", path)
        if m:
            task, seed = m.group(1), m.group(2)
            rows = report_mod.read_metrics(REPORTS_DIR / task / seed / "metrics.csv")
            labels = [r.get("label", "") for r in rows]
            values = [float(r.get("success_rate", 0) or 0) for r in rows]
            svg = report_mod.make_bar_svg(labels, values, "场景成功率")
            self._send(200, svg.encode("utf-8"), "image/svg+xml")
            return

        if path == "/resources-chart":
            rows = read_resource_history()
            series = [
                (
                    "可用内存GB",
                    [(ts, (r.get("mem_available_mb") or 0.0) / 1024.0) for ts, r in rows],
                    "#4fc3f7",
                ),
                (
                    "Swap%",
                    [(ts, r.get("swap_percent") or 0.0) for ts, r in rows],
                    "#f2cc60",
                ),
            ]
            svg = report_mod.make_lines_svg(series, "系统资源趋势（可用内存 GB / Swap%）")
            self._send(200, svg.encode("utf-8"), "image/svg+xml")
            return

        if path == "/api/cloud":
            self._send(
                200,
                json.dumps(cloud_sync_health(), ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
            )
            return

        if path == "/api/token":
            self._send(
                200,
                json.dumps(token_dashboard.snapshot(), ensure_ascii=False).encode(),
                "application/json; charset=utf-8",
            )
            return

        if path == "/token":
            page = WEB_DIR / "token.html"
            if page.exists():
                self._send(200, page.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(404, b"token.html missing", "text/plain")
            return

        m = re.fullmatch(r"/reports/(\w+)/(seed\d{2})/", path)
        if m:
            task, seed = m.group(1), m.group(2)
            page = report_html(task, seed)
            if page is not None:
                self._send(200, page, "text/html; charset=utf-8")
            else:
                self._send(404, b"report not found", "text/plain")
            return

        if path.startswith("/reports/"):
            rel = path.removeprefix("/reports/")
            target = safe_report_file(rel)
            if target is not None:
                ctype = {
                    ".md": "text/plain; charset=utf-8",
                    ".svg": "image/svg+xml",
                    ".csv": "text/plain; charset=utf-8",
                }.get(target.suffix, "application/octet-stream")
                self._send(200, target.read_bytes(), ctype)
                return
            self._send(404, b"not found", "text/plain")
            return

        m = re.fullmatch(r"/media/([^/]+)", path)
        if m:
            target = safe_media_file(m.group(1))
            if target is not None:
                ctype = {
                    ".mp4": "video/mp4",
                    ".gif": "image/gif",
                }.get(target.suffix, "application/octet-stream")
                self._send(200, target.read_bytes(), ctype)
                return
            self._send(404, b"not found", "text/plain")
            return

        self._send(404, b"not found", "text/plain")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length) if length else b""
        path = unquote(urlparse(self.path).path)

        if path == "/login":
            token = None
            try:
                token = json.loads(raw).get("token")
            except (json.JSONDecodeError, ValueError):
                token = parse_qs(raw.decode("utf-8", errors="replace")).get("token", [None])[0]
            if token and token == TOKEN:
                self.send_response(302)
                self.send_header("Location", "/")
                self.send_header("Set-Cookie", f"go2w_token={TOKEN}; Path=/; SameSite=Lax")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send(401, b'{"ok":false,"error":"wrong token"}',
                       "application/json; charset=utf-8")
            return

        if not _authorized(self):
            self._send(401, b'{"ok":false,"error":"unauthorized"}',
                       "application/json; charset=utf-8")
            return

        try:
            data = json.loads(raw or b"{}")
        except (json.JSONDecodeError, ValueError):
            data = {}

        if path == "/api/action/sync":
            try:
                (PROJECT_ROOT / "rl" / ".autosync.trigger").touch()
                self._send(200, b'{"ok":true}', "application/json; charset=utf-8")
            except OSError as exc:
                self._send(500, json.dumps({"ok": False, "error": str(exc)}).encode(),
                           "application/json; charset=utf-8")
            return

        if path == "/api/action/restart-guard":
            try:
                subprocess.run(
                    ["systemctl", "--user", "restart", "go2w-train-guard"],
                    capture_output=True,
                    timeout=30,
                )
                self._send(200, b'{"ok":true}', "application/json; charset=utf-8")
            except Exception as exc:
                self._send(500, json.dumps({"ok": False, "error": str(exc)}).encode(),
                           "application/json; charset=utf-8")
            return

        if path == "/api/action/report":
            task = str(data.get("task", ""))
            seed = str(data.get("seed", ""))
            if not valid_task_seed(task, seed):
                self._send(400, b'{"ok":false,"error":"invalid task/seed"}',
                           "application/json; charset=utf-8")
                return

            def run_report_job() -> None:
                subprocess.run(
                    [
                        sys.executable,
                        str(PROJECT_ROOT / "rl" / "report.py"),
                        "--task", task,
                        "--seed", str(int(seed[4:])),
                    ],
                    cwd=str(PROJECT_ROOT),
                    timeout=1800,
                )

            threading.Thread(target=run_report_job, daemon=True).start()
            self._send(200, b'{"ok":true,"started":true}', "application/json; charset=utf-8")
            return

        if path == "/api/action/viewer-start":
            task = str(data.get("task", ""))
            seed = str(data.get("seed", ""))
            scenario = str(data.get("scenario", "random"))
            result = viewer_launcher.launch_viewer(task, seed, scenario)
            code = 400 if result.get("invalid") else (409 if result.get("pid") and not result.get("ok") else 200)
            self._send(code, json.dumps(result, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
            return

        if path == "/api/action/viewer-stop":
            task = str(data.get("task", ""))
            seed = str(data.get("seed", ""))
            result = viewer_launcher.stop_viewer(task, seed)
            code = 400 if result.get("invalid") else 200
            self._send(code, json.dumps(result, ensure_ascii=False).encode(),
                       "application/json; charset=utf-8")
            return

        self._send(404, b"not found", "text/plain")


def main() -> None:
    global TOKEN, REMOTE
    parser = argparse.ArgumentParser(description="RL 本地 Web 面板")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8787)
    parser.add_argument("--token", default=None, help="访问密码；不设置则无鉴权")
    parser.add_argument("--remote", action="store_true",
                        help="云端聚合模式：数据新鲜即视为运行")
    args = parser.parse_args()

    TOKEN = args.token or os.environ.get("GO2W_TOKEN", "")
    REMOTE = args.remote

    threading.Thread(target=refresh_loop, daemon=True).start()
    threading.Thread(target=token_loop, daemon=True).start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Web 面板: http://{args.host}:{args.port}" + ("（需 token）" if TOKEN else ""))
    server.serve_forever()


if __name__ == "__main__":
    main()
