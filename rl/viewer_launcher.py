#!/usr/bin/env python3
"""MuJoCo 查看器启动器：Web 面板一键启动/关闭 gui_viewer.py。

设计要点：
1. 复用 rl/runs/<task>/<seed>/best_model.zip + best_vec_normalize.pkl；
2. task/seed/scenario 全部白名单校验，防止路径穿越；
3. 服务环境可能没有 DISPLAY，自动探测 /tmp/.X11-unix 下的 X socket；
4. 查看器以独立进程组启动（不随面板退出而终止），日志写
   rl/runs/_guard/viewer_*.log，运行状态记录在 viewers.json（已忽略）。
"""

from __future__ import annotations

import json
import os
import pathlib
import re
import signal
import subprocess
import sys
import time
from typing import Any, Optional

from rl.go2w_env import SCENARIOS as ENV_SCENARIOS
from rl.go2w_env import TASK_SCENARIO, TRAVERSE_TASKS


PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS_ROOT = PROJECT_ROOT / "rl" / "runs"
VIEWERS_STATE = RUNS_ROOT / "_guard" / "viewers.json"
VIEWER_SCRIPT = PROJECT_ROOT / "rl" / "gui_viewer.py"

TASKS = ("balance", "full_chain", *TRAVERSE_TASKS)
SCENARIOS = (*ENV_SCENARIOS, "random")
SEED_RE = re.compile(r"^seed\d{2}$")


def valid_task(task: str) -> bool:
    return task in TASKS


def valid_seed(seed: str) -> bool:
    return bool(SEED_RE.fullmatch(seed))


def valid_scenario(scenario: str) -> bool:
    return scenario in SCENARIOS


def find_model(task: str, seed: str,
               runs_root: pathlib.Path = RUNS_ROOT) -> Optional[tuple[pathlib.Path, pathlib.Path]]:
    """返回 (best_model.zip, best_vec_normalize.pkl)，缺任一文件则返回 None。"""
    if not valid_task(task) or not valid_seed(seed):
        return None
    run_dir = runs_root / task / seed
    model = run_dir / "best_model.zip"
    vec_norm = run_dir / "best_vec_normalize.pkl"
    if model.exists() and vec_norm.exists():
        return model, vec_norm
    return None


def build_command(task: str, seed: str, scenario: str = "random",
                  runs_root: pathlib.Path = RUNS_ROOT,
                  python: str = sys.executable) -> list[str]:
    """构造 gui_viewer.py 命令；只有 traverse 传场景参数。"""
    model, vec_norm = find_model(task, seed, runs_root=runs_root) or (None, None)
    if model is None or vec_norm is None:
        raise FileNotFoundError(f"缺少模型文件: {runs_root / task / seed}")
    cmd = [
        python,
        str(VIEWER_SCRIPT),
        "--task", task,
        "--model", str(model),
        "--vec-norm", str(vec_norm),
    ]
    if task in TRAVERSE_TASKS:
        cmd += ["--scenario", scenario if valid_scenario(scenario) else TASK_SCENARIO[task]]
    return cmd


def detect_display() -> str:
    """返回可用的 DISPLAY；优先取环境变量，其次探测 X socket。"""
    env_display = os.environ.get("DISPLAY", "")
    if env_display:
        return env_display
    x11_dir = pathlib.Path("/tmp/.X11-unix")
    if x11_dir.is_dir():
        for sock in sorted(x11_dir.glob("X*")):
            try:
                num = int(sock.name[1:])
            except ValueError:
                continue
            return f":{num}"
    return ""


def read_state(state_path: pathlib.Path = VIEWERS_STATE) -> dict[str, dict]:
    if not state_path.exists():
        return {}
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def write_state(state: dict[str, dict], state_path: pathlib.Path = VIEWERS_STATE) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def viewer_status(state_path: pathlib.Path = VIEWERS_STATE) -> list[dict[str, Any]]:
    """返回所有查看器状态（含存活标记），按启动时间排序。"""
    state = read_state(state_path)
    items = []
    for key, rec in state.items():
        pid = int(rec.get("pid", 0))
        items.append(
            {
                "key": key,
                "task": rec.get("task"),
                "seed": rec.get("seed"),
                "scenario": rec.get("scenario"),
                "pid": pid,
                "alive": is_alive(pid),
                "started_at": rec.get("started_at"),
                "display": rec.get("display"),
                "model": rec.get("model"),
            }
        )
    items.sort(key=lambda item: item.get("started_at") or 0)
    return items


def viewer_info(task: str, seed: str,
                state_path: pathlib.Path = VIEWERS_STATE) -> dict[str, Any]:
    """单 seed 的查看器信息，供前端表格与详情页使用。"""
    key = f"{task}/{seed}"
    model_available = find_model(task, seed) is not None
    for item in viewer_status(state_path=state_path):
        if item["key"] == key:
            return {
                "model_available": model_available,
                "running": item["alive"],
                "pid": item["pid"] if item["alive"] else None,
                "scenario": item["scenario"] if item["alive"] else None,
            }
    return {"model_available": model_available, "running": False,
            "pid": None, "scenario": None}


def launch_viewer(task: str, seed: str, scenario: str = "random",
                  state_path: pathlib.Path = VIEWERS_STATE,
                  log_dir: Optional[pathlib.Path] = None,
                  display: Optional[str] = None,
                  python: Optional[str] = None,
                  runs_root: Optional[pathlib.Path] = None) -> dict[str, Any]:
    """启动 gui_viewer.py；返回 JSON 风格结果。"""
    if not valid_task(task) or not valid_seed(seed):
        return {"ok": False, "error": f"非法的 task/seed: {task}/{seed}", "invalid": True}
    if task in TRAVERSE_TASKS and not valid_scenario(scenario):
        return {"ok": False, "error": f"非法的场景: {scenario}", "invalid": True}

    rr = runs_root or RUNS_ROOT
    model = find_model(task, seed, runs_root=rr)
    if model is None:
        return {"ok": False, "error": "该 seed 还没有可用的 best_model.zip / best_vec_normalize.pkl"}

    key = f"{task}/{seed}"
    state = read_state(state_path)
    old = state.get(key)
    if old and is_alive(int(old.get("pid", 0))):
        return {"ok": False, "error": "该任务的查看器已在运行", "pid": old["pid"]}

    resolved_display = display if display is not None else detect_display()
    if not resolved_display:
        return {"ok": False, "error": "找不到图形显示环境 (DISPLAY)，无法打开 MuJoCo 窗口"}

    cmd = build_command(
        task, seed,
        scenario=scenario if task in TRAVERSE_TASKS else "random",
        python=python or sys.executable,
        runs_root=rr,
    )
    log_path = (log_dir or rr / "_guard") / f"viewer_{task}_{seed}_{scenario}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(log_path, "a", encoding="utf-8") as log:
            proc = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                env={**os.environ, "DISPLAY": resolved_display},
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
    except OSError as exc:
        return {"ok": False, "error": f"启动失败: {exc}"}

    state[key] = {
        "pid": proc.pid,
        "task": task,
        "seed": seed,
        "scenario": scenario if task in TRAVERSE_TASKS else "random",
        "started_at": time.time(),
        "display": resolved_display,
        "model": str(model[0]),
    }
    write_state(state, state_path=state_path)
    return {"ok": True, "pid": proc.pid, "log": str(log_path)}


def stop_viewer(task: str, seed: str,
                state_path: pathlib.Path = VIEWERS_STATE) -> dict[str, Any]:
    """关闭指定 seed 的查看器并清理状态。"""
    if not valid_task(task) or not valid_seed(seed):
        return {"ok": False, "error": f"非法的 task/seed: {task}/{seed}", "invalid": True}
    key = f"{task}/{seed}"
    state = read_state(state_path)
    rec = state.get(key)
    if not rec:
        return {"ok": False, "error": "该 seed 没有查看器在运行"}
    pid = int(rec.get("pid", 0))
    try:
        if is_alive(pid):
            os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (OSError, ProcessLookupError):
        pass
    state.pop(key, None)
    write_state(state, state_path=state_path)
    return {"ok": True, "pid": pid}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MuJoCo 查看器启动/停止/状态")
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=str, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, default="random")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--start", action="store_true")
    action.add_argument("--stop", action="store_true")
    action.add_argument("--status", action="store_true")
    args = parser.parse_args()

    if args.status:
        print(json.dumps(viewer_status(), ensure_ascii=False, indent=2))
    elif args.start:
        print(json.dumps(launch_viewer(args.task, args.seed, args.scenario),
                         ensure_ascii=False, indent=2))
    else:
        print(json.dumps(stop_viewer(args.task, args.seed),
                         ensure_ascii=False, indent=2))
