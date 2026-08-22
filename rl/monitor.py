"""终端实时监控 RL 训练进度（不修改训练端）。

数据源：eval_log.csv + 最新 TensorBoard 事件 + psutil 系统资源。
功能：进度/奖励/速度/ETA、多 seed 聚合、ASCII 趋势图、进程存活与
奖励异常告警（终端高亮 + notify-send 桌面通知）。

用法：
    python rl/monitor.py                 # 每 10 秒刷新
    python rl/monitor.py --once          # 只打印一次
    python rl/monitor.py --json          # 单次机器可读输出
    python rl/monitor.py --no-notify     # 关闭桌面通知
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import pathlib
import shutil
import subprocess
import time
import types
from typing import Optional

import psutil
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "rl" / "runs"
TASKS = ("balance", "full_chain", "full_chain_simple",
         "traverse_flat_slope", "traverse_slope", "traverse_curve")

# 当前目标：traverse_slope 补训到 8M，其余单 seed 4M 步
TARGET_STEPS: dict[str, int] = {task: 4_000_000 for task in TASKS}
TARGET_STEPS["traverse_slope"] = 8_000_000


def target_for(task: str, default: int = 8_000_000) -> int:
    return TARGET_STEPS.get(task, default)
SPARK_CHARS = "▁▂▃▄▅▆▇█"


@dataclasses.dataclass
class EvalRow:
    timesteps: float
    mean_reward: float
    std_reward: float
    ep_len: float


@dataclasses.dataclass
class SeedState:
    task: str
    seed: str
    eval_row: Optional[EvalRow]
    eval_reward: Optional[float]
    tb_reward: Optional[float]
    tb_ep_len: Optional[float]
    timesteps: float
    completed: bool
    alive: bool
    stale: bool
    nan: bool
    speed: Optional[float]
    eta: Optional[float]
    reward_history: list[float]
    source_mtime: float
    sample_time: float
    tb_std: Optional[float] = None
    tb_value_loss: Optional[float] = None
    tb_kl: Optional[float] = None
    tb_curve: list[list[float]] = dataclasses.field(default_factory=list)
    source: str = "local"

    @property
    def key(self) -> str:
        return f"{self.task}/seed{self.seed}"


def parse_eval_csv(csv_path: pathlib.Path) -> tuple[list[EvalRow], bool]:
    """解析 eval_log.csv，返回 (行列表, 是否出现 NaN/Inf)。"""
    rows: list[EvalRow] = []
    nan = False
    try:
        with open(csv_path, encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except OSError:
        return rows, nan
    for line in lines[1:]:
        if "nan" in line.lower() or "inf" in line.lower():
            nan = True
            continue
        parts = line.split(",")
        if len(parts) < 4:
            continue
        try:
            rows.append(
                EvalRow(
                    timesteps=float(parts[0]),
                    mean_reward=float(parts[1]),
                    std_reward=float(parts[2]),
                    ep_len=float(parts[3]),
                )
            )
        except ValueError:
            continue
    return rows, nan


def sparkline(values: list[float], width: int = 20) -> str:
    """把最近 N 个值渲染成 ASCII 趋势条。"""
    vals = values[-width:]
    if not vals:
        return "—"
    lo, hi = min(vals), max(vals)
    if hi <= lo:
        return "▄" * len(vals)
    out = []
    for v in vals:
        idx = int((v - lo) / (hi - lo) * (len(SPARK_CHARS) - 1) + 0.5)
        out.append(SPARK_CHARS[max(0, min(idx, len(SPARK_CHARS) - 1))])
    return "".join(out)


def latest_tensorboard_dir(seed_dir: pathlib.Path) -> Optional[pathlib.Path]:
    tb_root = seed_dir / "tensorboard"
    if not tb_root.is_dir():
        return None
    runs = [d for d in tb_root.iterdir() if d.is_dir()]
    if not runs:
        return None
    return max(runs, key=lambda d: d.stat().st_mtime)


def read_tensorboard(seed_dir: pathlib.Path) -> Optional[dict]:
    """读取最新 TensorBoard 事件，返回 rollout 级指标与内部速度。"""
    tb_dir = latest_tensorboard_dir(seed_dir)
    if tb_dir is None:
        return None
    try:
        acc = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
        acc.Reload()
    except Exception:
        return None

    def last(tag: str):
        try:
            events = acc.Scalars(tag)
            return events[-1] if events else None
        except KeyError:
            return None

    try:
        rew_events = acc.Scalars("rollout/ep_rew_mean")
    except KeyError:
        rew_events = []
    if not rew_events:
        return None
    tt = rew_events[-1]
    rew = tt
    std = last("train/std")
    ep_len = last("rollout/ep_len_mean")
    value_loss = last("train/value_loss")
    kl = last("train/approx_kl")

    speed = None
    if len(rew_events) >= 2:
        dt = rew_events[-1].wall_time - rew_events[-2].wall_time
        if dt > 0:
            speed = max(0.0, (rew_events[-1].step - rew_events[-2].step) / dt)

    return {
        "timesteps": float(tt.step),
        "reward": float(rew.value) if rew else None,
        "std": float(std.value) if std else None,
        "value_loss": float(value_loss.value) if value_loss else None,
        "kl": float(kl.value) if kl else None,
        "ep_len": float(ep_len.value) if ep_len else None,
        "speed": speed,
        "mtime": tb_dir.stat().st_mtime,
        "curve": [[float(e.step), float(e.value)] for e in rew_events[-200:]],
    }


def find_running() -> set[tuple[str, str]]:
    """扫描 psutil，返回正在运行的 (task, seed) 集合。"""
    running: set[tuple[str, str]] = set()
    for proc in psutil.process_iter(["cmdline"]):
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
                seed = str(cmd[i + 1]).zfill(2)
        if task and seed is not None:
            running.add((task, seed))
    return running


def external_seeds() -> set[tuple[str, str]]:
    """读取守护状态，返回由外部机器（云端）负责训练的 (task, seed) 集合。"""
    state_path = RUNS_DIR / "_guard" / "state.json"
    external: set[tuple[str, str]] = set()
    try:
        data = json.loads(state_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return external
    for key, rec in data.get("runs", {}).items():
        if rec.get("status") == "external" and rec.get("owner", "cloud") != "local":
            task, seed_part = key.split("/", 1)
            external.add((task, seed_part.removeprefix("seed").zfill(2)))
    return external


def resource_snapshot() -> dict:
    load = os.getloadavg()
    try:
        cpu = psutil.cpu_percent(interval=0.3)
    except Exception:
        cpu = 0.0
    vm = psutil.virtual_memory()
    try:
        sw = psutil.swap_memory()
    except Exception:
        sw = None
    return {
        "load": load,
        "cpu": cpu,
        "mem": vm.percent,
        "mem_available_mb": vm.available / 1024**2,
        "mem_total_mb": vm.total / 1024**2,
        "swap_used_mb": sw.used / 1024**2 if sw else 0.0,
        "swap_total_mb": sw.total / 1024**2 if sw else 0.0,
        "swap_percent": sw.percent if sw else 0.0,
    }


_oom_cache: dict = {"ts": 0.0, "event": None}


def last_oom_event() -> Optional[dict]:
    """低频读取内核日志，返回最近一次 OOM 事件（被杀进程 + 日志片段）。"""
    global _oom_cache
    now = time.time()
    if now - _oom_cache["ts"] < 60.0:
        return _oom_cache["event"]
    _oom_cache["ts"] = now
    event = None
    try:
        out = subprocess.run(
            ["journalctl", "-k", "--since", "24 hours ago", "--no-pager"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
        for line in reversed(out.splitlines()):
            if "oom-kill" in line or "Out of memory" in line:
                m = __import__("re").search(r"Killed process \d+ \((\w+)\)", line)
                event = {
                    "process": m.group(1) if m else "unknown",
                    "line": line.strip()[:200],
                }
                break
    except Exception:
        event = None
    _oom_cache["event"] = event
    return event


def resource_alerts(res: dict) -> list[tuple[str, str, str]]:
    """基于资源快照生成系统级告警（低内存 / 高 Swap）。"""
    alerts: list[tuple[str, str, str]] = []
    avail = res.get("mem_available_mb")
    if avail is not None and avail < 2048:
        alerts.append(("low_mem", "system", f"可用内存仅 {avail:.0f}MB"))
    swap_pct = res.get("swap_percent")
    if swap_pct is not None and swap_pct > 50:
        alerts.append(("high_swap", "system", f"Swap 使用 {swap_pct:.0f}%"))
    return alerts


def collect_seed_state(
    task: str,
    seed: str,
    target: int,
    stale_minutes: float,
    now: float,
    running: set[tuple[str, str]],
    external: set[tuple[str, str]],
    remote: bool = False,
) -> SeedState:
    seed_dir = RUNS_DIR / task / f"seed{int(seed):02d}"
    eval_rows, nan = parse_eval_csv(seed_dir / "eval_log.csv")
    last_eval = eval_rows[-1] if eval_rows else None
    tb = read_tensorboard(seed_dir)

    timesteps = float(
        tb["timesteps"] if tb else (last_eval.timesteps if last_eval else 0.0)
    )
    eval_reward = last_eval.mean_reward if last_eval else None
    tb_reward = tb["reward"] if tb else None
    tb_ep_len = tb["ep_len"] if tb else None
    tb_std = tb["std"] if tb else None
    tb_value_loss = tb["value_loss"] if tb else None
    tb_kl = tb["kl"] if tb else None
    tb_curve = tb["curve"] if tb else []
    reward_history = [r.mean_reward for r in eval_rows]
    if not reward_history and tb_reward is not None:
        reward_history = [tb_reward]

    t_target = target_for(task, target)
    # 完成以训练端写入的 final_model.zip + .completed 为准（避免“超步数但未完成”误判）
    completed = (seed_dir / "final_model.zip").exists() and (
        seed_dir / ".completed"
    ).exists()
    mtimes = [seed_dir.stat().st_mtime] if seed_dir.exists() else [0.0]
    eval_csv = seed_dir / "eval_log.csv"
    if eval_csv.exists():
        mtimes.append(eval_csv.stat().st_mtime)
    if tb is not None:
        mtimes.append(tb["mtime"])
    source_mtime = max(mtimes)
    stale = (not completed) and (now - source_mtime) > stale_minutes * 60.0
    # 外部机器负责的 seed 没有本地训练进程，改用数据新鲜度判断是否仍在推进。
    # remote 模式下，任何数据新鲜且已有步数的 seed 都视为运行（用于云端聚合面板）。
    alive = (
        (task, seed) in running
        or ((task, seed) in external and not stale)
        or (remote and not stale and timesteps > 0)
    )

    return SeedState(
        task=task,
        seed=seed,
        eval_row=last_eval,
        eval_reward=eval_reward,
        tb_reward=tb_reward,
        tb_ep_len=tb_ep_len,
        timesteps=timesteps,
        completed=completed,
        alive=alive,
        stale=stale,
        nan=nan,
        speed=tb["speed"] if tb else None,
        eta=None,
        reward_history=reward_history,
        source_mtime=source_mtime,
        sample_time=now,
        tb_std=tb_std,
        tb_value_loss=tb_value_loss,
        tb_kl=tb_kl,
        tb_curve=tb_curve,
        source="cloud" if (task, seed) in external else "local",
    )


def discover_seeds(args, now: float, remote: bool = False) -> list[SeedState]:
    running = find_running()
    external = external_seeds()
    states: list[SeedState] = []
    for task in TASKS:
        task_dir = RUNS_DIR / task
        if not task_dir.is_dir():
            continue
        for seed_dir in sorted(task_dir.glob("seed*")):
            seed = seed_dir.name.removeprefix("seed")
            st = collect_seed_state(
                task, seed, args.target, args.stale_minutes, now, running, external, remote
            )
            recent = (now - st.source_mtime) < args.max_age_hours * 3600.0
            if args.all or recent or st.completed:
                states.append(st)
    return states


def evaluate_alerts(
    states: list[SeedState],
    prev_alive: dict[str, bool],
    prev_reward: dict[str, float],
    min_reward: Optional[float],
    now: float,
) -> list[tuple[str, str, str]]:
    alerts: list[tuple[str, str, str]] = []
    for st in states:
        key = st.key
        if prev_alive.get(key) and not st.alive:
            alerts.append(("process_died", key, f"{key} 训练进程已停止"))
        if st.stale and st.timesteps >= 50_000:
            alerts.append(("stale", key, f"{key} 数据超过 {now - st.source_mtime:.0f}s 未更新"))
        if st.nan:
            alerts.append(("nan", key, f"{key} 评估日志出现 NaN"))
        r = st.eval_reward if st.eval_reward is not None else st.tb_reward
        if r is not None:
            prev = prev_reward.get(key)
            if prev is not None and r < prev * 0.9:
                alerts.append(("reward_drop", key, f"{key} 奖励下降 {prev:.2f} -> {r:.2f}"))
            if min_reward is not None and r < min_reward:
                alerts.append(("low_reward", key, f"{key} 奖励低于阈值 {min_reward:.2f}"))
    return alerts


def throttle_alerts(
    alerts: list[tuple[str, str, str]],
    last_sent: dict[str, float],
    now: float,
    cooldown: float = 600.0,
) -> list[tuple[str, str, str]]:
    out = []
    for kind, key, msg in alerts:
        alert_id = f"{kind}:{key}"
        if now - last_sent.get(alert_id, 0.0) >= cooldown:
            last_sent[alert_id] = now
            out.append((kind, key, msg))
    return out


def notify_desktop(title: str, msg: str, enabled: bool) -> None:
    if not enabled or shutil.which("notify-send") is None:
        return
    try:
        subprocess.run(
            ["notify-send", "-u", "normal", title, msg],
            timeout=5,
            capture_output=True,
        )
    except Exception:
        pass


def format_eta(seconds: Optional[float]) -> str:
    if seconds is None or seconds < 0:
        return "--"
    hours, rem = divmod(int(seconds), 3600)
    minutes, secs = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def color(text: str, code: int, enabled: bool) -> str:
    return f"\033[{code}m{text}\033[0m" if enabled else text


def aggregate_tasks(states: list[SeedState], seeds_total: int, target: int) -> list[dict]:
    by_task: dict[str, list[SeedState]] = {}
    for st in states:
        by_task.setdefault(st.task, []).append(st)
    rows = []
    for task in TASKS:
        task_states = by_task.get(task, [])
        if not task_states:
            continue
        t_target = target_for(task, target)
        done = sum(1 for st in task_states if st.completed)
        best = None
        for st in task_states:
            r = st.eval_reward if st.eval_reward is not None else st.tb_reward
            if r is not None and (best is None or r > best):
                best = r
        total_steps = sum(st.timesteps for st in task_states)
        speeds = [st.speed for st in task_states if st.speed]
        mean_speed = sum(speeds) / len(speeds) if speeds else None
        eta = None
        if mean_speed and mean_speed > 0:
            eta = (t_target * seeds_total - total_steps) / mean_speed
        rows.append(
            {
                "task": task,
                "done": done,
                "total": seeds_total,
                "best": best,
                "total_steps": total_steps,
                "target_steps": t_target * seeds_total,
                "eta": eta,
            }
        )
    return rows


def fill_speed_eta(states: list[SeedState], target: int) -> None:
    for st in states:
        if st.speed and st.speed > 0 and not st.completed:
            st.eta = max(0.0, (target_for(st.task, target) - st.timesteps) / st.speed)


def snapshot_dict(
    states: list[SeedState],
    tasks: list[dict],
    res: dict,
    alerts: list[tuple[str, str, str]],
) -> dict:
    return {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "resources": {
            "cpu_percent": res["cpu"],
            "mem_percent": res["mem"],
            "load": list(res["load"]),
            "mem_available_mb": res.get("mem_available_mb"),
            "mem_total_mb": res.get("mem_total_mb"),
            "swap_used_mb": res.get("swap_used_mb"),
            "swap_total_mb": res.get("swap_total_mb"),
            "swap_percent": res.get("swap_percent"),
        },
        "seeds": [
            {
                "key": st.key,
                "timesteps": st.timesteps,
                "eval_reward": st.eval_reward,
                "tb_reward": st.tb_reward,
                "ep_len": (
                    st.tb_ep_len
                    if st.tb_ep_len is not None
                    else (st.eval_row.ep_len if st.eval_row else None)
                ),
                "speed": st.speed,
                "eta_seconds": st.eta,
                "completed": st.completed,
                "alive": st.alive,
                "stale": st.stale,
                "nan": st.nan,
                "history": st.reward_history[-20:],
                "tb_std": st.tb_std,
                "tb_value_loss": st.tb_value_loss,
                "tb_kl": st.tb_kl,
                "tb_curve": st.tb_curve[-200:],
                "source": st.source,
            }
            for st in states
        ],
        "tasks": tasks,
        "alerts": [{"kind": k, "key": key, "message": msg} for k, key, msg in alerts],
        "oom_event": last_oom_event(),
    }


def build_snapshot(
    target: int = 8_000_000,
    seeds_total: int = 3,
    max_age_hours: float = 6.0,
    stale_minutes: float = 10.0,
    all_seeds: bool = False,
    min_reward: Optional[float] = None,
    remote: bool = False,
) -> dict:
    """无副作用的训练状态快照（供 Web 面板后台线程调用）。"""
    now = time.time()
    res = resource_snapshot()
    args = types.SimpleNamespace(
        target=target,
        stale_minutes=stale_minutes,
        max_age_hours=max_age_hours,
        all=all_seeds,
    )
    states = discover_seeds(args, now, remote)
    fill_speed_eta(states, target)
    alerts = evaluate_alerts(states, {}, {}, min_reward, now) + resource_alerts(res)
    tasks = aggregate_tasks(states, seeds_total, target)
    return snapshot_dict(states, tasks, res, alerts)


def render(
    states: list[SeedState],
    tasks: list[dict],
    res: dict,
    alerts: list[tuple[str, str, str]],
    args,
) -> str:
    c = not args.no_color
    lines = []
    load_txt = "/".join(f"{x:.1f}" for x in res["load"])
    lines.append(
        time.strftime("RL 训练进度  %Y-%m-%d %H:%M:%S")
        + f"   CPU {res['cpu']:.0f}%   内存 {res['mem']:.0f}%   负载 {load_txt}"
    )
    header = (
        f"{'任务/seed':<24}{'进度':>18}{'奖励':>10}{'ep_len':>8}"
        f"{'速度':>9}{'预计剩余':>11}{'状态':>6}  趋势"
    )
    lines.append(header)
    for st in states:
        t_target = target_for(st.task, args.target)
        pct = 100.0 * st.timesteps / t_target
        reward = st.eval_reward if st.eval_reward is not None else st.tb_reward
        reward_txt = f"{reward:.2f}" if reward is not None else "--"
        ep_len = st.tb_ep_len if st.tb_ep_len is not None else (
            st.eval_row.ep_len if st.eval_row else None
        )
        ep_txt = f"{ep_len:.0f}" if ep_len is not None else "--"
        speed_txt = f"{st.speed:.0f}" if st.speed is not None else "--"
        if st.completed:
            status = color("完成", 36, c)
        elif st.alive:
            status = color("运行", 32, c)
        else:
            status = color("停止", 91, c)
        lines.append(
            f"{st.key:<24}{st.timesteps/1e6:>6.2f}/{t_target/1e6:.2f}M ({pct:>5.1f}%)"
            f"{reward_txt:>10}{ep_txt:>8}{speed_txt:>9}{format_eta(st.eta):>11}"
            f"{status:>6}  {sparkline(st.reward_history)}"
        )
    for t in tasks:
        best_txt = f"{t['best']:.2f}" if t["best"] is not None else "--"
        lines.append(color(f"{t['task']}（合计）", 33, c))
        lines.append(
            f"  完成 {t['done']}/{t['total']} seeds   最佳奖励 {best_txt}"
            f"   总步数 {t['total_steps']/1e6:.2f}/{t['target_steps']/1e6:.2f}M"
            f"   预计剩余 {format_eta(t['eta'])}"
        )
    if alerts:
        lines.append("")
        lines.append(color("告警:", 91, c))
        for kind, key, msg in alerts:
            lines.append(color(f"  [{kind}] {msg}", 91 if kind in ("process_died", "nan") else 93, c))
    elif not states:
        lines.append("尚未找到训练数据，等待输出……")
    return "\n".join(lines)


def to_json(states, tasks, res, alerts) -> str:
    return json.dumps(snapshot_dict(states, tasks, res, alerts), ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="RL 训练实时监控（不修改训练端）")
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--target", type=int, default=8_000_000)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--max-age-hours", type=float, default=6.0,
                        help="只显示最近 N 小时内更新过或已完成的训练")
    parser.add_argument("--all", action="store_true", help="显示所有历史训练")
    parser.add_argument("--stale-minutes", type=float, default=10.0,
                        help="数据超过 N 分钟未更新视为停滞")
    parser.add_argument("--min-reward", type=float, default=None,
                        help="评估奖励低于该值触发告警")
    parser.add_argument("--no-notify", action="store_true", help="关闭桌面通知")
    parser.add_argument("--no-color", action="store_true", help="关闭 ANSI 颜色")
    args = parser.parse_args()

    prev_alive: dict[str, bool] = {}
    prev_reward: dict[str, float] = {}
    prev_timesteps: dict[str, tuple[float, float]] = {}
    last_sent: dict[str, float] = {}

    while True:
        now = time.time()
        res = resource_snapshot()
        states = discover_seeds(args, now)

        for st in states:
            if st.speed is None:
                prev = prev_timesteps.get(st.key)
                if prev:
                    t0, steps0 = prev
                    dt = now - t0
                    if dt > 0 and st.timesteps > steps0:
                        st.speed = (st.timesteps - steps0) / dt
            if st.speed and st.speed > 0:
                st.eta = max(0.0, (args.target - st.timesteps) / st.speed)

        alerts = evaluate_alerts(
            states, prev_alive, prev_reward, args.min_reward, now
        ) + resource_alerts(res)
        tasks = aggregate_tasks(states, args.seeds, args.target)
        if args.json:
            print(to_json(states, tasks, res, alerts), flush=True)
            return

        alerts = throttle_alerts(alerts, last_sent, now)
        for kind, key, msg in alerts:
            notify_desktop(f"RL 训练告警 [{kind}] {key}", msg, not args.no_notify)

        if not args.once:
            print("\033[2J\033[H", end="")
        print(render(states, tasks, res, alerts, args), flush=True)
        if args.once:
            return

        prev_alive = {st.key: st.alive for st in states}
        prev_reward = {
            st.key: r
            for st in states
            if (r := st.eval_reward if st.eval_reward is not None else st.tb_reward) is not None
        }
        prev_timesteps = {st.key: (now, st.timesteps) for st in states}
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
