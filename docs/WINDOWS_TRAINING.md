# Windows Training Orchestration

All Go2W training is submitted from macOS to the Windows host through
`tools/win_train.py`. Jobs run detached from the SSH session, write logs and
metadata on the Windows side, and can be fetched back when finished.

## Host

| Item | Value |
| --- | --- |
| Host alias | `winbox` (LAN/Tailscale only) |
| Workspace | `D:/work/go2w` |
| Python | `D:/work/go2w/.venv/Scripts/python.exe` |
| CPU | 14 cores / 20 logical processors |
| GPU | RTX 5060 Ti 16 GB |
| Torch | 2.14.1+cu130 (CUDA 13.0) |

## Why jobs survive SSH

A plain `subprocess.Popen(..., DETACHED_PROCESS)` is still owned by the Windows
OpenSSH job object and is killed when the SSH channel closes. `win_job_runner.py`
therefore writes a small `launch.bat` and starts it through WMI
(`Invoke-CimMethod Win32_Process Create`). The resulting process belongs to the
WMI provider, not to `sshd`, and survives disconnects. This was verified with a
25-second detached job.

Job state is in `rl/runs/_jobs/<job-id>/`:

- `spec.json` — command, cwd, env, run directory, source revision
- `meta.json` — status, PIDs, timestamps, exit code
- `train.log` — stdout/stderr
- `launcher.log` — WMI/batch launcher diagnostics

## Device choice

SB3 prints a warning when PPO uses a GPU with `MlpPolicy`: the network is tiny
and MuJoCo environment stepping dominates. A same-seed multi-segment smoke run
measured:

| Device | Smoke wall time |
| --- | ---: |
| CPU | 106 s |
| CUDA | 156 s |

CPU is therefore the default (`--device cpu`). `--device cuda` remains
available for comparison or future CNN/vision policies.

## Commands

```bash
python tools/win_train.py sync
```

```bash
python tools/win_train.py submit --preset multi-segment --seed 0
```

```bash
python tools/win_train.py submit --preset junction --seed 0
```

```bash
python tools/win_train.py submit --preset high-level-curve --seed 0 \
  -- --total-steps 2000000 --envs 4
```

```bash
python tools/win_train.py submit --preset curve --seed 0
```

```bash
python tools/win_train.py list
```

```bash
python tools/win_train.py tail mseg-full-s0 -n 40
```

```bash
python tools/win_train.py fetch mseg-full-s0
```

```bash
python tools/win_train.py stop mseg-full-s0
```

Presets: `multi-segment`, `junction`, `high-level-curve`, `curve`, `ppo`
(requires `--task`), `sentinel`, `custom` (requires `--cmd`).

`submit` syncs the working tree first unless `--no-sync` is passed. The sync
excludes `.git`, `.venv`, `build`, `deps`, `rl/runs`, `media`, and `reports`;
`data/` and `models/` are included because the training scripts need the BC
checkpoints and MJCF assets.

## Run directories

Every job gets an exclusive directory:

```text
rl/runs/multi_segment/<job-id>/
rl/runs/junction/<job-id>/
rl/runs/high_level_curve/<job-id>/
rl/runs/traverse_curve_curriculum/<job-id>/
rl/runs/ppo/<job-id>/
```

The standard `start_run`/`--run-dir` guards are preserved, so a job never
overwrites an existing run.

## Training batch status (2026-10-19)

The four navigation pipelines were smoke-tested and then stopped for a
project-scope review; **no long training run is currently active**. The intended
batch is:

| Job | Preset | Steps | Device |
| --- | --- | ---: | --- |
| `mseg-full-s0` | multi-segment | 2.0M | CPU |
| `junction-full-s0` | junction | 1.8M | CPU |
| `hlc-full-s0` | high-level-curve | 2.0M | CPU |
| `curve-full-s0` | curve | 4×1.0M | CPU |

Lower-level `rl/train.py --task balance/full_chain` jobs are not part of this
batch; use the `ppo` preset to queue them after the navigation batch is
approved.

## Limits

- Code sync is a tar copy, not a git checkout; the Windows workspace has no
  `.git`. The job spec records the macOS commit and dirty-file count.
- The runner has no dependency-aware queue; concurrent jobs start immediately.
- Do not sync while jobs are running unless `--force` is intentional.
- WMI-launched processes run without an interactive desktop; training is
  headless and does not need one.
