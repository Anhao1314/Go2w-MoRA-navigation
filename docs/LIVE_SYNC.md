# Live sync (local <-> GitHub)

Keep this clone and `origin` in sync without manual `git pull` / `git push`:

- **pull**: `scripts/sync_repo.py` fast-forwards local `main` to
  `origin/main`. It only fast-forwards -- it refuses dirty trees and diverged
  branches, and never rebases, merges non-linearly, or force-pushes.
- **push**: the `.githooks/post-commit` hook pushes the current branch right
  after every commit (only when strictly ahead of its upstream).
- **schedule**: a Windows scheduled task runs the same helper every 2 minutes,
  so remote merges made on GitHub show up locally without any action.

Everything is reversible; see "Uninstall" below.

## Install

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_live_sync.ps1
```

This sets, for this clone only:

- `core.hooksPath = .githooks` (commit-time push)
- `pull.ff = only` (a pull can never create a merge commit)
- a scheduled task `go2w-locomotion-live-sync` (every 2 minutes)

## Uninstall

```powershell
powershell -ExecutionPolicy Bypass -File scripts/install_live_sync.ps1 -Remove
```

## Manual use

```bash
python scripts/sync_repo.py                 # sync now, human readable
python scripts/sync_repo.py --status        # report ahead/behind only
python scripts/sync_repo.py --push          # also push current branch
python scripts/sync_repo.py --prune-merged-branches
```

## Log

Scheduled and hook runs write only meaningful events (syncs, pushes, refusals)
to `logs/autosync.log` (gitignored). Check it if something looks stale.

## Boundaries

- It never touches uncommitted changes, never discards work, never force-pushes.
- A diverged `main` (local commits not on the remote) is reported and left alone.
- The scheduled task runs in your user session; it does not need admin rights.
- CI status is not part of sync: pushing to `main` here is direct, as the
  repository has no branch protection. Use PRs when you want CI gating.
