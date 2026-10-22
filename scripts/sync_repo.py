#!/usr/bin/env python3
"""Keep local ``main`` fast-forwarded to ``origin/main``.

This is the "live sync" helper for the repository.  It is safe by
construction -- it only ever *fast-forwards*:

- it does nothing unless the current branch is ``main``;
- it does nothing when ``main`` has uncommitted changes;
- it never rebases, never merges non-linearly, never force-pushes;
- a diverged branch (local commits not on the remote) is reported, not touched.

Usage::

    python scripts/sync_repo.py            # sync once, human readable
    python scripts/sync_repo.py --status   # report only (still fetches)
    python scripts/sync_repo.py --quiet    # for scheduled runs; appends to a log
    python scripts/sync_repo.py --prune-merged-branches

Exit codes: 0 = in sync / synced / nothing to do, 1 = sync blocked or failed.
"""

from __future__ import annotations

import argparse
import datetime as dt
import pathlib
import shutil
import subprocess
import sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
DEFAULT_LOG = PROJECT_ROOT / "logs" / "autosync.log"
SYNC_BRANCH = "main"
REMOTE = "origin"


class GitError(RuntimeError):
    pass


def find_git() -> str:
    """Locate the git executable (scheduled runs may have a minimal PATH)."""
    found = shutil.which("git")
    if found:
        return found
    for candidate in (
        pathlib.Path(r"C:\Program Files\Git\cmd\git.exe"),
        pathlib.Path(r"C:\Program Files\Git\bin\git.exe"),
        pathlib.Path(r"C:\Program Files (x86)\Git\cmd\git.exe"),
    ):
        if candidate.is_file():
            return str(candidate)
    raise GitError("git executable not found (install Git or add it to PATH)")


def run_git(git: str, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        [git, *args],
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        raise GitError(
            f"git {' '.join(args)} failed (rc={result.returncode}): "
            f"{(result.stderr or result.stdout).strip()}"
        )
    return result


def current_branch(git: str) -> str:
    return run_git(git, "rev-parse", "--abbrev-ref", "HEAD").stdout.strip()


def is_dirty(git: str) -> bool:
    return bool(run_git(git, "status", "--porcelain").stdout.strip())


def counts(git: str, local_ref: str, remote_ref: str) -> tuple[int, int]:
    """Return ``(ahead, behind)`` for ``local_ref...remote_ref``."""
    out = run_git(git, "rev-list", "--left-right", "--count", f"{local_ref}...{remote_ref}").stdout
    parts = out.split()
    if len(parts) != 2:
        raise GitError(f"unexpected rev-list output: {out!r}")
    ahead, behind = int(parts[0]), int(parts[1])
    return ahead, behind


def prune_merged_branches(git: str) -> list[str]:
    """Delete local branches already merged into ``main``; return their names."""
    deleted: list[str] = []
    for line in run_git(git, "branch", "--merged", SYNC_BRANCH, "--format=%(refname:short)").stdout.splitlines():
        name = line.strip()
        if not name or name == SYNC_BRANCH or name.startswith("("):
            continue
        result = run_git(git, "branch", "-d", name, check=False)
        if result.returncode == 0:
            deleted.append(name)
        # an undeletable branch is not an error: it simply stays, unpruned
    return deleted


def push_current_branch(git: str) -> list[str]:
    """Push the current branch to its upstream when it is strictly ahead."""
    branch = current_branch(git)
    upstream = run_git(
        git, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", check=False
    )
    if upstream.returncode != 0 or not upstream.stdout.strip():
        return [f"push skipped: '{branch}' has no upstream branch"]
    upstream_ref = upstream.stdout.strip()
    ahead, behind = counts(git, branch, upstream_ref)
    if ahead == 0:
        return []
    if behind:
        return [f"push skipped: '{branch}' is behind {upstream_ref}"]
    run_git(git, "push", REMOTE, branch)
    short = run_git(git, "rev-parse", "--short", "HEAD").stdout.strip()
    return [f"pushed {ahead} commit(s) on '{branch}' (HEAD {short})"]


def sync(status_only: bool, prune: bool, push: bool = False) -> tuple[bool, list[str]]:
    """Perform one sync pass.  Returns ``(ok, notes)``."""
    notes: list[str] = []
    git = find_git()

    branch = current_branch(git)
    if branch != SYNC_BRANCH:
        notes.append(f"pull skipped: current branch is '{branch}', not '{SYNC_BRANCH}'")
        if push and not status_only:
            notes.extend(push_current_branch(git))
        return True, notes

    if is_dirty(git):
        notes.append("pull skipped: working tree has uncommitted changes")
        if push and not status_only:
            notes.extend(push_current_branch(git))
        return False, notes

    run_git(git, "fetch", "--prune", REMOTE)
    ahead, behind = counts(git, SYNC_BRANCH, f"{REMOTE}/{SYNC_BRANCH}")

    if ahead and behind:
        local_only = run_git(
            git, "log", "--oneline", f"{REMOTE}/{SYNC_BRANCH}..{SYNC_BRANCH}", "-5"
        ).stdout.strip().splitlines()
        notes.append(
            f"diverged: {ahead} local commit(s) not on {REMOTE}/{SYNC_BRANCH}, "
            f"{behind} remote commit(s) not local -- not touched"
        )
        notes.extend(f"  local only: {line}" for line in local_only)
        return False, notes

    if ahead:
        if push and not status_only:
            notes.extend(push_current_branch(git))
        else:
            notes.append(f"ahead: {ahead} local commit(s) not pushed")
        return True, notes

    if behind:
        if status_only:
            notes.append(f"behind: {behind} commit(s) available (status only, not pulled)")
            return True, notes
        before = run_git(git, "rev-parse", "--short", "HEAD").stdout.strip()
        run_git(git, "merge", "--ff-only", f"{REMOTE}/{SYNC_BRANCH}")
        after = run_git(git, "rev-parse", "--short", "HEAD").stdout.strip()
        subjects = run_git(
            git, "log", "--oneline", f"{before}..{after}"
        ).stdout.strip().splitlines()
        notes.append(f"fast-forwarded {before} -> {after} ({len(subjects)} commit(s))")
        notes.extend(f"  {line}" for line in subjects[:10])
    else:
        notes.append("up to date")

    if prune and not status_only:
        deleted = prune_merged_branches(git)
        if deleted:
            notes.append("deleted merged local branch(es): " + ", ".join(deleted))

    if push and not status_only:
        notes.extend(push_current_branch(git))

    return True, notes


def write_log(path: pathlib.Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().astimezone().isoformat(timespec="seconds")
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(f"{stamp} {line}\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--status", action="store_true", help="report ahead/behind without pulling"
    )
    parser.add_argument(
        "--quiet", action="store_true", help="append output to logs/autosync.log"
    )
    parser.add_argument(
        "--prune-merged-branches",
        action="store_true",
        help="delete local branches already merged into main",
    )
    parser.add_argument(
        "--push",
        action="store_true",
        help="also push the current branch to its upstream when strictly ahead",
    )
    parser.add_argument(
        "--log", type=pathlib.Path, default=DEFAULT_LOG, help="log file for --quiet"
    )
    args = parser.parse_args(argv)

    try:
        ok, notes = sync(
            status_only=args.status,
            prune=args.prune_merged_branches,
            push=args.push,
        )
    except GitError as exc:
        notes = [f"error: {exc}"]
        ok = False

    if args.quiet:
        # keep the scheduled-run log readable: only record actual changes/problems
        meaningful = [note for note in notes if note.strip() != "up to date"]
        if meaningful:
            write_log(args.log, meaningful)
    else:
        print(f"[sync] {PROJECT_ROOT}")
        for note in notes:
            print(f"  {note}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
