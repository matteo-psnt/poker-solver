"""Git provenance helpers for stamping runs and evaluations.

Every recorded number is only as reproducible as the code that produced it. Runs
are pinned to their *train* commit and evals to their *measure* commit (LBR
methodology has changed across commits — the scorer, the resolver-in-LBR replay
— so the eval commit is as load-bearing as the train commit). ``dirty`` records
whether the working tree had uncommitted changes at capture time, which turns a
bare commit hash from "probably this code" into "this code, verified clean".

Results are process-cached: the repo state does not change mid-run, and each
lookup shells out to git.
"""

from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_git(*args: str) -> str | None:
    """Run a git command in the repo root; return stripped stdout or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


@lru_cache(maxsize=1)
def get_git_commit() -> str | None:
    """Full HEAD commit SHA, or None outside a git checkout."""
    commit = _run_git("rev-parse", "HEAD")
    return commit or None


@lru_cache(maxsize=1)
def is_git_dirty() -> bool | None:
    """True if the working tree has uncommitted changes, None if git is unavailable."""
    status = _run_git("status", "--porcelain")
    if status is None:
        return None
    return bool(status)
