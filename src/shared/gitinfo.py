"""Git provenance helpers for stamping runs and evaluations.

Every recorded number is only as reproducible as the code that produced it. Runs
are pinned to their *train* commit and evals to their *measure* commit (LBR
methodology has changed across commits — the scorer, the resolver-in-LBR replay
— so the eval commit is as load-bearing as the train commit). ``dirty`` records
whether the working tree had uncommitted changes at capture time, which turns a
bare commit hash from "probably this code" into "this code, verified clean".

Results are process-cached: the repo state does not change mid-run, and each
lookup shells out to git.

A cloud node has no checkout
----------------------------
The code snapshot excludes ``.git`` (``share.SNAPSHOT_EXCLUDES``), so on a Batch
node ``git rev-parse`` has nothing to answer from -- and **every cloud-trained
run and every cloud-run evaluation recorded a null commit** for as long as
training has been in the cloud. The submitter therefore stamps its own HEAD into
the task environment, and the values below are read from there FIRST.

That precedence is not a compromise. The tree on a node is an extracted tarball
whose provenance was decided when the snapshot was sealed; git's upward search
from that directory can only ever describe some *other* repository that happens
to be an ancestor on the node's filesystem. Where the environment speaks, it is
the only witness that was present at the decision.
"""

from __future__ import annotations

import os
import subprocess
from functools import lru_cache

from src.shared import repo

_REPO_ROOT = repo.ROOT

# Set on the submitting machine and inherited all the way down to the
# `uv run poker-solver` child, so nothing threads it through a command line.
COMMIT_ENV = "RUN_GIT_COMMIT"

# Three-state, unlike the other RUN_* booleans: ``1`` dirty, ``0`` verified
# clean, empty/absent unknown. Collapsing clean into unknown would lose the
# distinction the module docstring calls load-bearing.
DIRTY_ENV = "RUN_GIT_DIRTY"

# The branch is how a HUMAN names an experiment. Worktrees share a commit far
# more often than not -- one branches from main and carries its change
# UNCOMMITTED while it is iterated on -- so a hash plus a dirty bit cannot tell
# two apart and the branch can.
BRANCH_ENV = "RUN_GIT_BRANCH"

# Not a git fact, and the only answer that is COMPLETE: a commit plus a dirty bit
# says "these bytes, and some unrecorded changes", while the snapshot IS the
# bytes a task extracted and ran. Empty off a node, where the working tree is
# already described by the commit and dirty bit.
SNAPSHOT_ENV = "CODE_SNAPSHOT"


def _run_git(*args: str) -> str | None:
    """Run a git command in the repo root; return stripped stdout or None on failure."""
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


@lru_cache(maxsize=1)
def get_git_commit() -> str | None:
    """Full HEAD commit SHA, or None when nothing can vouch for one.

    The stamped value wins where present -- see the module docstring.
    """
    stamped = os.environ.get(COMMIT_ENV)
    if stamped:
        return stamped
    return _run_git("rev-parse", "HEAD") or None


@lru_cache(maxsize=1)
def is_git_dirty() -> bool | None:
    """True if the working tree had uncommitted changes, None if unknowable."""
    stamped = os.environ.get(DIRTY_ENV)
    if stamped:
        # "0" is a real answer, so this tests for a NON-EMPTY string rather than
        # for truthiness of the decoded value.
        return stamped == "1"
    status = _run_git("status", "--porcelain")
    if status is None:
        return None
    return bool(status)


@lru_cache(maxsize=1)
def get_git_branch() -> str | None:
    """The checked-out branch, or None when detached or unknowable.

    ``--abbrev-ref`` answers ``HEAD`` for a detached checkout, which is not a
    branch name and is reported as None rather than as the literal string: a
    record saying "branch HEAD" is worse than one saying nothing.
    """
    stamped = os.environ.get(BRANCH_ENV)
    if stamped:
        return stamped
    branch = _run_git("rev-parse", "--abbrev-ref", "HEAD")
    return None if not branch or branch == "HEAD" else branch


def get_code_snapshot() -> str | None:
    """The snapshot this process is running from, or None when it is not one.

    Read from the environment and nowhere else: only the submitting machine
    knows the id, the node receives it, and nothing on disk can reconstruct it.
    """
    return os.environ.get(SNAPSHOT_ENV) or None


def encode_dirty(dirty: bool | None) -> str:
    """The three-state encoding the environment carries."""
    if dirty is None:
        return ""
    return "1" if dirty else "0"


@lru_cache(maxsize=256)
def commits_ahead_of(commit: str | None) -> int | None:
    """How many commits HEAD is ahead of ``commit`` (0 when they are the same).

    Returns None when the age is unknowable: no commit recorded, the commit is
    absent from this checkout (trained on a branch/history that was never
    fetched, or rewritten), or git is unavailable. A run whose commit is *not* an
    ancestor of HEAD still yields the count of commits reachable from HEAD but not
    from it -- a sensible "distance" for the divergent case.
    """
    if not commit:
        return None
    # rev-list itself errors (-> None) on a sha absent from this checkout, so that
    # already separates "unknown" from a real zero distance -- no pre-check needed.
    count = _run_git("rev-list", "--count", f"{commit}..HEAD")
    if count is None:
        return None
    try:
        return int(count)
    except ValueError:
        return None
