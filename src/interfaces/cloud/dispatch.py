"""Staging and queueing: the one path every submission takes.

``submit`` and ``score`` differ only in the tasks they build.
They share this module so they cannot drift apart in how they stage code or
wire the environment -- the fresh and continuing paths diverging is exactly how
one of them silently stops being exercised.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from src.interfaces.cloud import batch, share, spec
from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.spec import TaskSpec
from src.interfaces.errors import CommandError
from src.shared import gitinfo
from src.shared.cloudtask import kinds

NONCE_CEILING = 32768

"""The tree that gets sealed into a snapshot: THIS checkout, not the working
directory.

`Path()` -- the previous default -- is wherever the shell happened to be, and no
caller has ever passed anything else. Submitting from `src/` would have tarred
`src/`'s children as the tree root, and from a sibling worktree's directory it
would have shipped that worktree's code under this one's provenance stamp. The
package's own location is the one answer that cannot be wrong, and it is the
same one `gitinfo` already resolves its git questions against."""
CHECKOUT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class Queued:
    """One task that reached the pool."""

    task_id: str
    job_id: str
    label: str


def stage_and_queue(
    make_tasks: Callable[[str], list[TaskSpec]], *, root: Path = CHECKOUT_ROOT
) -> dict[str, Any]:
    """Snapshot the tree, open a job, and queue every task against that snapshot.

    The snapshot is taken HERE rather than left as a step the caller must
    remember, because the property it provides is not optional: a task executes
    the tree as it stood when it was submitted, and a later push must not
    change what an in-flight job is running. ``make_tasks`` receives the
    snapshot id so a caller can put it inside its specs.

    One Batch task per unit of work, never one task looping over them: they are
    independent, so Batch is the scheduler and spreads them across nodes up to
    ``max_nodes``. A single looping task pins the whole set to one node however
    much pool is available.
    """
    config = CloudConfig.load()
    now = spec.utcnow()

    # VALIDATED BEFORE ANYTHING IS UPLOADED. Staging first would mean a
    # rejected submission -- `--to 0`, a `--set` missing its `=` -- still
    # leaving a full tarball on the share, permanently, for a task that never
    # ran. The specs are built against a placeholder id purely to check them;
    # the real snapshot id is substituted below.
    for task in make_tasks("unvalidated"):
        task.validate()

    snapshot = share.publish_code_snapshot(share.share_client(config), config.share_name, root, now)
    specs = [_stamped(task) for task in make_tasks(snapshot)]
    if not specs:
        raise CommandError("Nothing to submit.")

    client = batch.client(config)
    job_id = batch.ensure_job(client, config.pool_id, now)
    queued = []
    # The index, not only a random nonce, separates ids within one submission.
    # Every task here shares a single `now`, so a 30-rung score relied entirely
    # on 30 draws from a 32k space avoiding a collision (~1.4%) -- and a
    # collision raises mid-loop, after some rungs are already queued and with
    # no record of which.
    for index, task in enumerate(specs):
        nonce = index * NONCE_CEILING + secrets.randbelow(NONCE_CEILING)
        identifier = spec.task_id(task.label, now, nonce)
        # The KIND decides: work cheap to repeat wants retries, work with no
        # partial-progress marker does not. See `src.shared.cloudtask.kinds`.
        batch.submit_task(client, job_id, identifier, task, retries=kinds.kind(task.op).retries)
        queued.append(Queued(task_id=identifier, job_id=job_id, label=task.label))

    return {
        "code_snapshot": snapshot,
        "job_id": job_id,
        "tasks": [item.task_id for item in queued],
    }


def _stamped(task: TaskSpec) -> TaskSpec:
    """Attach this machine's git provenance to a task.

    HERE rather than in each caller, for the reason this module exists: a
    property that every submission must have cannot be a step three callers
    have to remember. It is applied after ``make_tasks`` so a caller cannot
    forget it, and cannot get it wrong either.

    Three answers, none of which replaces another. The commit says which
    history; the branch says which line of work, which is what a commit cannot
    say when several worktrees sit on the same one with uncommitted changes; and
    the snapshot id names the actual BYTES -- dirty tree and all -- so it is the
    only one of the three that is exact. The snapshot already travelled to the
    node as a fetch instruction and was recorded nowhere, which left the exact
    answer available and unasked for.
    """
    return replace(
        task,
        git_commit=gitinfo.get_git_commit() or "",
        git_dirty=gitinfo.encode_dirty(gitinfo.is_git_dirty()),
        git_branch=gitinfo.get_git_branch() or "",
    )


def render_queued(payload: dict[str, Any]) -> None:
    """Shared human rendering for a dispatch result."""
    print(f"  code snapshot: {payload['code_snapshot']}")
    print(f"  job:           {payload['job_id']}")
    for task in payload["tasks"]:
        print(f"  queued:        {task}")
    count = len(payload["tasks"])
    print(f"\n  {count} task(s) queued — walk away; watch with: poker-solver jobs")
