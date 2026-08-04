"""Staging and queueing: the one path every submission takes.

``submit``, ``score`` and ``repair-ladder`` differ only in the legs they build.
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
from src.interfaces.cloud.spec import LegSpec
from src.interfaces.errors import CommandError
from src.shared import gitinfo

NONCE_CEILING = 32768


@dataclass(frozen=True)
class Queued:
    """One task that reached the pool."""

    task_id: str
    job_id: str
    label: str


def stage_and_queue(
    make_legs: Callable[[str], list[LegSpec]], *, root: Path = Path()
) -> dict[str, Any]:
    """Snapshot the tree, open a job, and queue every leg against that snapshot.

    The snapshot is taken HERE rather than left as a step the caller must
    remember, because the property it provides is not optional: a leg executes
    the tree as it stood when it was submitted, and a later push must not
    change what an in-flight job is running. ``make_legs`` receives the
    snapshot id so a caller can put it inside its specs.

    One task per leg, never one task looping over them: legs are independent,
    so Batch is the scheduler and spreads them across nodes up to ``max_nodes``.
    A single looping task pins the whole set to one node however much pool is
    available.
    """
    config = CloudConfig.load()
    now = spec.utcnow()

    # VALIDATED BEFORE ANYTHING IS UPLOADED. Staging first would mean a
    # rejected submission -- `--to 0`, a `--set` missing its `=` -- still
    # leaving a full tarball on the share, permanently, for a leg that never
    # ran. The specs are built against a placeholder id purely to check them;
    # the real snapshot id is substituted below.
    for leg in make_legs("unvalidated"):
        leg.validate()

    snapshot = share.publish_code_snapshot(share.share_client(config), config.share_name, root, now)
    legs = [_stamped(leg) for leg in make_legs(snapshot)]
    if not legs:
        raise CommandError("Nothing to submit.")

    client = batch.client(config)
    job_id = batch.ensure_job(client, config.pool_id, now)
    queued = []
    # The index, not only a random nonce, separates ids within one submission.
    # Every leg here shares a single `now`, so a 30-rung score relied entirely
    # on 30 draws from a 32k space avoiding a collision (~1.4%) -- and a
    # collision raises mid-loop, after some rungs are already queued and with
    # no record of which.
    for index, leg in enumerate(legs):
        nonce = index * NONCE_CEILING + secrets.randbelow(NONCE_CEILING)
        task = spec.task_id(leg.label, now, nonce)
        # A precompute is NOT retried. Everything else here is cheap to repeat --
        # training resumes from its last published rung, scoring is idempotent --
        # but a precompute has no partial-progress marker (`metadata.json` is
        # written only on success), so a retry restarts the whole enumeration.
        # A deterministic failure would then bill three full runs to fail three
        # times, which is the one shape where the default retry is a liability.
        retries = 0 if leg.op == spec.PRECOMPUTE else batch.TASK_RETRIES
        batch.submit_leg(client, job_id, task, leg, retries=retries)
        queued.append(Queued(task_id=task, job_id=job_id, label=leg.label))

    return {
        "code_snapshot": snapshot,
        "job_id": job_id,
        "tasks": [item.task_id for item in queued],
    }


def _stamped(leg: LegSpec) -> LegSpec:
    """Attach this machine's git provenance to a leg.

    HERE rather than in each caller, for the reason this module exists: a
    property that every submission must have cannot be a step three callers
    have to remember. It is applied after ``make_legs`` so a caller cannot
    forget it, and cannot get it wrong either.

    The snapshot id above is strictly MORE precise -- it names the actual bytes,
    dirty tree and all -- but nothing in the record has a field for it, while
    `train_git_commit`/`eval_git_commit` have had columns since the ledger
    existed and have been null on every cloud row.
    """
    return replace(
        leg,
        git_commit=gitinfo.get_git_commit() or "",
        git_dirty=gitinfo.encode_dirty(gitinfo.is_git_dirty()),
    )


def render_queued(payload: dict[str, Any]) -> None:
    """Shared human rendering for a dispatch result."""
    print(f"  code snapshot: {payload['code_snapshot']}")
    print(f"  job:           {payload['job_id']}")
    for task in payload["tasks"]:
        print(f"  queued:        {task}")
    count = len(payload["tasks"])
    print(f"\n  {count} task(s) queued — walk away; watch with: poker-solver jobs")
