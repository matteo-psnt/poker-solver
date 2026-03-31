"""The `legs` subcommand: what happened to every leg, including the silent deaths.

``jobs`` reads Batch directly, so it shows only what Batch still retains -- and a
task's record ages out while the run it belonged to lives on. This reads the
durable copy on the share instead, then asks Batch about the legs the share
cannot explain.

Neither side can answer alone. ``run_leg.sh`` writes its own account on entry and
from its EXIT trap, covering every death the shell survives and distinguishing a
hang from an OOM from a cancellation -- Batch reports all three as ``failure``.
The deaths it cannot cover (OOM-kill, SIGKILL, node loss, task-level wall clock)
leave a leg stuck at ``started``, because the trap never ran; only Batch can
explain those.
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path
from typing import Any

from src.interfaces.cli.commands import jobs
from src.interfaces.cli.commands._base import Command
from src.interfaces.cloud import batch, share
from src.interfaces.cloud.config import CloudConfig
from src.shared import leg_log


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run legs`."""
    parser.add_argument(
        "--skip-reconcile",
        action="store_true",
        help="Read the share without asking Batch about unresolved legs.",
    )
    parser.add_argument(
        "--legs-dir",
        default=None,
        help="Read a local copy instead of the share (see `fetch`). Implies --skip-reconcile.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Show only the last N attempts (0 = all, the default). Unlike `jobs`, "
        "this does NOT truncate by default: the row worth finding here is a death, "
        "and hiding old rows by default would hide exactly the ones being looked for.",
    )


def _result(rows: list[dict[str, Any]], reconciled: int | None, limit: int) -> dict[str, Any]:
    """One payload shape for both sources, newest last."""
    shown = rows[-limit:] if limit > 0 else rows
    return {
        "op": "legs",
        "rows": shown,
        "reconciled": reconciled,
        "hidden_rows": len(rows) - len(shown),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Join the node's account with Batch's, and report one row per attempt."""
    if args.legs_dir:
        return _result(leg_log.read_legs(Path(args.legs_dir)), None, args.limit)

    config = CloudConfig.load()
    service = share.share_client(config)
    with tempfile.TemporaryDirectory() as tmp:
        local = Path(tmp)
        _download_legs(service, config.share_name, local)

        reconciled = None
        # Only the legs with no terminal record are worth asking about; the module
        # decides which those are, so the criterion lives in one place rather than
        # being re-derived from a rendered table.
        if not args.skip_reconcile and leg_log.unresolved_task_ids(local):
            # Batch's vocabulary is translated HERE, not in leg_log: the record
            # module is stdlib-only shared code that the node imports, and
            # `observed_cause` compares against bare `completed`/`success`. A raw
            # `BatchTaskState.COMPLETED` matches neither, so every reconciled leg
            # would read as its own state string instead of an outcome.
            tasks = [
                {
                    **task,
                    "job": job["job"],
                    "state": jobs.short_state(task.get("state")),
                    "result": jobs.short_state(task.get("result")) or None,
                }
                for job in batch.list_jobs_with_tasks(batch.client(config))
                for task in job.get("tasks", [])
            ]
            explained = leg_log.reconcile(local, tasks)
            _upload_observed(service, config.share_name, local, explained)
            reconciled = len(explained)

        return _result(leg_log.read_legs(local), reconciled, args.limit)


def _download_legs(service: Any, share_name: str, local: Path) -> None:
    """Pull the whole legs/ directory: the join needs every record."""
    target = leg_log.legs_dir(local)
    target.mkdir(parents=True, exist_ok=True)
    for path in share.walk_files(service, share_name, leg_log.LEGS_DIRNAME):
        share.download_file(service, share_name, path, target / Path(path).name)


def _upload_observed(service: Any, share_name: str, local: Path, explained: list[str]) -> None:
    """Push back only the observer records.

    The node owns the other half and must never be overwritten from here -- one
    writer per file is what makes this safe on a share with no atomic rename.
    """
    for task_id in explained:
        name = f"{task_id}{leg_log.OBSERVED_SUFFIX}"
        body = (leg_log.legs_dir(local) / name).read_text()
        share.write_text(service, share_name, f"{leg_log.LEGS_DIRNAME}/{name}", body)


def render(payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    reconciled = payload.get("reconciled")
    if reconciled:
        print(f"Asked Batch about {reconciled} leg(s) the share could not explain.")
    print(leg_log.format_table(rows))
    if payload.get("hidden_rows"):
        print(f"  {payload['hidden_rows']} earlier attempt(s) hidden — show with --limit 0")


COMMAND = Command(
    name="legs",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Per-leg outcomes from the share, reconciled against Batch.",
)
