"""The `tasks` subcommand: what happened to every task, including the silent deaths.

``jobs`` reads Batch directly, so it shows only what Batch still retains -- and a
task's record ages out while the run it belonged to lives on. This reads the
durable copy on the share instead, then asks Batch about the tasks the share
cannot explain.

Neither side can answer alone. The node wrapper writes its own account on entry and
from its EXIT trap, covering every death the shell survives and distinguishing a
hang from an OOM from a cancellation -- Batch reports all three as ``failure``.
The deaths it cannot cover (OOM-kill, SIGKILL, node loss, task-level wall clock)
leave a task stuck at ``started``, because the trap never ran; only Batch can
explain those.
"""

from __future__ import annotations

import argparse
import tempfile
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from src.interfaces.cloud import batch, share, workspace
from src.interfaces.cloud.config import CloudConfig
from src.interfaces.commands import jobs
from src.interfaces.commands._base import Command
from src.shared.cloudtask import task_log

# Round trips, not bytes: see `workspace._PARALLEL_DOWNLOADS`, measured there.
_PARALLEL_SHARE_IO = 64

# The shared-cache key for legs/. Its own, not the record's: they are different
# subtrees of the share, pulled by different readers.
_LEGS_KEY = "legs"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver tasks`."""
    parser.add_argument(
        "--skip-reconcile",
        action="store_true",
        help="Read the share without asking Batch about unresolved tasks.",
    )
    parser.add_argument(
        "--tasks-dir",
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
        "op": "tasks",
        "rows": shown,
        "reconciled": reconciled,
        "hidden_rows": len(rows) - len(shown),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Join the node's account with Batch's, and report one row per attempt."""
    if args.tasks_dir:
        return _result(task_log.read_tasks(Path(args.tasks_dir)), None, args.limit)

    config = CloudConfig.load()
    service = share.share_client(config)
    with _task_records(service, config.share_name) as local:
        reconciled = None
        # Only the tasks with no terminal record are worth asking about; the module
        # decides which those are, so the criterion lives in one place rather than
        # being re-derived from a rendered table.
        open_tasks = task_log.unresolved_tasks(local)
        if not args.skip_reconcile and open_tasks:
            explained = task_log.reconcile(local, _ask_batch(config, open_tasks))
            _upload_observed(service, config.share_name, local, explained)
            reconciled = len(explained)

        return _result(task_log.read_tasks(local), reconciled, args.limit)


def _translate(task: dict[str, Any]) -> dict[str, Any]:
    """Batch's vocabulary into this project's.

    Done HERE, not in task_log: the record module is stdlib-only shared code
    that the node imports, and `observed_cause` compares against bare
    `completed`/`success`. A raw `BatchTaskState.COMPLETED` matches neither, so
    every reconciled task would read as its own state string instead of an
    outcome.
    """
    return {
        **task,
        "state": jobs.short_state(task.get("state")),
        "result": jobs.short_state(task.get("result")) or None,
    }


def _ask_batch(config: CloudConfig, open_tasks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Ask Batch about exactly the tasks the share could not explain.

    One ``get_task`` per open question, concurrently, rather than listing every
    task of every job in the account -- which cost ~0.39s per job and scaled
    with history rather than with what was unexplained. A task record carries
    its own ``job_id``, so the pair is already known.

    A row with no ``job_id`` cannot be addressed this way and falls back to the
    old enumeration. That is not dead code: it covers records written before
    the field existed, and losing the explanation would be worse than the cost.
    """
    client = batch.client(config)
    pairs = {(row["job_id"], row["task_id"]) for row in open_tasks if row.get("job_id")}
    if len(pairs) < len({row["task_id"] for row in open_tasks}):
        listed = batch.attach_tasks(client, batch.list_jobs(client))
        return [_translate(task) for job in listed for task in job["tasks"]]

    with ThreadPoolExecutor(max_workers=min(16, len(pairs) or 1)) as pool:
        fetched = pool.map(lambda pair: batch.task_record(client, *pair), sorted(pairs))
    return [_translate(task) for task in fetched if task]


@contextmanager
def _task_records(service: Any, share_name: str) -> Iterator[Path]:
    """The legs/ directory, materialised locally for the duration.

    Shared with other readers when a :func:`workspace.shared_record_cache` is in
    force. `cost` is `tasks` plus arithmetic -- it invokes this very command --
    so a console showing both used to pull all 365 records twice, 22s each. The
    reconciled observations land in the shared tree too, which is what makes the
    second reader's write-back free rather than merely cheap.
    """
    cache = workspace.active_cache()
    if cache is None:
        with tempfile.TemporaryDirectory() as tmp:
            local = Path(tmp)
            _download_tasks(service, share_name, local)
            yield local
        return
    with cache.acquire(_LEGS_KEY, lambda root: _download_tasks(service, share_name, root)) as local:
        yield local


def _download_tasks(service: Any, share_name: str, local: Path) -> None:
    """Pull the whole legs/ directory: the join needs every record.

    Concurrently -- these are tiny JSON files at ~0.195s of round trip each, so
    serially they were 9.1s of almost pure latency against 1.1s parallel. The
    directory has since grown to 365 files and the width with it: latency is the
    entire cost, so the pool is sized against round trips, not bytes.
    """
    target = task_log.tasks_dir(local)
    target.mkdir(parents=True, exist_ok=True)
    paths = list(share.walk_files(service, share_name, task_log.RECORDS_DIRNAME))
    if not paths:
        return
    with ThreadPoolExecutor(max_workers=min(_PARALLEL_SHARE_IO, len(paths))) as pool:
        list(
            pool.map(
                lambda path: share.download_file(
                    service, share_name, path, target / Path(path).name
                ),
                paths,
            )
        )


def _upload_observed(service: Any, share_name: str, local: Path, explained: list[str]) -> None:
    """Push back only the observer records.

    The node owns the other half and must never be overwritten from here -- one
    writer per file is what makes this safe on a share with no atomic rename.

    Concurrently, and only for what `reconcile` found NEW. A write is ~2.4s of
    round trip; six of them, serially, on every read, was 14.1s spent restating
    what the share already said.
    """
    if not explained:
        return

    def publish(task_id: str) -> None:
        name = f"{task_id}{task_log.OBSERVED_SUFFIX}"
        body = (task_log.tasks_dir(local) / name).read_text()
        share.write_text(service, share_name, f"{task_log.RECORDS_DIRNAME}/{name}", body)

    with ThreadPoolExecutor(max_workers=min(_PARALLEL_SHARE_IO, len(explained))) as pool:
        list(pool.map(publish, explained))


def render(payload: dict[str, Any]) -> None:
    rows = payload["rows"]
    reconciled = payload.get("reconciled")
    if reconciled:
        print(f"Asked Batch about {reconciled} task(s) the share could not explain.")
    print(task_log.format_table(rows))
    if payload.get("hidden_rows"):
        print(f"  {payload['hidden_rows']} earlier attempt(s) hidden — show with --limit 0")


COMMAND = Command(
    name="tasks",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Per-task outcomes from the share, reconciled against Batch.",
)
