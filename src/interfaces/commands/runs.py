"""The `runs` subcommand: every published run, newest first.

Listing runs used to need no command: a caller globbed the local runs directory.
With the record living only on the share there is no directory to glob, and two
surfaces immediately needed one -- the interactive picker, which had started
reporting "no trained runs found" against a path that no longer exists, and the
console's `/runs` page.

Both go through here rather than reaching for the share themselves. That is the
same rule the rest of the command layer follows: one implementation per question.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from src.adapters.postgres import connect, queries
from src.interfaces.commands._base import Command, records_root
from src.pipeline import services
from src.pipeline.services.runs import RunSummary
from src.shared.gitinfo import commits_ahead_of

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver runs`."""
    parser.add_argument(
        "--limit", type=int, default=0, help="Show only the newest N runs (0 = all)."
    )
    parser.add_argument(
        "--loadable-only",
        action="store_true",
        help="Hide runs that never checkpointed. They are still worth listing by "
        "default: a run that died before its first checkpoint is exactly the one "
        "someone is looking for when they ask what happened.",
    )


class RunsPayload(BaseModel):
    """Every published run, newest first."""

    op: Literal["runs"] = "runs"
    runs: list[services.RunSummary] = []
    # WHICH STORE ANSWERED. A silent fallback would let a reader mistake a
    # stale answer for a fresh one, and during dual write the two can honestly
    # differ -- the database is ahead for a live run, behind for one whose task
    # predates the sink. Saying so costs one field.
    source: Literal["database", "share"] = "share"


def run(args: argparse.Namespace) -> RunsPayload:
    """Summarise every published run, newest first.

    From the database when one is configured, and from the share otherwise.
    The DSN is the same switch that turns on dual write, so a machine that
    writes rows reads them, and one that does not behaves exactly as before.
    """
    engine = connect.engine_from_environment()
    if engine is not None:
        summaries = _from_database(engine)
        source: Literal["database", "share"] = "database"
    else:
        with records_root(args) as root:
            summaries = services.describe_runs(root)
        source = "share"
    if args.loadable_only:
        summaries = [summary for summary in summaries if summary.loadable]
    if args.limit > 0:
        summaries = summaries[: args.limit]
    return RunsPayload(runs=summaries, source=source)


def _distances(commits: set[str | None]) -> dict[str | None, int | None]:
    """How far HEAD is ahead of each commit, resolved concurrently.

    `commits_ahead_of` spawns a git process per commit at ~24 ms, and 303 runs
    share only 72 distinct commits -- so the first win is asking once per
    COMMIT rather than once per run, and the second is not waiting for each
    answer before asking the next. Serial, those 72 cost 1.8 s and were the
    largest single cost in this listing, larger than the query.

    NOT a position lookup into one `git rev-list HEAD`, which is the obvious
    batch and is wrong: the count is a set difference, and with merge commits a
    commit's index in that list is not the number of commits reachable from
    HEAD but not from it. Same computation, concurrently.

    Per CALL rather than a module cache, because HEAD moves under a long-lived
    server and this is a fact about the checkout now.
    """
    known = {commit for commit in commits if commit}
    if not known:
        return dict.fromkeys(commits)
    with ThreadPoolExecutor(max_workers=min(16, len(known))) as pool:
        answers = list(pool.map(commits_ahead_of, known))
    resolved: dict[str | None, int | None] = dict(zip(known, answers, strict=True))
    resolved[None] = None
    return resolved


def _from_database(engine: Any) -> list[services.RunSummary]:
    """Rows into the model the surfaces already render.

    Built HERE rather than in the adapter: `RunSummary` lives in `pipeline`,
    and `an_adapter_does_not_do_the_work` forbids the adapter from importing
    it. The composition root is the only layer that may hold both.

    `commits_ago` stays a read-time computation against the local checkout,
    exactly as the share path computes it -- it is a fact about THIS working
    copy, not about the run, so storing it would be storing someone else's
    answer.
    """
    rows = queries.describe_runs(engine)
    ahead = _distances({row.git_commit for row in rows})
    summaries = []
    for row in rows:
        loadable = bool(row.has_checkpoint)
        summaries.append(
            services.RunSummary(
                name=row.run_id,
                commits_ago=ahead.get(row.git_commit),
                git_dirty=row.git_dirty,
                has_checkpoint=loadable,
                loadable=loadable,
                blocker=None if loadable else "no checkpoint",
                iterations=row.iterations,
                num_infosets=row.num_infosets,
                config_name=row.config_name,
                status=row.status,
                experiment_id=row.experiment_id,
                arm=row.arm,
            )
        )
    return summaries


def render(payload: RunsPayload) -> None:
    rows = payload.runs
    if not rows:
        print("No published runs.")
        return
    print(f"{len(rows)} published run(s), newest first")
    header = f"{'run':<34} {'config':<14} {'iterations':>13} {'status':<11} {'age':<16}"
    print(header)
    print("-" * len(header))
    for row in rows:
        commits = row.commits_ago
        age = "commit unknown" if commits is None else f"{commits} commit(s) ago"
        iterations = row.iterations
        print(
            f"{row.name[:34]:<34} "
            f"{(row.config_name or '—')[:14]:<14} "
            f"{(f'{iterations:,}' if iterations is not None else '—'):>13} "
            f"{(row.status or '—')[:11]:<11} "
            f"{age:<16}" + ("" if row.loadable else f"  ({row.blocker})")
        )


# `RunSummary` is RE-EXPORTED, not used here: `.importlinter` forbids
# `interfaces.web -> pipeline`, so a command module is the console's only legal
# route to a service model. Deleting it as an unused import breaks `contract.py`.
__all__ = ["COMMAND", "RunSummary", "add_arguments", "render", "run"]

COMMAND = Command(
    name="runs",
    help="Every published run, newest first.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
