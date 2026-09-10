"""The `prune-checkpoints` subcommand: drop retained rungs a run no longer needs.

A run keeps a LADDER of `static-<iteration>.zarr` snapshots because a ladder is
the only way to find a sampling trainer's best point. Once an arm's science is
settled the intermediate rungs are dead weight, and they dominate the share:
1,550 of them across 263 runs, ~1,098 GiB, against 26 MB of the JSON record that
every reader actually opens.

DRY RUN BY DEFAULT. `--apply` is the only thing that deletes, and the plan it
executes is the one printed without it.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from src.adapters.postgres import connect, queries
from src.interfaces.commands._base import Command, records_root, resolve_run_dir
from src.interfaces.errors import CommandError
from src.shared import records
from src.shared.cloudtask.node import archive

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    from src.shared.ports.record import RecordSource

GB = 1024**3

# `static-<iteration>` under either extension. The iteration is what orders a
# ladder; a name that does not carry one is not a rung this command knows how
# to reason about, and is therefore never a candidate.
#
# BOTH SPELLINGS, because the share carries whichever name a rung was published
# under. Fixed on `.zarr` this matched nothing for any run written after the
# format changed, so those runs were never pruned at all.
_RUNG = re.compile(
    rf"^static-(\d+)(?:{re.escape(records.LEGACY_SNAPSHOT_SUFFIX)}"
    rf"|{re.escape(records.SNAPSHOT_SUFFIX)})$"
)

# Parallel by round trip, not by bytes: a snapshot is thousands of tiny chunk
# files and Azure Files deletes them one at a time.
_PARALLEL_DELETES = 64


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver prune-checkpoints`."""
    parser.add_argument(
        "--keep",
        type=int,
        default=3,
        help="Newest rungs to keep per run (default 3). The latest is always kept.",
    )
    parser.add_argument(
        "--run",
        # `dest` is NOT `run`: `records_root` reads `args.run` to scope its pull
        # to ONE run, and a repeatable flag hands it a list. `run()` scopes the
        # pull itself when exactly one is named -- the whole record was cheap to
        # materialise when this was written and is now 331 manifest downloads.
        dest="runs",
        action="append",
        default=None,
        help="Limit to this run; repeatable. Omit to consider every published run.",
    )
    parser.add_argument(
        "--no-price",
        dest="price",
        action="store_false",
        help="Skip sizing the plan. Sizing is one HEAD per affected run against the container.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete. Without it this prints the plan and touches nothing.",
    )


class PrunePlan(BaseModel):
    """What would be removed, per run, and what it costs to remove it."""

    op: Literal["prune-checkpoints"] = "prune-checkpoints"
    applied: bool = False
    runs_considered: int = 0
    runs_affected: int = 0
    rungs_dropped: int = 0
    files_deleted: int = 0
    objects_deleted: int = 0
    freed_gib: float = 0.0
    protected: list[str] = []
    plan: list[dict[str, Any]] = []


def _scored_iterations(run_dir: Path) -> set[int]:
    """Rungs a LEGACY eval document names, for runs scored before the record.

    `evals/*.json` stopped being written when the sink became the database, so
    this answers for old runs only and the record answers for the rest. It was
    the whole answer once, and a run scored only in the database lost the rungs
    its scores name.
    """
    found: set[int] = set()
    evals = run_dir / "evals"
    if not evals.is_dir():
        return found
    for path in evals.glob("*.json"):
        try:
            row = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        iteration = row.get("checkpoint_iteration")
        if isinstance(iteration, int):
            found.add(iteration)
    return found


def _is_terminal(run_dir: Path, source: RecordSource | None) -> bool:
    """Whether the run has stopped writing, defaulting to NO.

    Through `RunMetadata.load`, which is the ONE place that knows all three
    layouts -- the source, the event log, and the `.run.json` a run written
    before the log carries. This folded the events itself and therefore could
    not see a legacy run's status at all: `tail_value` handed back its
    `running` default, and 9 completed runs holding 44 rungs were protected as
    "still running" forever. `reconcile-runs._status_of` already asked the
    question this way; two answers to "is this run finished" is one too many.

    The scoping matters and comes free with the fold: a run still training
    publishes new rungs and an ATTEMPT that ended `died` under it must not read
    as the run's own terminal state -- that is the direction that deletes a live
    ladder. An unreadable record protects rather than prunes.
    """
    from src.pipeline.training.run_tracker.metadata import RunMetadata  # noqa: PLC0415

    try:
        status = RunMetadata.load(run_dir, source).status or "running"
    except (OSError, ValueError, KeyError):
        return False
    # `abandoned` is `reconcile-runs`' word for a run whose last task exited
    # cleanly and was never continued. It stopped, which is all this asks.
    return status in {"completed", "failed", "cancelled", "abandoned"}


def _published_rungs(run_dir: Path) -> dict[int, str]:
    """Iteration -> the snapshot NAME its completion marker names.

    The markers are the run's own answer and the manifest is not: pruning
    removes a snapshot without rewriting the manifest that advertises it, which
    is the disagreement `verify_published_rungs` exists to absorb.

    The name as well as the iteration, because a rung is `static-N.zarr` before
    the format change and `static-N.ckpt.zst` after it. Rebuilding a spelling
    at each of the three places that delete or price one is how a rung gets
    reported and then not removed.
    """
    found: dict[int, str] = {}
    for path in run_dir.glob(f"{archive.MARKER_PREFIX}static-*"):
        name = path.name[len(archive.MARKER_PREFIX) :]
        match = _RUNG.match(name)
        if match:
            found[int(match.group(1))] = name
    return found


def run(args: argparse.Namespace) -> PrunePlan:
    """Decide which rungs to drop, and drop them only under `--apply`."""
    if args.keep < 1:
        raise CommandError("--keep must be at least 1: a run always keeps its latest rung.")

    from src.interfaces.cloud.config import CloudConfig  # noqa: PLC0415 -- Azure only when applying
    from src.interfaces.cloud.store import blob  # noqa: PLC0415

    source = connect.record_source_from_environment()
    engine = connect.engine_from_environment()
    scored_by_run = queries.scored_rungs(engine)
    plan = PrunePlan(applied=bool(args.apply))
    # ONE named run scopes the pull; several still filter a whole-record tree,
    # because the alternative is a listing per run against a store in another
    # country.
    scoped = args.runs[0] if args.runs and len(args.runs) == 1 else None
    with records_root(args, run=scoped) as root:
        wanted = (
            [resolve_run_dir(name, str(root)) for name in args.runs]
            if args.runs
            else sorted(p for p in root.iterdir() if p.is_dir())
        )
        plan.runs_considered = len(wanted)
        # Every TERMINAL run considered, whether or not it still has a rung to
        # drop: the litter below lives in runs that have already been pruned.
        swept: list[str] = []

        for run_dir in wanted:
            published = _published_rungs(run_dir)
            rungs = sorted(published)
            if not rungs:
                continue
            if not _is_terminal(run_dir, source):
                plan.protected.append(f"{run_dir.name}: still running")
                continue
            swept.append(run_dir.name)
            scored = _scored_iterations(run_dir) | scored_by_run.get(run_dir.name, set())
            keep = set(rungs[-args.keep :]) | scored
            drop = [rung for rung in rungs if rung not in keep]
            if not drop:
                continue
            plan.runs_affected += 1
            plan.rungs_dropped += len(drop)
            plan.plan.append(
                {
                    "run": run_dir.name,
                    "held": len(rungs),
                    # The ITERATIONS, not just how many: `--apply` executes this
                    # list rather than recomputing it, so what is printed and
                    # what is deleted cannot diverge.
                    "drop": drop,
                    # The NAMES beside the iterations, so `--apply` deletes what
                    # was printed rather than rebuilding a spelling for it.
                    "snapshots": [published[iteration] for iteration in drop],
                    "dropping": len(drop),
                    "keeping": sorted(keep),
                    "scored_kept": sorted(scored & set(rungs)),
                }
            )

    # Priced by SAMPLING one snapshot per affected run, not by walking all 605:
    # rungs of a run are the same table at different iterations and vary by a few
    # percent, while across runs they vary 0.30-2.21 GiB -- so per-run is where
    # the accuracy is, and per-rung would be thousands of listings for it.
    config = CloudConfig.load()

    def _price(entry: dict[str, Any]) -> float:
        snapshot = entry["snapshots"][0]
        return blob.rung_size(config, entry["run"], records.object_name(snapshot)) / GB

    if plan.plan and args.price:
        with ThreadPoolExecutor(max_workers=32) as pool:
            each = list(pool.map(_price, plan.plan))
        for entry, gib in zip(plan.plan, each, strict=True):
            entry["gib_each"] = round(gib, 2)
            entry["gib_freed"] = round(gib * entry["dropping"], 1)
        plan.freed_gib = round(sum(e["gib_freed"] for e in plan.plan), 1)

    if not args.apply:
        return plan

    for entry in plan.plan:
        for snapshot in entry["snapshots"]:
            # THE ONLY DELETE THIS PROJECT PERFORMS against the container, and
            # the container is now the only store a rung is in. What stood here
            # was twice this long: a marker delete, a threaded file walk, an
            # empty-directory tidy and a self-healing sweep for litter a killed
            # sweep had left -- all of it the shape of removing thousands of
            # small files from an SMB mount that no longer holds any.
            if blob.delete_rung(config, entry["run"], records.object_name(snapshot)):
                plan.objects_deleted += 1
    return plan


def render(payload: PrunePlan) -> None:
    verb = "DELETED" if payload.applied else "would drop"
    print(f"{payload.runs_considered} runs considered, {payload.runs_affected} affected")
    for entry in payload.plan:
        print(
            f"  {entry['run']}: holds {entry['held']}, {verb} {entry['dropping']}"
            f" ({entry.get('gib_freed', 0):.0f} GiB), keeping {len(entry['keeping'])}"
        )
        if entry["scored_kept"]:
            print(f"      scored rungs kept: {entry['scored_kept']}")
    if payload.protected:
        print("\nprotected:")
        for line in payload.protected:
            print(f"  {line}")
    print(f"\nrungs {verb}: {payload.rungs_dropped}  ({payload.freed_gib:,.0f} GiB)")
    if payload.applied:
        # BOTH STORES, counted apart. A rung lives on the share as thousands of
        # chunk files or in the container as one object, so a single total
        # cannot say whether the container was reached -- and for a year it was
        # not reached at all.
        print(f"share files deleted:      {payload.files_deleted:,}")
        print(f"container objects deleted: {payload.objects_deleted:,}")
    else:
        print("DRY RUN -- nothing was deleted. Re-run with --apply to execute this plan.")


COMMAND = Command(
    name="prune-checkpoints",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Drop retained checkpoint rungs a settled run no longer needs (dry run by default).",
)
