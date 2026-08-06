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

from src.interfaces.commands._base import Command, records_root, resolve_run_dir
from src.interfaces.errors import CommandError
from src.shared import run_events
from src.shared.cloudtask.node import archive

if TYPE_CHECKING:
    import argparse
    from pathlib import Path

GB = 1024**3

# `static-<iteration>.zarr`. The iteration is what orders a ladder; a name that
# does not carry one is not a rung this command knows how to reason about, and
# is therefore never a candidate.
_RUNG = re.compile(r"^static-(\d+)\.zarr$")

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
        # to ONE run, and a repeatable flag hands it a list. The whole record is
        # cheap to materialise and this filters it locally.
        dest="runs",
        action="append",
        default=None,
        help="Limit to this run; repeatable. Omit to consider every published run.",
    )
    parser.add_argument(
        "--no-price",
        dest="price",
        action="store_false",
        help="Skip sizing the plan. Sizing is one listing per affected run against the share.",
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
    freed_gib: float = 0.0
    protected: list[str] = []
    plan: list[dict[str, Any]] = []


def _scored_iterations(run_dir: Path) -> set[int]:
    """Rungs an eval document names. Deleting one makes its score unreproducible."""
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


def _is_terminal(run_dir: Path) -> bool:
    """Whether the run has stopped writing, defaulting to NO.

    A run still training publishes new rungs, and its newest may be mid-copy.
    Read through `run_events` rather than the file: a bare scan for `status`
    returns the last ATTEMPT's, and an attempt that ended `died` under a run
    still training would read as terminal here -- which is the direction that
    deletes a live ladder. `kind=STATUS` is what separates the two facts, and
    the default is `running` so an unreadable record protects rather than
    prunes.
    """
    try:
        events = run_events.read(run_dir)
    except (OSError, ValueError):
        return False
    status = run_events.tail_value(events, "status", "running", kind=run_events.STATUS)
    return status in {"completed", "failed", "cancelled"}


def _published_rungs(run_dir: Path) -> list[int]:
    """Iterations the SHARE holds, read from the completion markers.

    The markers are the share's own answer and the manifest is not: pruning
    removes a snapshot without rewriting the manifest that advertises it, which
    is the disagreement `verify_published_rungs` exists to absorb.
    """
    rungs = []
    for path in run_dir.glob(f"{archive.MARKER_PREFIX}static-*.zarr"):
        match = _RUNG.match(path.name[len(archive.MARKER_PREFIX) :])
        if match:
            rungs.append(int(match.group(1)))
    return sorted(rungs)


def run(args: argparse.Namespace) -> PrunePlan:
    """Decide which rungs to drop, and drop them only under `--apply`."""
    if args.keep < 1:
        raise CommandError("--keep must be at least 1: a run always keeps its latest rung.")

    from src.interfaces.cloud.config import CloudConfig  # noqa: PLC0415 -- Azure only when applying
    from src.interfaces.cloud.store import share  # noqa: PLC0415

    plan = PrunePlan(applied=bool(args.apply))
    with records_root(args) as root:
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
            rungs = _published_rungs(run_dir)
            if not rungs:
                continue
            if not _is_terminal(run_dir):
                plan.protected.append(f"{run_dir.name}: still running")
                continue
            swept.append(run_dir.name)
            scored = _scored_iterations(run_dir)
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
    service = share.share_client(config)

    def _price(entry: dict[str, Any]) -> float:
        snapshot = f"static-{entry['drop'][0]}.zarr"
        base = f"{share.ARCHIVE_DIR}/{entry['run']}/{snapshot}"
        entries = share.list_entries(service, config.share_name, base)
        total = sum(e.size or 0 for e in entries if not e.is_directory)
        for sub in (e.name for e in entries if e.is_directory):
            total += sum(
                e.size or 0
                for e in share.list_entries(service, config.share_name, f"{base}/{sub}")
                if not e.is_directory
            )
        return total / GB

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
        for iteration in entry["drop"]:
            snapshot = f"static-{iteration}.zarr"
            base = f"{share.ARCHIVE_DIR}/{entry['run']}/{snapshot}"
            # The marker FIRST, and this ordering was chosen the wrong way round
            # once. A marker is the share's own claim that a rung is complete --
            # `verify_published_rungs` reads exactly these -- so a sweep that
            # deletes bytes first and is then interrupted leaves a rung that
            # ADVERTISES itself and cannot load, and the failure surfaces on a
            # node after a snapshot upload and an allocation. Deleting the claim
            # first leaves orphaned bytes instead: invisible, harmless, and
            # swept by the next run of this command.
            share.delete_file(
                service,
                config.share_name,
                f"{share.ARCHIVE_DIR}/{entry['run']}/{archive.marker_for(snapshot)}",
            )
            paths = [path for path, _etag in share.walk_files(service, config.share_name, base)]
            with ThreadPoolExecutor(max_workers=_PARALLEL_DELETES) as pool:
                deleted = list(
                    pool.map(
                        lambda path: share.delete_file(service, config.share_name, path), paths
                    )
                )
            plan.files_deleted += sum(1 for ok in deleted if ok)
            # Azure Files keeps the directory when its files go, and an empty one
            # still costs the parent listing a name -- which is the walk every
            # metadata read pays per run.
            _remove_empty_tree(share, service, config.share_name, base)

    # SELF-HEALING, and the reason the ordering comment above can promise it:
    # a sweep interrupted before this existed left directories emptied of files
    # but not removed, and a later run has nothing left to drop so would never
    # revisit them. An empty snapshot directory with no marker is litter by
    # definition -- nothing advertises it and nothing can load it.
    for name in swept:
        run_base = f"{share.ARCHIVE_DIR}/{name}"
        listing = share.list_entries(service, config.share_name, run_base)
        claimed = {
            e.name[len(archive.MARKER_PREFIX) :]
            for e in listing
            if not e.is_directory and e.name.startswith(archive.MARKER_PREFIX)
        }
        for snapshot in (e.name for e in listing if e.is_directory and _RUNG.match(e.name)):
            if snapshot in claimed:
                continue
            # UNCLAIMED on a terminal run, so unloadable whatever it holds: the
            # marker is the definition of complete, and a reader is driven by
            # what the manifest names rather than by what a directory happens to
            # contain. Either our own interrupted sweep or a copy that died with
            # the run -- both are bytes nothing can reference.
            base = f"{run_base}/{snapshot}"
            paths = [path for path, _etag in share.walk_files(service, config.share_name, base)]
            if paths:
                with ThreadPoolExecutor(max_workers=_PARALLEL_DELETES) as pool:
                    swept_files = list(
                        pool.map(
                            lambda path: share.delete_file(service, config.share_name, path), paths
                        )
                    )
                plan.files_deleted += sum(1 for ok in swept_files if ok)
            _remove_empty_tree(share, service, config.share_name, base)
    return plan


def _remove_empty_tree(share: Any, service: Any, share_name: str, base: str) -> None:
    """Remove `base` and the directories under it, deepest first.

    A zarr snapshot nests one level, so the children have to go before the
    parent can. Anything still holding a file is simply left: this tidies, it
    does not decide.
    """
    children = [e.name for e in share.list_entries(service, share_name, base) if e.is_directory]
    for child in children:
        share.delete_directory(service, share_name, f"{base}/{child}")
    share.delete_directory(service, share_name, base)


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
        print(f"files deleted: {payload.files_deleted:,}")
    else:
        print("DRY RUN -- nothing was deleted. Re-run with --apply to execute this plan.")


COMMAND = Command(
    name="prune-checkpoints",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Drop retained checkpoint rungs a settled run no longer needs (dry run by default).",
)
