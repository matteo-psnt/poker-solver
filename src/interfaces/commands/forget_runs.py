"""The `forget-runs` subcommand: remove whole runs from the record and the container.

`prune-checkpoints` thins a ladder; this drops the run. Three kinds of run have
nothing left to say and were 45% of the container when this was written:

* **old-game** -- trained before the big blind got its option behind a limp
  (`e9f80c0`, 2026-08-22). That commit changed the tree under a fixed action
  config, so every score taken before it is on a scale nothing current can be
  put beside. 163 runs, 392 GiB.
* **smoke** -- `quick_test` runs, which exist to prove a dispatch path works.
* **empty** -- runs the container holds no rung for: they died before their
  first checkpoint and can never be loaded.

A run still `running` on the record is never touched here -- a stale one goes
through `reconcile-runs` first. DRY RUN BY DEFAULT: `--apply` is the only thing
that deletes, and it executes exactly the plan printed without it. The record
row goes before the objects, and the manifest before the rungs: an interrupted
delete leaves bytes nobody references, never a reference to bytes that are gone.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from src.adapters.postgres import connect, queries
from src.interfaces import run_names
from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError
from src.shared.gitinfo import is_ancestor

if TYPE_CHECKING:
    import argparse
    from collections.abc import Callable, Iterable, Mapping

GB = 1024**3

# The big blind's option behind a limp. Runs whose code predates this are a
# different game -- see `test_tree_shape_is_pinned`, re-pinned the same day.
GAME_CHANGED_AT = "e9f80c0"
SMOKE_CONFIG = "quick_test"
LIVE = frozenset({"running", "queued"})

Reason = Literal["old-game", "smoke", "empty", "orphan", "named"]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver forget-runs`."""
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        default=None,
        help="Forget this run (fragment accepted); repeatable. Added to the plan "
        "on top of the standing reasons.",
    )
    parser.add_argument(
        "--only",
        choices=("old-game", "smoke", "empty", "orphan", "named"),
        action="append",
        default=None,
        help="Plan only runs with this reason; repeatable. Default: every reason.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually delete. Without it this prints the plan and touches nothing.",
    )


class ForgetPlan(BaseModel):
    """What would be forgotten, why, and what it frees."""

    op: Literal["forget-runs"] = "forget-runs"
    applied: bool = False
    runs_considered: int = 0
    plan: list[dict[str, Any]] = []
    by_reason: dict[str, dict[str, float]] = {}
    protected: list[str] = []
    unknown_lineage: list[str] = []
    freed_gib: float = 0.0
    rows_deleted: int = 0
    objects_deleted: int = 0


def plan_runs(
    rows: Iterable[Any],
    held: Mapping[str, Mapping[str, int]],
    *,
    new_game: Callable[[str | None], bool | None],
    named: Iterable[str] = (),
    only: Iterable[str] | None = None,
) -> ForgetPlan:
    """The decision, apart from the stores: which runs go, and for which reasons.

    ``rows`` are the record's runs, ``held`` the container's objects per run,
    ``new_game`` answers whether a commit includes the game change (None when
    the commit is unknown to this checkout, which protects rather than drops).
    """
    plan = ForgetPlan()
    wanted = set(only) if only else None
    named = set(named)
    seen: set[str] = set()
    for row in rows:
        seen.add(row.run_id)
        plan.runs_considered += 1
        objects = held.get(row.run_id, {})
        reasons: list[str] = []
        if row.run_id in named:
            reasons.append("named")
        current = new_game(row.git_commit)
        if current is None and row.run_id not in named:
            plan.unknown_lineage.append(row.run_id)
        elif current is False:
            reasons.append("old-game")
        if row.config_name == SMOKE_CONFIG:
            reasons.append("smoke")
        if not any(name != "STATIC_CHECKPOINT.json" for name in objects):
            reasons.append("empty")
        if wanted is not None:
            reasons = [reason for reason in reasons if reason in wanted]
        if not reasons:
            continue
        if row.status in LIVE:
            plan.protected.append(f"{row.run_id}: {row.status} -- reconcile-runs first")
            continue
        _add(plan, row.run_id, reasons, objects, status=row.status, config=row.config_name)
    for run_id, objects in held.items():
        if run_id in seen:
            continue
        if wanted is None or "orphan" in wanted:
            _add(plan, run_id, ["orphan"], objects, status="(no record)", config="")
    plan.freed_gib = round(sum(entry["gib"] for entry in plan.plan), 1)
    return plan


def _add(
    plan: ForgetPlan,
    run_id: str,
    reasons: list[str],
    objects: Mapping[str, int],
    *,
    status: str,
    config: str,
) -> None:
    gib = sum(objects.values()) / GB
    plan.plan.append(
        {
            "run": run_id,
            "reasons": reasons,
            "status": status,
            "config": config,
            # The NAMES, so `--apply` deletes what was printed.
            "objects": sorted(objects),
            "gib": round(gib, 2),
        }
    )
    tally = plan.by_reason.setdefault(reasons[0], {"runs": 0, "gib": 0.0})
    tally["runs"] += 1
    tally["gib"] = round(tally["gib"] + gib, 1)


def run(args: argparse.Namespace) -> ForgetPlan:
    """Plan against the live record and container; delete only under `--apply`."""
    from src.interfaces.cloud.config import CloudConfig  # noqa: PLC0415 -- Azure only here
    from src.interfaces.cloud.store import blob  # noqa: PLC0415

    engine = connect.engine_from_environment()
    rows = queries.describe_runs(engine, limit=100_000)
    ids = [row.run_id for row in rows]
    named: list[str] = []
    for fragment in args.runs or ():
        matches = run_names.matching(fragment, ids)
        if len(matches) > 1:
            raise CommandError(run_names.ambiguous_message(fragment, matches))
        if not matches:
            raise CommandError(run_names.unknown_message(fragment, ids))
        named.append(matches[0])

    config = CloudConfig.load()
    held = blob.run_objects(config)
    plan = plan_runs(
        rows,
        held,
        new_game=lambda commit: is_ancestor(GAME_CHANGED_AT, commit),
        named=named,
        only=args.only,
    )
    if not args.apply:
        return plan

    from src.adapters.postgres import forget  # noqa: PLC0415 -- the one write, applied only

    plan.applied = True
    for entry in plan.plan:
        # The CLAIM before the bytes: the record row first, then the manifest,
        # then the rungs. `delete_run` orders the last two itself.
        if forget.forget_run(engine, entry["run"]):
            plan.rows_deleted += 1
        plan.objects_deleted += blob.delete_run(config, entry["run"], entry["objects"])
    return plan


def render(payload: ForgetPlan) -> None:
    verb = "FORGOTTEN" if payload.applied else "would forget"
    print(f"{payload.runs_considered} runs on the record, {len(payload.plan)} {verb}")
    for reason, tally in sorted(payload.by_reason.items(), key=lambda item: -item[1]["gib"]):
        print(f"  {reason:<9} {int(tally['runs']):4d} runs  {tally['gib']:8,.1f} GiB")
    print()
    for entry in payload.plan:
        reasons = ",".join(entry["reasons"])
        print(
            f"  {entry['run'][:60]:<60} {entry['gib']:7.1f} GiB  "
            f"{len(entry['objects']):3d} obj  {entry['status']:<10} {reasons}"
        )
    if payload.protected:
        print("\nprotected (still live on the record):")
        for line in payload.protected:
            print(f"  {line}")
    if payload.unknown_lineage:
        print(
            f"\n{len(payload.unknown_lineage)} run(s) trained at a commit this checkout does "
            "not know; left alone (name them with --run to forget them):"
        )
        for run_id in payload.unknown_lineage:
            print(f"  {run_id}")
    print(f"\n{verb}: {len(payload.plan)} runs, {payload.freed_gib:,.1f} GiB")
    if payload.applied:
        print(f"record rows deleted:       {payload.rows_deleted:,}")
        print(f"container objects deleted: {payload.objects_deleted:,}")
    else:
        print("DRY RUN -- nothing was deleted. Re-run with --apply to execute this plan.")


COMMAND = Command(
    name="forget-runs",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Remove settled runs from the record and the container (dry run by default).",
)
