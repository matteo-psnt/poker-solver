"""The `ledger` subcommand: its flags, handler and renderer."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from src.interfaces.cli.commands._base import (
    Command,
    add_source_argument,
    ledger_for,
    records_root,
)
from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.training.run_tracker import migrate_run_log
from src.shared.config import DEFAULT_RUNS_DIR


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver-run ledger`."""
    add_source_argument(parser)
    parser.add_argument(
        "--ledger",
        default=str(eval_ledger.DEFAULT_LEDGER_PATH),
        help="Eval ledger path to read.",
    )
    parser.add_argument("--run", default=None, help="Filter to a single run id.")
    parser.add_argument("--experiment", default=None, help="Filter to one experiment id.")
    parser.add_argument(
        "--method", default=None, choices=["lbr", "exact_br"], help="Filter by method."
    )
    parser.add_argument(
        "--since", default=None, metavar="ISO8601", help="Only rows at or after this timestamp."
    )
    parser.add_argument(
        "--limit", type=int, default=25, help="Show only the last N rows (0 = all)."
    )
    parser.add_argument(
        "--runs-dir", default=DEFAULT_RUNS_DIR, help="Runs dir scanned by --rebuild."
    )
    parser.add_argument(
        "--migrate",
        action="store_true",
        help=(
            "Bring on-disk records up to the current layout: one document per "
            "evaluation, and an event log per run. Non-destructive."
        ),
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Regenerate the ledger from the per-run records on disk before listing. "
        "Recovers rows lost to concurrent writers; rows predating per-run records are "
        "preserved as-is, never dropped.",
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    """List recent eval-ledger rows as a compact table, optionally rebuilding first."""
    if getattr(args, "source", "local") == "share" and args.migrate:
        # `--source share` materialises a THROWAWAY copy, so migrating it wrote
        # consolidated documents into a temporary directory and then deleted it
        # -- while printing "Originals left in place", which reads as though the
        # published record had been migrated. Refusing beats a silent no-op that
        # reports success.
        raise SystemExit(
            "--migrate rewrites records in place, and --source share reads a temporary "
            "copy that is discarded. Nothing on the share would change.\n"
            "Migrating the published record is not wired up yet; run it against a local "
            "runs directory instead."
        )
    with records_root(args) as root:
        return _list(args, root)


def _list(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    ledger_path = ledger_for(args, root)
    migrated = None
    if args.migrate:
        migrated = eval_ledger.migrate_eval_files(root, ledger_path)
        # The run logs too: both are "bring what is on disk up to the current
        # layout", and doing them separately invites a half-migrated tree.
        runs_root = root
        migrated["run_logs"] = (
            sum(1 for d in sorted(runs_root.iterdir()) if d.is_dir() and migrate_run_log(d))
            if runs_root.is_dir()
            else 0
        )
    rebuilt = None
    if args.rebuild:
        recovered, preserved = eval_ledger.rebuild_ledger(root, ledger_path)
        rebuilt = {"recovered": recovered, "preserved": preserved}

    records = eval_ledger.read_records(ledger_path)
    if args.run:
        records = [r for r in records if r.get("run_id") == args.run]
    if args.experiment:
        records = [r for r in records if r.get("experiment_id") == args.experiment]
    if args.method:
        records = [r for r in records if r.get("method") == args.method]
    if args.since:
        # Instants, not strings: the ledger holds naive-local legacy rows beside
        # UTC-aware new ones, so a lexicographic cutoff skews by the writer's
        # offset — the exact defect `record_instant` exists to remove.
        cutoff = eval_ledger.record_instant({"timestamp": args.since})
        records = [r for r in records if eval_ledger.record_instant(r) >= cutoff]
    # `records[-0:]` is the whole list, so a 0 limit already meant "all" by accident.
    # Made deliberate: `--limit 0` is how a rebuild shows everything it recovered.
    if args.limit > 0:
        records = records[-args.limit :]
    return {
        "op": "ledger",
        # The path read, not the one asked for: under `--source share` the index
        # is derived into a temp dir.
        "ledger": str(ledger_path),
        "migrated": migrated,
        "rebuilt": rebuilt,
        "rows": records,
    }


def _fmt_commit(commit: str | None, dirty: bool | None) -> str:
    if not commit:
        return "—"
    short = commit[:7]
    if dirty:
        short += "-dirty"
    return short


def render(payload: dict[str, Any]) -> None:
    # Printed BEFORE the empty-rows early return: `just fetch` runs `--rebuild`
    # without --json, and the recovery counts are the entire point of that call.
    # A rebuild that found nothing and one that recovered 200 rows must not look
    # identical.
    migrated = payload.get("migrated")
    if migrated:
        print(
            f"Migrated {payload['ledger']}: {migrated['merged']} eval(s) merged from a "
            f"payload+record pair, {migrated['payload_only']} recovered from a payload "
            f"alone, {migrated['record_only']} from a record alone."
        )
        if migrated.get("run_logs"):
            print(f"  and {migrated['run_logs']} run(s) converted to an event log.")
        print("  Originals left in place — delete them once you are satisfied.")
    rebuilt = payload.get("rebuilt")
    if rebuilt:
        print(
            f"Rebuilt {payload['ledger']}: {rebuilt['recovered']} row(s) recovered "
            f"from per-run records, {rebuilt['preserved']} preserved (no record to "
            "rebuild from — pre-dating per-run records)."
        )

    rows = payload["rows"]
    if not rows:
        print(f"No eval-ledger entries in {payload['ledger']}.")
        return
    print(f"Eval ledger ({payload['ledger']}): {len(rows)} row(s)")
    header = (
        f"{'run_id':<26} {'commit':<14} {'scorer':<10} {'opp':<10} "
        f"{'seed':>12} {'hands':>6} {'mbb/g':>12}"
    )
    print(header)
    print("-" * len(header))
    for r in rows:
        knobs = r.get("knobs", {})
        res = r.get("results", {})
        mbb = res.get("exploitability_mbb")
        se = res.get("std_error_mbb")
        score = f"{mbb:.1f}±{se:.1f}" if isinstance(mbb, (int, float)) and se is not None else "—"
        print(
            f"{r.get('run_id', '')[:26]:<26} "
            f"{_fmt_commit(r.get('eval_git_commit'), r.get('eval_git_dirty')):<14} "
            f"{knobs.get('scorer', '')!s:<10} "
            f"{knobs.get('opponent', '')!s:<10} "
            f"{knobs.get('base_seed', '')!s:>12} "
            f"{res.get('num_hands', '')!s:>6} "
            f"{score:>12}"
        )


COMMAND = Command(
    name="ledger",
    help="List recorded evaluations from the eval ledger.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
