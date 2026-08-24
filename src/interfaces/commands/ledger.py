"""The `ledger` subcommand: its flags, handler and renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.interfaces.commands._base import (
    Command,
    ledger_for,
    records_root,
)
from src.pipeline.evaluation import ledger as eval_ledger

if TYPE_CHECKING:
    import argparse
    from pathlib import Path


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver ledger`."""
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
        "--full",
        action="store_true",
        help="Carry each row's FULL results (per-seat values, decomposition, policy "
        "profile) instead of the index's four summary fields.",
    )


class LedgerRow(BaseModel):
    """One recorded evaluation, as it sits in the derived index.

    LENIENT, unlike the payloads this file builds. A row is a RECORD read off
    the share, written by whatever version of `evaluate` produced it -- so this
    names the fields a surface reads and lets the rest through, which is what
    `extra="allow"` was for before the models became producers.
    """

    model_config = ConfigDict(extra="allow")

    run_id: str | None = None
    eval_git_commit: str | None = None
    """The instrument, and what it measured. Never merge rows across knobs: an
    exploitability figure is meaningless without the tier it was measured at."""
    knobs: dict[str, Any] = Field(default_factory=dict)
    results: dict[str, Any] = Field(default_factory=dict)


class LedgerPayload(BaseModel):
    """Recorded evaluations, derived from the published per-run documents."""

    op: Literal["ledger"] = "ledger"
    """The path read, always a derived file in a temporary tree -- reported
    rather than hidden, so an empty listing names something to go and look at."""
    ledger: str
    """How many rows the FILTERS matched, before `--limit` paged them."""
    matched: int
    rows: list[LedgerRow] = Field(default_factory=list)


def run(args: argparse.Namespace) -> LedgerPayload:
    """List recent eval rows, derived from the published documents.

    ``--rebuild`` and ``--migrate`` are gone with the local runs directory they
    acted on. Rebuilding is not a mode any more: the index has no stored form,
    so every read of it IS a rebuild. Migration rewrites records in place and
    the materialised tree is a throwaway copy -- it could only ever have
    reported a success that changed nothing.
    """
    with records_root(args) as root:
        return _list(args, root)


def _list(args: argparse.Namespace, root: Path) -> LedgerPayload:
    ledger_path = ledger_for(root)
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
    # `records[-0:]` is the whole list, so a 0 limit already meant "all" by
    # accident. Made deliberate: `--limit 0` is how to see the whole history.
    matched = len(records)
    if args.limit > 0:
        records = records[-args.limit :]
    if getattr(args, "full", False):
        # The index row keeps four summary fields; the document has the rest.
        # Loaded only for the rows that survived the filters, after paging.
        for record in records:
            record["results"] = eval_ledger.load_payload(record, root).get("results", {})
    return LedgerPayload(
        ledger=str(ledger_path),
        matched=matched,
        rows=[LedgerRow.model_validate(row) for row in records],
    )


def _fmt_commit(commit: str | None, dirty: bool | None) -> str:
    if not commit:
        return "—"
    short = commit[:7]
    if dirty:
        short += "-dirty"
    return short


def render(payload: LedgerPayload) -> None:
    rows = payload.rows
    if not rows:
        print(f"No eval-ledger entries in {payload.ledger}.")
        return
    matched = payload.matched
    shown = (
        f"{len(rows)} row(s)"
        if matched <= len(rows)
        else f"{len(rows)} of {matched} row(s) (--limit 0 for all)"
    )
    print(f"Eval ledger ({payload.ledger}): {shown}")
    header = f"{'run_id':<44} {'at':>12} {'commit':<9} {'mbb/g':>12}  tier"
    print(header)
    print("-" * len(header))
    for r in rows:
        res = r.results
        mbb = res.get("exploitability_mbb")
        se = res.get("std_error_mbb")
        score = f"{mbb:.1f}±{se:.1f}" if isinstance(mbb, (int, float)) and se is not None else "—"
        at = getattr(r, "checkpoint_iteration", None)
        # The TIER, not three of its knobs: an exact_br row carries no
        # scorer/opponent/seed, so 4/2/2-annulled and 4/16/16-conditional rows
        # rendered as identical lines and only their `mbb/g` said otherwise.
        print(
            f"{(r.run_id or '')[:44]:<44} "
            f"{(f'{at:,}' if isinstance(at, int) else '—'):>12} "
            f"{_fmt_commit(r.eval_git_commit, getattr(r, 'eval_git_dirty', None)):<9} "
            f"{score:>12}  {eval_ledger.tier_label(r.model_dump())}"
        )


COMMAND = Command(
    name="ledger",
    help="List recorded evaluations from the eval ledger.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
