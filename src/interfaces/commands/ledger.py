"""The `ledger` subcommand: its flags, handler and renderer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.adapters.postgres import connect, queries
from src.interfaces.commands._base import Command
from src.pipeline.evaluation import ledger as eval_ledger

if TYPE_CHECKING:
    import argparse


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
    # How many rows the FILTERS matched, before `--limit` paged them.
    matched: int
    rows: list[LedgerRow] = Field(default_factory=list)


def run(args: argparse.Namespace) -> LedgerPayload:
    """List recent eval rows, cut to the page in SQL.

    `--since` is the one filter that stays in Python: it compares the instants
    `record_instant` derives from each document's own timestamp, where naive
    legacy values mean local time -- a rule the column's UTC conversion does
    not apply. With it set, the whole filtered set comes over and pages here.
    """
    matched, documents = queries.ledger_page(
        connect.engine_from_environment(),
        run_id=args.run,
        method=args.method,
        experiment_id=args.experiment,
        limit=0 if args.since else args.limit,
    )
    records = [eval_ledger.ledger_row(document) for document in documents]
    if args.since:
        cutoff = eval_ledger.record_instant({"timestamp": args.since})
        records = [r for r in records if eval_ledger.record_instant(r) >= cutoff]
        matched = len(records)
        if args.limit > 0:
            records = records[-args.limit :]
    return LedgerPayload(matched=matched, rows=[LedgerRow.model_validate(r) for r in records])


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
        print("No recorded evaluations match.")
        return
    matched = payload.matched
    shown = (
        f"{len(rows)} row(s)"
        if matched <= len(rows)
        else f"{len(rows)} of {matched} row(s) (--limit 0 for all)"
    )
    print(f"Recorded evaluations: {shown}")
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
