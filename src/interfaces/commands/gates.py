"""The `gates` subcommand: head-to-head matches pooled over their deal seeds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field

from src.adapters.postgres import connect, queries
from src.interfaces.commands._base import Command
from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.evaluation.ledger.gates import Gate, pool_matches

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver gates`."""
    parser.add_argument("--run", default=None, help="Only gates where this run is side A.")
    parser.add_argument(
        "--experiment", default=None, help="Only gates tagged with this experiment."
    )
    parser.add_argument("--limit", type=int, default=50, help="Newest N gates (0 = all).")


class GatesPayload(BaseModel):
    """Pooled head-to-head gates, newest first. `rows` counts the match rows read."""

    op: Literal["gates"] = "gates"
    rows: int
    duplicates: int
    gates: list[Gate] = Field(default_factory=list)


def run(args: argparse.Namespace) -> GatesPayload:
    """Read every `blueprint_match` row the filters allow and pool them per gate.

    The page is cut AFTER pooling: a SQL limit would split a gate's seeds and
    report a two-seed gate that actually has thirteen.
    """
    _, documents = queries.ledger_page(
        connect.engine_from_environment(),
        run_id=args.run,
        method="blueprint_match",
        experiment_id=args.experiment,
        limit=0,
    )
    records = [eval_ledger.ledger_row(document) for document in documents]
    gates = pool_matches(records)
    shown = gates[: args.limit] if args.limit > 0 else gates
    return GatesPayload(rows=len(records), duplicates=sum(g.duplicates for g in gates), gates=shown)


def _short(run_id: str | None) -> str:
    if not run_id:
        return "?"
    parts = run_id.removeprefix("run-").split("-")
    return f"{parts[0]}-{parts[-1]}" if len(parts) > 2 else run_id


def _rung(at: int | None) -> str:
    if at is None:
        return "?"
    for scale, suffix in ((1e9, "B"), (1e6, "M"), (1e3, "k")):
        if at >= scale:
            return f"{at / scale:g}{suffix}"
    return str(at)


def render(payload: GatesPayload) -> None:
    if not payload.gates:
        print("No head-to-head gates recorded.")
        return
    print(
        f"{len(payload.gates)} gate(s) from {payload.rows} match row(s); "
        f"{payload.duplicates} re-scored seed row(s) dropped"
    )
    print("  edge: A's chips per hand against B, pooled over seeds (+ = A wins)")
    header = f"{'A':<22} {'B':<22} {'edge':>8} {'se':>6} {'sigma':>6} {'seeds':>5}  experiment"
    print(header)
    print("-" * len(header))
    for g in payload.gates:
        a = f"{_short(g.run_id)}@{_rung(g.at)}"
        b = f"{_short(g.run_b)}@{_rung(g.at_b)}" + (" (dep)" if g.b_side == "deployed" else "")
        sigma = f"{g.sigma:+.1f}" if g.sigma is not None else "—"
        print(
            f"{a:<22} {b:<22} {g.edge_mbb:>+8.1f} {g.se_mbb:>6.1f} {sigma:>6} "
            f"{len(g.seeds):>5}  {g.experiment_id or '—'}"
        )


COMMAND = Command(
    name="gates",
    help="Head-to-head matches pooled over their deal seeds: edge, error, sigma.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
