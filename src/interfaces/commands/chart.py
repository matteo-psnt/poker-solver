"""The `chart` subcommand: a run's preflop strategy as a 13x13 grid.

The one artifact that answers "did this train into poker?" without a number. A
person reads a chart in seconds -- raises top-left, folds bottom-right, suited
above the diagonal playing more than its offsuit twin below -- and no summary
statistic carries that judgement.

Cells show aggression as a percentage, and the legend counts what was never
trained: an untrained class is left blank rather than shown as uniform, because a
uniform row looks exactly like a deliberate mixed strategy and is not one.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from src.interfaces.commands._base import Command, resolve_run_dir

if TYPE_CHECKING:
    import argparse

# Aggression bands, low to high. Chosen so an unraised limp and a pure raise are
# never the same shade; the boundaries are read points, not thresholds anything
# depends on.
_SHADES = ((0.02, "·"), (0.20, "░"), (0.50, "▒"), (0.80, "▓"), (1.01, "█"))


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver chart`."""
    parser.add_argument("--run", required=True, help="Run id, fragment, or path to a run dir.")
    parser.add_argument(
        "--runs-dir",
        default="/mnt/work/data/runs",
        help="Where runs live on this box. Local disk, never the share.",
    )
    parser.add_argument("--at", type=int, default=None, help="Chart this iteration's checkpoint.")
    parser.add_argument(
        "--path",
        default="",
        help="Preflop line to chart. Empty is the opening spot; 'c' is the big "
        "blind's option behind a limp; a raise token is the spot facing it.",
    )
    parser.add_argument(
        "--counts",
        action="store_true",
        help="Show training visits per class instead of aggression.",
    )


class ChartPayload(BaseModel):
    """One preflop node's strategy, class by class."""

    op: Literal["chart"] = "chart"
    run: str
    path: str
    actor: int
    actions: list[str]
    trained: int
    untrained: int
    counts: bool = False
    """Render visit counts rather than the mix -- a flag the payload carries so
    the renderer reads it from the payload like every other field."""
    rows: dict[str, dict[str, Any]]


def run(args: argparse.Namespace) -> ChartPayload:
    """Load the run and read its preflop strategy."""
    from src.pipeline.blueprint.preflop import preflop_chart  # noqa: PLC0415 -- heavy import
    from src.pipeline.services.scoring._shared import (  # noqa: PLC0415 -- see above
        build_blueprint_for,
    )
    from src.pipeline.training.run_tracker import RunTracker  # noqa: PLC0415 -- see above

    run_dir = resolve_run_dir(args.run, args.runs_dir)
    metadata = RunTracker.load(run_dir).metadata
    blueprint, _storage = build_blueprint_for(
        run_dir,
        metadata,
        abstraction_hash=metadata.card_abstraction_hash,
        at_iteration=args.at,
    )
    chart = preflop_chart(blueprint, args.path)
    return ChartPayload(
        run=run_dir.name,
        path=chart.path,
        actor=chart.actor,
        actions=list(chart.actions),
        trained=len(chart.classes),
        untrained=len(chart.untrained),
        counts=bool(args.counts),
        rows={
            label: {
                "strategy": list(row.strategy),
                "aggression": row.aggression,
                "fold": row.fold,
                "passive": row.passive,
                "reach_count": row.reach_count,
            }
            for label, row in sorted(chart.classes.items())
        },
    )


def render(payload: ChartPayload) -> None:
    counts = payload.counts
    print(f"{payload.run} · preflop{f' after {payload.path}' if payload.path else ''}")
    print(f"seat {payload.actor} to act · actions {' '.join(payload.actions)}")
    print()
    _print_grid(payload, counts=counts)
    print()
    print(f"{payload.trained} of 169 classes trained, {payload.untrained} never visited.")
    if not counts:
        print("cell = P(raise or all-in).  · <2%   ░ <20%   ▒ <50%   ▓ <80%   █ 80%+")
        print("suited above the diagonal, offsuit below, pairs on it.")


def _print_grid(payload: ChartPayload, *, counts: bool) -> None:
    from src.pipeline.blueprint.preflop import RANKS, class_label  # noqa: PLC0415 -- render-only

    header = "    " + "".join(f"{rank:>4}" for rank in RANKS)
    print(header)
    for high in RANKS:
        cells = []
        for low in RANKS:
            row = payload.rows.get(class_label(high, low))
            if row is None:
                cells.append("   .")
            elif counts:
                cells.append(f"{row['reach_count']:>4}")
            else:
                cells.append(f"{_cell(row['aggression']):>4}")
        print(f"{high:>3} " + "".join(cells))


def _cell(aggression: float) -> str:
    """A percentage with a shade, so the shape reads before the numbers do."""
    for ceiling, shade in _SHADES:
        if aggression < ceiling:
            return f"{shade}{round(aggression * 100):>2}"
    return f"█{round(aggression * 100):>2}"


COMMAND = Command(
    name="chart",
    help="A run's preflop strategy as a 13x13 grid.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
