"""The `score` subcommand: evaluate a published run on the pool.

Scoring runs on a node because the share is a LOCAL mount there. One
checkpoint is ~540 MB of small zarr chunks, roughly twenty minutes to pull over
SMB, which makes scoring a ladder from a laptop impractical.

**One task per rung, not one task looping over them.** Rungs are independent,
so Batch spreads them across the pool and queues the rest; a single looping
task pinned an entire curve to one node however much pool was available. Each
task also parallelises its own best-response walks, so a rung uses ~4 cores and
a node holds one comfortably.
"""

from __future__ import annotations

import argparse
from typing import Literal

from pydantic import Field

from src.interfaces.cloud.tasks import dispatch, spec
from src.interfaces.commands._base import Command
from src.interfaces.commands.evaluate import EVAL_METHODS
from src.shared.cloudtask.kinds import TaskName


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver score`."""
    parser.add_argument("--run", required=True, help="Published run id to score.")
    parser.add_argument(
        "--method",
        default="exact_br",
        choices=EVAL_METHODS,
        help="Estimator the node runs. Kept in step with `evaluate --method`: a "
        "value this accepts but the node rejects costs a snapshot upload, a "
        "~3-minute pool spin-up and three node allocations (task retries) before "
        "failing on an argparse error.",
    )
    parser.add_argument(
        "--at",
        default="",
        help="Comma-separated ladder rungs to score (empty = the latest checkpoint). "
        "Each rung becomes its own task.",
    )
    parser.add_argument(
        "--timeout",
        default=spec.DEFAULT_TIMEOUT,
        help="Wall-clock ceiling on each scoring process.",
    )
    parser.add_argument(
        "flags",
        nargs=argparse.REMAINDER,
        # Declared, though argparse supplies `[]` for an unmatched REMAINDER on
        # the command line whatever this says. It is the parser that
        # `Command.arguments` reads to answer "what does this command accept",
        # so an undeclared default means the schema says `None` where the CLI
        # produces `[]` -- and `_passthrough(None)` raises TypeError.
        default=[],
        help="Extra flags for `evaluate` on the node, AFTER a `--` separator: "
        "`score --run r -- --br-flops 8`. The separator is required — argparse "
        "will not hand a bare `--br-flops` to a passthrough, it rejects it as "
        "an unrecognised argument of this command.",
    )


def _rungs(raw: str) -> list[str]:
    """Split the rung list, treating empty as 'the latest checkpoint'."""
    parsed = [part.strip() for part in raw.split(",") if part.strip()]
    return parsed or [""]


def _passthrough(flags: list[str]) -> tuple[str, ...]:
    """Drop the `--` separator argparse leaves at the head of REMAINDER.

    It is ours, not the node's: passing it on would make `evaluate` parse a
    bare `--` as the end of its own options.
    """
    return tuple(flags[1:] if flags and flags[0] == "--" else flags)


class ScorePayload(dispatch.Dispatched):
    """One scoring task per ladder rung, all against one code snapshot."""

    op: Literal["score"] = "score"
    run_id: str
    method: str
    """The rungs covered, one task each -- they are independent, so Batch is the
    scheduler. STRINGS, not ints: an empty one means the latest checkpoint, which
    is a rung the ladder cannot name in advance."""
    rungs: list[str] = Field(default_factory=list)


def run(args: argparse.Namespace) -> ScorePayload:
    """Stage the tree and queue one scoring task per rung."""
    # Imported here, not at module scope, so `--help` does not pay for the
    # Azure SDK -- the same rule `_base.records_root` follows.
    from src.interfaces.cloud.store.workspace import (  # noqa: PLC0415 -- see above
        resolve_published_run,
    )

    # The NODE has no fragment matcher, so a fragment must become a full id
    # HERE or it fails after a snapshot upload and three retries.
    run_id = resolve_published_run(args.run)
    rungs = _rungs(args.at)
    payload = dispatch.stage_and_queue(
        lambda snapshot: [
            spec.TaskSpec(
                code_snapshot=snapshot,
                op=TaskName.EVALUATE,
                run_id=run_id,
                eval_method=args.method,
                eval_at=rung,
                eval_flags=_passthrough(args.flags),
                timeout=args.timeout,
            )
            for rung in rungs
        ]
    )
    return payload.extend(ScorePayload, run_id=run_id, method=args.method, rungs=rungs)


def render(payload: ScorePayload) -> None:
    rungs = ", ".join(rung or "latest" for rung in payload.rungs)
    print(f"Scoring {payload.run_id} with {payload.method} at: {rungs}")
    dispatch.render_queued(payload)


COMMAND = Command(
    name="score",
    help="Evaluate a published run on the pool, one task per ladder rung.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
