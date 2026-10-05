"""The `score` subcommand: evaluate a published run on the pool.

Scoring runs on a node because the share is a LOCAL mount there. One
checkpoint is ~540 MB of small zarr chunks, roughly twenty minutes to pull over
SMB, which makes scoring a ladder from a laptop impractical.

**One task per rung, not one task looping over them.** Rungs are independent,
so Batch spreads them across the pool and queues the rest; a single looping
task pinned an entire curve to one node however much pool was available. Each
task also fans its flop subtrees over the node's cores, so one rung fills one
node.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from typing import Literal

from pydantic import Field

from src.interfaces.cloud.tasks import dispatch, spec
from src.interfaces.commands._base import Command, CommandError, eval_index_rows
from src.interfaces.commands.evaluate import EVAL_METHODS
from src.shared.cloudtask.kinds import TaskName
from src.shared.cloudtask.rung_spec import format_ladder, parse_ladder


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
        "--vs",
        default=None,
        help="[blueprint_match, depth_duel] The OTHER arm, as a published run id or a fragment of "
        "one. Resolved and verified HERE, like --run: the node has no fragment matcher "
        "and finds out only after a snapshot upload and three retries.",
    )
    parser.add_argument(
        "--vs-at",
        default="",
        help="[blueprint_match, depth_duel] Rung of --vs to play (empty = its latest checkpoint). "
        "One rung, not a list: the ladder --at fans over is run A's.",
    )
    parser.add_argument(
        "--ladder-a",
        default="",
        help="[elimination] Comma-separated `run[:at[:threshold]]` rungs, deploy.sh's "
        "spelling; runs may be fragments. Must include --run, which files the row and "
        "takes its rung from here when --at is empty.",
    )
    parser.add_argument("--ladder-b", default="", help="[elimination] The opposing ladder.")
    parser.add_argument(
        "--timeout",
        default=spec.DEFAULT_TIMEOUT,
        help="Wall-clock ceiling on each scoring process.",
    )
    parser.add_argument(
        "--pool",
        choices=("train", "big", "huge", "mem"),
        default="train",
        help="Which pool runs it. `exact_br` forks its workers off the loaded "
        "blueprint, so each costs ~1.5 GB and a box's cores are the limit. MEASURED "
        "09-26, production @2,400 at 4/16/16: train (D16, 15 workers) 56 min, huge "
        "(D64, 63 workers) 20 min, same number. For ONE rung take huge. For a "
        "ladder the lever is CONCURRENCY: `--at` makes one task per rung, and "
        "huge holds 4 nodes where train holds 36 -- ten rungs are one ~56-min "
        "round on train and three ~20-min rounds on huge.",
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


def _opponent_flags(method: str, vs: str | None, vs_at: str) -> tuple[str, ...]:
    """`--vs`/`--vs-at` as the flags the node's `evaluate` takes.

    Resolved and verified on THIS side for the same reason `--run` is: a
    fragment or a pruned rung fails on the node after a snapshot upload, a
    ~3-minute pool spin-up and three task retries.
    """
    from src.interfaces.cloud.store.workspace import (  # noqa: PLC0415 -- Azure SDK off `--help`
        resolve_published_run,
        verify_published_rungs,
    )

    if method in ("blueprint_match", "depth_duel") and not vs:
        raise CommandError(f"`--method {method}` needs an opponent: pass `--vs <run>`.")
    if not vs:
        return ()
    vs_id = resolve_published_run(vs)
    rung = _rungs(vs_at)[0]
    verify_published_rungs(vs_id, [rung])
    return ("--vs-run", vs_id) + (("--vs-at", rung) if rung else ())


def _ladder_flags(
    method: str, run_id: str, ladder_a: str, ladder_b: str, rungs: list[str]
) -> tuple[tuple[str, ...], list[str]]:
    """Both ladders, every run resolved and every rung verified, as node flags.

    Also the rung of ``run_id`` to score: the row is filed under it, so it must be
    one of ladder A's rungs, and an empty ``--at`` takes that rung's.
    """
    from src.interfaces.cloud.store.workspace import (  # noqa: PLC0415 -- Azure SDK off `--help`
        resolve_published_run,
        verify_published_rungs,
    )

    if method != "elimination":
        if ladder_a or ladder_b:
            raise CommandError("--ladder-a/--ladder-b are for `--method elimination` only.")
        return (), rungs
    if not ladder_a or not ladder_b:
        raise CommandError("`--method elimination` needs --ladder-a and --ladder-b.")
    try:
        ladders = [parse_ladder(ladder_a), parse_ladder(ladder_b)]
    except ValueError as exc:
        raise CommandError(str(exc)) from exc
    resolved = [
        [replace(rung, run=resolve_published_run(rung.run)) for rung in ladder]
        for ladder in ladders
    ]
    for rung in {rung for ladder in resolved for rung in ladder}:
        verify_published_rungs(rung.run, [str(rung.at) if rung.at is not None else ""])
    anchor = [rung for rung in resolved[0] if rung.run == run_id]
    if not anchor:
        raise CommandError(f"--run {run_id} is not a rung of --ladder-a; file the row under one.")
    own = str(anchor[0].at) if anchor[0].at is not None else ""
    if rungs != [""] and rungs != [own]:
        raise CommandError(f"--at {','.join(rungs)} disagrees with ladder A's {anchor[0]}.")
    flags = ("--ladder-a", format_ladder(resolved[0]), "--ladder-b", format_ladder(resolved[1]))
    return flags, [own]


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
    """The tiers this run's EXISTING scores of this method sit in. Printed so a
    mismatch is visible at dispatch: the knobs live on `evaluate`, so omitting
    them silently takes ITS defaults (8/2/2, unthresholded) rather than whatever
    the run was measured at, and a ladder on the coarse ruler INVERTS A/B signs.
    Cost me a seven-rung ladder on 09-26 before `curve` reported the collision."""
    existing_tiers: list[str] = Field(default_factory=list)


def run(args: argparse.Namespace) -> ScorePayload:
    """Stage the tree and queue one scoring task per rung."""
    # Imported here, not at module scope, so `--help` does not pay for the
    # Azure SDK -- the same rule `_base.records_root` follows.
    from src.interfaces.cloud.store.workspace import (  # noqa: PLC0415 -- see above
        resolve_published_run,
        verify_published_rungs,
    )

    # The NODE has no fragment matcher, so a fragment must become a full id
    # HERE or it fails after a snapshot upload and three retries.
    run_id = resolve_published_run(args.run)
    rungs = _rungs(args.at)
    # Same rule, one step further: a rung the manifest advertises may have been
    # pruned off the share, and the node only finds out after `uv sync`.
    verify_published_rungs(run_id, rungs)
    opponent = _opponent_flags(args.method, args.vs, args.vs_at)
    ladders, rungs = _ladder_flags(args.method, run_id, args.ladder_a, args.ladder_b, rungs)
    opponent += ladders
    payload = dispatch.stage_and_queue(
        lambda snapshot: [
            spec.TaskSpec(
                code_snapshot=snapshot,
                op=TaskName.EVALUATE,
                run_id=run_id,
                eval_method=args.method,
                eval_at=rung,
                eval_flags=opponent + _passthrough(args.flags),
                timeout=args.timeout,
            )
            for rung in rungs
        ],
        pool=args.pool,
    )
    return payload.extend(
        ScorePayload,
        run_id=run_id,
        method=args.method,
        rungs=rungs,
        existing_tiers=_existing_tiers(run_id, args.method),
    )


def _existing_tiers(run_id: str, method: str) -> list[str]:
    """Distinct tiers this run already has scores of ``method`` in.

    Best-effort: a record that cannot be reached must not stop a dispatch, since
    the tier note is an aid and the scoring is the job.
    """
    from src.adapters.postgres import connect  # noqa: PLC0415 -- see `run`
    from src.pipeline.evaluation import ledger as eval_ledger  # noqa: PLC0415

    try:
        rows = eval_index_rows(connect.engine_from_environment(), run_id)
    except Exception:  # noqa: BLE001 -- an unreachable record is not a refusal
        return []
    # "method", the key the eval DOCUMENT uses and `tier_label` reads -- not
    # "eval_method", which is the flag's name on this side of the wire.
    labels = {eval_ledger.tier_label(row) for row in rows if row.get("method") == method}
    return sorted(labels)


def render(payload: ScorePayload) -> None:
    rungs = ", ".join(rung or "latest" for rung in payload.rungs)
    print(f"Scoring {payload.run_id} with {payload.method} at: {rungs}")
    dispatch.render_queued(payload)
    # AFTER the queue lines, because the tier is a check on what was just sent:
    # if it does not match what these tasks will produce, cancel now rather than
    # reading the curve in an hour and finding two rulers in it.
    for tier in payload.existing_tiers:
        print(f"  existing {payload.method} scores of this run are in: {tier}")


COMMAND = Command(
    name="score",
    help="Evaluate a published run on the pool, one task per ladder rung.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
