"""The `chipzen-seat` subcommand: play a run on chipzen.ai's external-API division.

Two modes, and the offline one is not a courtesy. Their arena is invite-gated and
a match cannot be conjured, so ``--replay`` runs a recorded ``turn_request``
through the identical decision path and prints what would have gone back on the
wire. That is the only way to exercise this against a real payload without an
account, and the only way to diff a decision after a retrain.

Live play needs the SDK (`uv sync --extra chipzen`) and a `cz_extbot_` token for
a bot you own. The token is read from the environment by default: it is shown
once at creation and never again, and a shell history is a bad place for it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from src.interfaces.commands._base import Command, resolve_run_dir
from src.interfaces.errors import CommandError

if TYPE_CHECKING:
    import argparse

TOKEN_ENV = "CHIPZEN_EXTBOT_TOKEN"
BOT_ENV = "CHIPZEN_BOT_ID"
# Staging, not production. Every external-API instruction in their SDK points at
# staging.chipzen.ai, and their protocol spec says a 404 on the matchmaking
# endpoints means "not deployed on that environment yet" -- so production is the
# likelier first-run 404, on a path that cannot be debugged without an account.
DEFAULT_ENV = "staging"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver chipzen-seat`."""
    parser.add_argument("--run", required=True, help="Run id, fragment, or path to a run dir.")
    parser.add_argument(
        "--runs-dir",
        default="/mnt/work/runs",
        help="Where runs live on this box. Local disk, never the share.",
    )
    parser.add_argument(
        "--at", type=int, default=None, help="Seat the checkpoint at this iteration."
    )
    parser.add_argument(
        "--replay",
        default=None,
        help="Decide one recorded turn from a JSON file and exit. No account needed.",
    )
    parser.add_argument("--bot-id", default=None, help=f"Bot UUID (default: ${BOT_ENV}).")
    parser.add_argument(
        "--env",
        default=DEFAULT_ENV,
        help=f"Chipzen environment (default {DEFAULT_ENV}, which is the one their "
        "external-API docs actually document; pass production once it is live there).",
    )
    parser.add_argument(
        "--resolver",
        action="store_true",
        help="Arm the subgame resolver. Off by default: it has been measured to "
        "collapse off-tree, which is what arena play is.",
    )
    parser.add_argument(
        "--budget-ms",
        type=int,
        default=1200,
        help="Per-decision budget. Their ranked and tournament clock is 2000 ms "
        "round-trip, so leave room for the frame (default 1200).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Play a single match and exit, rather than holding the seat.",
    )


class ChipzenSeatPayload(BaseModel):
    """What was seated, and -- in replay mode -- what it decided."""

    op: Literal["chipzen-seat"] = "chipzen-seat"
    run: str
    run_dir: str
    runs_dir: str
    at_iteration: int | None = None
    mode: Literal["replay", "live"]
    resolver: bool
    budget_ms: int
    env: str | None = None
    bot_id: str | None = None
    once: bool = False
    # Replay only.
    decision: dict[str, Any] | None = None
    depth_matches: bool | None = None
    their_depth: float | None = None
    our_depth: float | None = None
    off_tree: int | None = None
    truncated: bool | None = None


def run(args: argparse.Namespace) -> ChipzenSeatPayload:
    """Resolve the run and the credentials; :func:`render` does the playing.

    A server never returns and neither does a held seat, so the same split
    `blueprint-serve` makes applies: everything that can be refused cheaply is
    refused here, before a minute of checkpoint loading.
    """
    run_dir = resolve_run_dir(args.run, args.runs_dir)
    if args.replay:
        return _replay_payload(args, run_dir)

    bot_id = args.bot_id or os.environ.get(BOT_ENV)
    if not bot_id:
        raise CommandError(f"Pass --bot-id, or set ${BOT_ENV}.")
    if not os.environ.get(TOKEN_ENV):
        raise CommandError(
            f"Set ${TOKEN_ENV} to a cz_extbot_ token for this bot. It is shown "
            "once at creation and cannot be read back."
        )
    return ChipzenSeatPayload(
        run=run_dir.name,
        run_dir=str(run_dir),
        runs_dir=args.runs_dir,
        at_iteration=args.at,
        mode="live",
        resolver=args.resolver,
        budget_ms=args.budget_ms,
        env=args.env,
        bot_id=bot_id,
        once=args.once,
    )


def _replay_payload(args: argparse.Namespace, run_dir: Path) -> ChipzenSeatPayload:
    """Decide one recorded turn, without a socket or an account.

    The file is ``{"game_config": {...}, "seat": N, "state": {...}}`` -- the two
    payloads a live client gets from ``match_start`` and ``turn_request``, plus
    which seat is ours. Their ``state`` goes in verbatim, so a frame captured
    from a real match can be replayed by pasting it in.
    """
    from src.interfaces.chipzen.seat import BlueprintSeat  # noqa: PLC0415 -- see module docstring

    recorded = _load_recording(Path(args.replay))
    blueprint = _build_blueprint(run_dir, args.at)
    seat = BlueprintSeat.for_match(
        blueprint,
        {"game_config": recorded["game_config"]},
        seat=int(recorded.get("seat", 0)),
        use_resolver=args.resolver,
        budget_ms=args.budget_ms,
    )
    decision = seat.decide_frame(recorded["state"])
    return ChipzenSeatPayload(
        run=run_dir.name,
        run_dir=str(run_dir),
        runs_dir=args.runs_dir,
        at_iteration=args.at,
        mode="replay",
        resolver=args.resolver,
        budget_ms=args.budget_ms,
        decision=decision,
        depth_matches=seat.scale.depth_matches,
        their_depth=seat.scale.their_depth,
        our_depth=seat.scale.our_depth,
        off_tree=seat.tally.off_tree,
        truncated=bool(seat.tally.truncated),
    )


def _load_recording(path: Path) -> dict[str, Any]:
    """Read and shape-check a recorded turn before anything expensive loads."""
    if not path.is_file():
        raise CommandError(f"No such file: {path}.")
    try:
        recorded = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise CommandError(f"{path} is not JSON: {exc}.") from exc
    for key in ("game_config", "state"):
        if key not in recorded:
            raise CommandError(f"{path} has no '{key}'. Expected match_start's and turn_request's.")
    return recorded


def _build_blueprint(run_dir: Path, at_iteration: int | None):
    """A run directory on local disk -> a blueprint. ~1 min in production."""
    from src.pipeline.services.scoring._shared import (  # noqa: PLC0415 -- see module docstring
        build_blueprint_for,
    )
    from src.pipeline.training.run_tracker import RunTracker  # noqa: PLC0415 -- see above

    metadata = RunTracker.load(run_dir).metadata
    solver, _storage = build_blueprint_for(
        run_dir,
        metadata,
        abstraction_hash=metadata.card_abstraction_hash,
        at_iteration=at_iteration,
    )
    return solver


def render(payload: ChipzenSeatPayload) -> None:
    if payload.mode == "replay":
        _render_replay(payload)
        return
    _play(payload)


def _render_replay(payload: ChipzenSeatPayload) -> None:
    decision = payload.decision or {}
    amount = decision.get("params", {}).get("amount")
    sized = f" to {amount}" if amount is not None else ""
    print(f"{payload.run}: {decision.get('action', '?')}{sized}")
    if not payload.depth_matches:
        print(
            f"  table is {payload.their_depth:.1f} bb deep, blueprint trained at "
            f"{payload.our_depth:.1f} bb -- extrapolating."
        )
    if payload.off_tree:
        print(f"  {payload.off_tree} opponent action(s) snapped on-tree.")
    if payload.truncated:
        print("  replay did not land on our seat; the decision is a safe default.")


def _play(payload: ChipzenSeatPayload) -> None:
    """Hold the seat. Does not return until interrupted."""
    from pathlib import Path  # noqa: PLC0415 -- deferred with the rest of the play path

    from src.interfaces.chipzen.seat import run_seat  # noqa: PLC0415 -- see above

    run_dir = Path(payload.run_dir)
    print(f"Seating {payload.run} on chipzen {payload.env} as {payload.bot_id}.")
    run_seat(
        lambda: _build_blueprint(run_dir, payload.at_iteration),
        bot_id=payload.bot_id or "",
        token=os.environ[TOKEN_ENV],
        env=payload.env or DEFAULT_ENV,
        use_resolver=payload.resolver,
        budget_ms=payload.budget_ms,
        loop=not payload.once,
    )


COMMAND = Command(
    name="chipzen-seat",
    help="Play a run on chipzen.ai, or replay one recorded turn offline.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
