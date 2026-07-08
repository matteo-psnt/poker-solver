"""The `chipzen-seat` subcommand: play a run on chipzen.ai's external-API division.

Two modes, and the offline one is not a courtesy. Their arena is invite-gated and
a match cannot be conjured, so ``--replay`` runs a recorded ``turn_request``
through the identical decision path and prints what would have gone back on the
wire. That is the only way to exercise this against a real payload without an
account, and the only way to diff a decision after a retrain.

Live play needs the SDK (`uv sync --extra chipzen`) and a `cz_extbot_` token for
a bot you own. There is deliberately no `--token`: it is shown once at creation
and never again, and a shell history is a bad place for it. Supply it through
``$CHIPZEN_EXTBOT_TOKEN`` or the SDK's own ``~/.chipzen/chipzen.toml`` -- NOT a
file in this tree, which a dispatch would seal into the code snapshot and upload
to the share.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel

from src.interfaces.chipzen.seat import (
    CLOCK_FRACTION,
    DEFAULT_POLICY_THRESHOLD,
    TIGHT_CLOCK_MS,
)
from src.interfaces.commands._base import Command, resolve_run_dir
from src.interfaces.errors import CommandError

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import argparse

TOKEN_ENV = "CHIPZEN_EXTBOT_TOKEN"
BOT_ENV = "CHIPZEN_BOT_ID"

# Where `infra/serve/deploy.sh` actually puts a staged run -- it writes
# `RUNS_DIR=$WORK/data/runs` into /etc/blueprint.env, and the `data` directory is
# a symlink the deploy makes because the abstraction resolver scans `<cwd>/data`.
# `blueprint-serve` defaults to /mnt/work/runs and gets away with it only because
# systemd passes RUNS_DIR explicitly; nothing passes it to this command.
BOX_RUNS_DIR = "/mnt/work/data/runs"
# Their names, not ours -- the SDK resolves exactly these three and "production"
# is not one of them.
#
# `prod`, measured 2026-08-24, against their docs. Every external-API
# instruction they publish points at staging.chipzen.ai, so this defaulted there
# first; a real token then returned EXTAPI_INVALID_TOKEN on staging and a live
# `{"status":"idle"}` on prod, and the prod lobby completed a handshake. Tokens
# are minted per environment and the division is deployed on prod -- so the
# documentation is what is stale here, not the platform.
ENVIRONMENTS = ("prod", "staging", "local")
DEFAULT_ENV = "prod"

# The SDK's own config, searched in this order. The repo root is on their list
# and NOT on ours by preference: a snapshot seals the working tree, so a token
# left there rides a dispatch up to the share. `~/.chipzen/` is outside every
# tree and is the one to tell people about.
#
# The cwd entry is bound at IMPORT, so it is where the process started rather
# than where it is running -- true of the SDK's own discovery too, and harmless
# for the path we document. Tests inject this tuple rather than write to $HOME.
PREFERRED_CONFIG = "~/.chipzen/chipzen.toml"
CONFIG_SEARCH = (
    Path.cwd() / "chipzen.toml",
    Path.home() / ".chipzen" / "chipzen.toml",
    Path("/etc/chipzen/chipzen.toml"),
)


def sdk_config_path() -> Path | None:
    """The `chipzen.toml` the SDK would find, if any -- so we can refuse before it."""
    for path in CONFIG_SEARCH:
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver chipzen-seat`."""
    parser.add_argument("--run", required=True, help="Run id, fragment, or path to a run dir.")
    parser.add_argument(
        "--runs-dir",
        default=BOX_RUNS_DIR,
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
        choices=ENVIRONMENTS,
        help=f"Chipzen environment (default {DEFAULT_ENV}, where the division is "
        "actually deployed; tokens are minted per environment, so a staging token "
        "is not a prod one).",
    )
    parser.add_argument(
        "--no-resolver",
        action="store_true",
        help="Play the bare blueprint. The resolver otherwise follows "
        "`resolver.enabled` (on), which off-tree LBR puts 528 mbb/hand AHEAD of "
        "the bare blueprint since the shadow board-sync fix.",
    )
    parser.add_argument(
        "--budget-ms",
        type=int,
        default=None,
        help="Per-decision budget. Default sizes itself from the clock each match "
        f"says it enforces ({int(CLOCK_FRACTION * 100)}%% of it, less an overshoot "
        f"allowance), assuming the tight {TIGHT_CLOCK_MS} ms one when the frame "
        "does not say -- which is exactly when it is tight. The RESOLVER spends "
        "the whole budget, so this is how long each decision takes.",
    )
    parser.add_argument(
        "--policy-threshold",
        type=float,
        default=DEFAULT_POLICY_THRESHOLD,
        help="Zero every action below this share of its infoset's average "
        f"strategy, then renormalise (default {DEFAULT_POLICY_THRESHOLD}, which "
        "measured 940.1 -> 854.0 mbb/hand on the programme gate over three "
        "seeds). 0 disables it; 0.10 measured WORSE, so this is not a knob to "
        "turn up. Applied once at load, so the resolver reads the thresholded "
        "table too -- a composite the gate has NOT measured.",
    )
    parser.add_argument(
        "--rung",
        action="append",
        default=None,
        metavar="RUN[:AT]",
        help="A SHALLOWER blueprint to add to the depth ladder, repeatable. "
        "`--run` is the deepest rung; each `--rung` is another, and the seat "
        "plays whichever sits at or below the hand's effective stack. MEASURED "
        "over 40,000 duplicate deals, a native rung beats the 100 bb blueprint "
        "by 466 mbb/hand at 6 bb, 316 at 10 bb and 218 at 15 bb.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Play a single match and exit, rather than holding the seat.",
    )
    parser.add_argument(
        "--no-seek",
        action="store_true",
        help="Hold the lobby but never join the matchmaking queue, so the seat "
        "plays only matches someone else starts. The default JOINS: their SDK "
        "never does, and a bot that only waits measured zero matches in nine "
        "hours connected.",
    )


class ChipzenSeatPayload(BaseModel):
    """What was seated, and -- in replay mode -- what it decided."""

    op: Literal["chipzen-seat"] = "chipzen-seat"
    run: str
    run_dir: str
    runs_dir: str
    at_iteration: int | None = None
    mode: Literal["replay", "live"]
    resolver: bool | None = None
    """None follows `resolver.enabled`; False is an explicit `--no-resolver`."""
    budget_ms: int | None = None
    """None lets each match size it from the clock it enforces."""
    env: str | None = None
    bot_id: str | None = None
    rungs: list[str] = []
    """Shallower rungs beside `run`, as `run[:at]`. Empty is a single blueprint."""
    policy_threshold: float = DEFAULT_POLICY_THRESHOLD
    """Zeroed below this share of a row, at load. 0 fields the table as trained."""
    once: bool = False
    seek: bool = True
    """Whether to ask for matches. False is an explicit `--no-seek`."""
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
    configured = sdk_config_path()
    if not bot_id and not configured:
        raise CommandError(
            f"Pass --bot-id, set ${BOT_ENV}, or put `bot_id` under [external_api] "
            f"in {PREFERRED_CONFIG}. The bot's UUID is the last path segment when "
            "you open it at chipzen.ai/bots/<botId>."
        )
    if not os.environ.get(TOKEN_ENV) and not configured:
        raise CommandError(
            f"Set ${TOKEN_ENV} to a cz_extbot_ token, or put `token` under "
            f"[external_api] in {PREFERRED_CONFIG}. It is shown once at creation "
            "and cannot be read back."
        )
    return ChipzenSeatPayload(
        run=run_dir.name,
        run_dir=str(run_dir),
        runs_dir=args.runs_dir,
        at_iteration=args.at,
        mode="live",
        resolver=False if args.no_resolver else None,
        budget_ms=args.budget_ms,
        env=args.env,
        bot_id=bot_id,
        rungs=list(args.rung or []),
        policy_threshold=args.policy_threshold,
        once=args.once,
        seek=not args.no_seek,
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
    blueprint = _build_blueprint(run_dir, args.at, args.policy_threshold)
    seat = BlueprintSeat.for_match(
        blueprint,
        {"game_config": recorded["game_config"]},
        seat=int(recorded.get("seat", 0)),
        use_resolver=False if args.no_resolver else None,
        budget_ms=args.budget_ms,
    )
    decision = seat.decide_frame(recorded["state"])
    return ChipzenSeatPayload(
        run=run_dir.name,
        run_dir=str(run_dir),
        runs_dir=args.runs_dir,
        at_iteration=args.at,
        mode="replay",
        resolver=False if args.no_resolver else None,
        budget_ms=args.budget_ms,
        policy_threshold=args.policy_threshold,
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


def _build_blueprint(
    run_dir: Path,
    at_iteration: int | None,
    policy_threshold: float = 0.0,
    *,
    play_only: bool = False,
):
    """A run directory on local disk -> a blueprint. ~1 min in production.

    ``policy_threshold`` is spent HERE, once, rather than per decision: the
    resolver reads the blueprint for leaf values and range inference as well as
    for the fall-through action, and a transform applied at one call site would
    reach one of the three.
    """
    from src.engine.solver.policy.threshold import (  # noqa: PLC0415 -- see module docstring
        apply_policy_threshold,
    )
    from src.pipeline.services.scoring._shared import (  # noqa: PLC0415 -- see above
        build_blueprint_for,
    )
    from src.pipeline.training.run_tracker import RunTracker  # noqa: PLC0415 -- see above

    metadata = RunTracker.load(run_dir).metadata
    solver, storage, _policy = build_blueprint_for(
        run_dir,
        metadata,
        abstraction_hash=metadata.card_abstraction_hash,
        at_iteration=at_iteration,
        play_only=play_only,
    )
    if policy_threshold > 0.0:
        changed, trained = apply_policy_threshold(storage, solver.tree, policy_threshold)
        logger.info(
            "Policy threshold %.3f pruned %s of %s trained rows (%.1f%%).",
            policy_threshold,
            f"{changed:,}",
            f"{trained:,}",
            100.0 * changed / trained if trained else 0.0,
        )
    return solver


def _build_ladder(payload: ChipzenSeatPayload):
    """The deepest rung from `--run`, plus every `--rung`, as one ladder.

    Loaded `play_only`: a player reads `strategy_sum` and `visited` and nothing
    else, so the three resume-only arrays -- 1.23 GB of a 2.7 GB blueprint -- are
    never faulted in. That is what lets several rungs sit in memory at once.
    """
    from src.interfaces.chipzen.ladder import DepthLadder  # noqa: PLC0415 -- see above

    specs = [(Path(payload.run_dir), payload.at_iteration), *_parse_rungs(payload)]
    loaded = []
    for index, (run_dir, at) in enumerate(specs):
        try:
            loaded.append(_build_blueprint(run_dir, at, payload.policy_threshold, play_only=True))
        except Exception:
            # A SHALLOW rung that will not load costs coverage at that depth; the
            # seat playing on without it is strictly better than a seat that
            # cannot start. `--run` is different: losing it leaves nothing to
            # play, so that one is allowed to propagate and take the process
            # down, where `Restart=always` retries it.
            if index == 0:
                raise
            logger.exception(
                "Rung %s failed to load; the ladder continues without it.", run_dir.name
            )
    logger.info("Ladder built from %d of %d rung(s).", len(loaded), len(specs))
    return DepthLadder(loaded)


def _parse_rungs(payload: ChipzenSeatPayload) -> list[tuple[Path, int | None]]:
    out: list[tuple[Path, int | None]] = []
    for spec in payload.rungs:
        name, _, at = spec.partition(":")
        run_dir = resolve_run_dir(name, payload.runs_dir)
        out.append((run_dir, int(at) if at else None))
    return out


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
    """Hold the seat. Does not return until interrupted.

    Either credential may be ``None``: the SDK then reads it from the
    ``chipzen.toml`` :func:`sdk_config_path` already confirmed exists, which is
    how a file-based setup works without us ever parsing the file.
    """
    from src.interfaces.chipzen.seat import run_seat  # noqa: PLC0415 -- the play path

    run_dir = Path(payload.run_dir)
    print(
        f"Seating {payload.run} on chipzen {payload.env} as {payload.bot_id or 'the configured bot'}."
    )
    run_seat(
        (lambda: _build_ladder(payload))
        if payload.rungs
        else (lambda: _build_blueprint(run_dir, payload.at_iteration, payload.policy_threshold)),
        bot_id=payload.bot_id,
        token=os.environ.get(TOKEN_ENV),
        env=payload.env or DEFAULT_ENV,
        use_resolver=payload.resolver,
        budget_ms=payload.budget_ms,
        max_matches=1 if payload.once else None,
        seek_matches=payload.seek,
    )


COMMAND = Command(
    name="chipzen-seat",
    help="Play a run on chipzen.ai, or replay one recorded turn offline.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
