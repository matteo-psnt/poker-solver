"""HTTP over :mod:`src.pipeline.analysis` -- transport only, no logic of its own.

Every handler here parses a request, calls one analysis function and returns its
result. Anything a caller could have got right raises
:class:`~src.pipeline.analysis.paths.PathError` and comes back as a 422 with the
sentence the analysis layer wrote; a bug still tracebacks. That split is the same
one `Command.invoke` makes, for the same reason -- a surface that greys out one
panel is more useful than one that dies.

The blueprint is supplied as a factory rather than loaded here, so a test can
serve a four-iteration solver through the identical app the node serves a 30M
one through. Loading takes ~1 minute and allocates the full table, so it happens
once at construction and the app holds it for its lifetime: this process is one
run, and switching runs means a new process.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from src.core.game.state import Card
from src.engine.search.heads_up_session import HeadsUpHand
from src.engine.search.range_inference import ALL_COMBOS
from src.engine.solver.policy.source import ScorableBlueprint
from src.interfaces.blueprint.idle import IdleWatch
from src.interfaces.blueprint.sessions import Sessions, UnknownSessionError
from src.pipeline.analysis.grid import StrategyGrid, strategy_grid
from src.pipeline.analysis.paths import PathError, encode_action, match_action, replay


# `repr`, not `str`: a Card's `__str__` is the pretty terminal form ("[ 2 ♣ ]")
# and its `__repr__` is the compact one ("2c"). Only the compact spelling round
# trips through `parse_board`, so the wire uses it everywhere -- a payload the
# server cannot read back is a dead end for links and bookmarks.
def _card_text(card: Card) -> str:
    """The compact spelling of a card, which is what the wire speaks."""
    return repr(card)


# Sent once and cached by the client forever: the canonical combo order never
# changes, and repeating 1326 labels on every node read would be most of the
# payload. The grid speaks in indices into this.
_COMBO_LABELS = tuple(f"{_card_text(a)}{_card_text(b)}" for a, b in ALL_COMBOS)


def parse_board(board: str) -> tuple[Card, ...]:
    """``"2c7d9h"`` or ``"2c 7d 9h"`` into cards, refusing anything else.

    A malformed board is the caller's to fix and would otherwise surface as a
    replay failure about the wrong thing entirely.
    """
    text = board.replace(" ", "").replace(",", "")
    if not text:
        return ()
    if len(text) % 2:
        raise PathError(f"'{board}' is not a whole number of cards.")
    cards = []
    for index in range(0, len(text), 2):
        token = text[index : index + 2]
        try:
            cards.append(Card.new(token))
        except Exception as error:
            raise PathError(f"'{token}' is not a card.") from error
    if len({card.mask for card in cards}) != len(cards):
        raise PathError(f"'{board}' repeats a card.")
    return tuple(cards)


def grid_payload(grid: StrategyGrid) -> dict[str, Any]:
    """The wire shape of a grid.

    ``buckets`` is keyed by string because JSON object keys are strings, and a
    client that had to guess whether "41" meant the int or the str would get it
    wrong exactly once.
    """
    return {
        "street": grid.street,
        "board": [_card_text(card) for card in grid.board],
        "actor": grid.actor,
        "actions": list(grid.actions),
        "combo_buckets": list(grid.combo_buckets),
        "blocked": grid.blocked,
        "trained_buckets": grid.trained_buckets,
        "buckets": {
            str(bucket): {
                "trained": entry.trained,
                "strategy": list(entry.strategy) if entry.strategy else None,
                "reach_count": entry.reach_count,
            }
            for bucket, entry in grid.buckets.items()
        },
    }


class StartPlay(BaseModel):
    """A request to sit down. ``seed`` is for replaying a hand you want to study."""

    human_seat: int = 0
    button: int | None = None
    seed: int | None = None


class SubmitAction(BaseModel):
    """The human's move, as the same token the tree browser speaks."""

    token: str


_GONE = (
    "That hand is no longer on the server — it ended, or was dropped to make room. Deal a new one."
)


def hand_payload(hand: HeadsUpHand) -> dict[str, Any]:
    """One hand's visible state.

    The bot's hole cards are withheld until the hand is over, which is not
    politeness: a client that received them could show them, and a sit-down where
    you can see the opponent's cards measures nothing at all.

    ``untrained`` on a bot decision is the field that matters most. A human
    explores far off the self-play distribution, so misses are common, and a bot
    playing uniform-random because it has NO strategy there must not be read as a
    bad blueprint.
    """
    state = hand.state
    return {
        "over": hand.is_over,
        "street": str(state.street),
        "board": [_card_text(card) for card in state.board],
        "pot": state.pot,
        "stacks": list(state.stacks),
        "human_seat": hand.human_seat,
        "button": hand.button,
        "to_act": None if hand.is_over else state.current_player,
        "hole_cards": [_card_text(card) for card in state.hole_cards[hand.human_seat]],
        "bot_hole_cards": (
            [_card_text(card) for card in state.hole_cards[1 - hand.human_seat]]
            if hand.is_over
            else None
        ),
        "legal": [
            {"token": encode_action(action), "type": str(action.type), "amount": action.amount}
            for action in hand.legal_actions()
        ],
        "payoff": hand.human_payoff() if hand.is_over else None,
        "showdown": hand.showdown,
        "bot_decisions": hand.bot_decisions,
        "bot_untrained_decisions": hand.bot_untrained_decisions,
        "log": [
            {
                "seat": event.seat,
                "actor": event.actor,
                "action": event.action_type,
                "amount": event.amount,
                "street": event.street,
                "untrained": event.untrained,
                # Revealed only at the end: the mix is what the bot WOULD have
                # done, and seeing it mid-hand is seeing the opponent's strategy.
                "mix": [[name, weight] for name, weight in event.mix]
                if (hand.is_over and event.mix)
                else None,
            }
            for event in hand.log
        ],
    }


def create_app(
    load_blueprint: Callable[[], ScorableBlueprint],
    *,
    run_id: str = "unknown",
    idle_timeout_seconds: float = 0.0,
) -> FastAPI:
    """Build the app around one blueprint, loaded now.

    Eagerly, not lazily: a server that loads on first request answers its
    readiness check before it can serve anything, and the first caller pays a
    minute with no way to tell that from a hang.

    ``idle_timeout_seconds`` of 0 means stay up, which is what a laptop and a
    test want. On the hosted box it is what turns "nobody is here" into a
    stopped VM, via the systemd unit that escalates this process exiting.
    """
    blueprint = load_blueprint()
    idle = IdleWatch(idle_timeout_seconds)
    sessions = Sessions(blueprint)

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        # Started here, not at construction: loading takes ~1 minute on a
        # production run, and a clock started before it would spend most of the
        # timeout waiting for the server to become able to answer at all.
        idle.touch()
        idle.start()
        yield
        idle.stop()

    app = FastAPI(title=f"blueprint server — {run_id}", docs_url="/api/docs", lifespan=lifespan)
    # Exposed so the caller can tell WHY the server stopped once uvicorn returns.
    # Idle expiry has to exit with its own code -- see `idle.IDLE_EXIT_CODE` --
    # and a SIGTERM is indistinguishable from `systemctl stop` by then.
    app.state.idle = idle

    @app.middleware("http")
    async def _touch(request: Request, call_next):
        """Every request counts as someone being here.

        Deliberately BEFORE the handler and unconditional -- a request that
        404s or 422s is still a person at the keyboard, and only counting
        successes would let a session spent hitting a stale link time out
        underneath them.
        """
        idle.touch()
        return await call_next(request)

    @app.get("/api/health")
    def _health() -> JSONResponse:
        """What is loaded, and how close this box is to switching itself off.

        The console polls this to decide whether it can show anything at all, so
        it must be the cheapest endpoint here -- no table reads, no session walk.
        """
        return JSONResponse(
            {
                "run": run_id,
                "ready": True,
                "idle_seconds": round(idle.idle_seconds(), 1),
                "idle_timeout_seconds": idle.timeout_seconds,
                "sessions": len(sessions),
            }
        )

    @app.get("/api/run")
    def _run() -> JSONResponse:
        """What is loaded here, so a client can label what it is looking at."""
        config = blueprint.config
        return JSONResponse(
            {
                "run": run_id,
                "starting_stack": config.game.starting_stack,
                "small_blind": config.game.small_blind,
                "big_blind": config.game.big_blind,
                "combos": len(_COMBO_LABELS),
            }
        )

    @app.get("/api/combos")
    def _combos() -> JSONResponse:
        """The canonical combo order the grid indexes into. Fetch once."""
        return JSONResponse({"combos": list(_COMBO_LABELS)})

    @app.get("/api/node")
    def _node(path: str = "", board: str = "", average: bool = True) -> JSONResponse:
        """The strategy at one spot, for every combo the board allows."""
        try:
            cards = parse_board(board)
            node = replay(blueprint, path, cards)
            if node.actor is None:
                return JSONResponse(
                    {
                        "path": path,
                        "terminal": True,
                        "board": [_card_text(card) for card in node.state.board],
                        "grid": None,
                        "children": [],
                    }
                )
            grid = strategy_grid(blueprint, node, use_average=average)
            return JSONResponse(
                {
                    "path": path,
                    "terminal": False,
                    "board": [_card_text(card) for card in node.state.board],
                    "grid": grid_payload(grid),
                    # The children are here so a client can walk the tree without
                    # guessing which sizes are legal: the menu is a function of
                    # the chip configuration, not of the action model alone.
                    "children": [
                        {
                            "token": encode_action(action),
                            "type": str(action.type),
                            "amount": action.amount,
                        }
                        for action in node.legal_actions
                    ],
                }
            )
        except PathError as error:
            return JSONResponse({"error": str(error)}, status_code=422)

    @app.post("/api/play")
    def _start(request: StartPlay) -> JSONResponse:
        """Deal a hand. The button alternates unless the caller pins it."""
        try:
            session_id, hand = sessions.start(
                human_seat=request.human_seat, button=request.button, seed=request.seed
            )
        except ValueError as error:
            return JSONResponse({"error": str(error)}, status_code=422)
        return JSONResponse({"session": session_id, **hand_payload(hand)})

    @app.get("/api/play/{session_id}")
    def _hand(session_id: str) -> JSONResponse:
        try:
            return JSONResponse({"session": session_id, **hand_payload(sessions.get(session_id))})
        except UnknownSessionError:
            return JSONResponse({"error": _GONE}, status_code=404)

    @app.post("/api/play/{session_id}/action")
    def _act(session_id: str, request: SubmitAction) -> JSONResponse:
        """Take the human's action, then auto-play to their next turn."""
        try:
            hand = sessions.get(session_id)
        except UnknownSessionError:
            return JSONResponse({"error": _GONE}, status_code=404)
        legal = hand.legal_actions()
        try:
            hand.submit(match_action(request.token, legal))
        except (PathError, ValueError) as error:
            # A refusal, not a fault: the client offered a move that is not on
            # the menu, or moved when it was not their turn.
            return JSONResponse({"error": str(error)}, status_code=422)
        return JSONResponse({"session": session_id, **hand_payload(hand)})

    @app.delete("/api/play/{session_id}")
    def _leave(session_id: str) -> JSONResponse:
        sessions.drop(session_id)
        return JSONResponse({"session": session_id, "dropped": True})

    return app
