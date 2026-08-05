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
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from src.core.game.state import Card
from src.engine.search.range_inference import ALL_COMBOS
from src.engine.solver.policy_source import ScorableBlueprint
from src.pipeline.analysis.grid import StrategyGrid, strategy_grid
from src.pipeline.analysis.paths import PathError, encode_action, replay


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


def create_app(
    load_blueprint: Callable[[], ScorableBlueprint],
    *,
    run_id: str = "unknown",
) -> FastAPI:
    """Build the app around one blueprint, loaded now.

    Eagerly, not lazily: a server that loads on first request answers its
    readiness check before it can serve anything, and the first caller pays a
    minute with no way to tell that from a hang.
    """
    blueprint = load_blueprint()
    app = FastAPI(title=f"blueprint server — {run_id}", docs_url="/api/docs")

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

    return app
