"""HTTP over :mod:`src.pipeline.blueprint` -- transport only, no logic of its own.

Every handler here parses a request, calls one analysis function and returns its
result. Anything a caller could have got right raises
:class:`~src.pipeline.blueprint.paths.PathError` and comes back as a 422 with the
sentence the analysis layer wrote; a bug still tracebacks. That split is the same
one `Command.invoke` makes, for the same reason -- a surface that greys out one
panel is more useful than one that dies.

The blueprint is supplied as a factory rather than loaded here, so a test can
serve a four-iteration solver through the identical app the node serves a 30M
one through. Loading takes ~1 minute and allocates the full table, so it happens
once at construction.

ONE RUN PER PROCESS, and that is the point rather than a limitation. The box this
runs on holds a Chipzen ladder slot, so the strategy read here has to be the one
being fielded -- which makes the deploy that stages and seats a run the only
thing allowed to decide what is loaded, not a browser tab.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.core.game.state import Card
from src.engine.search.range_inference import ALL_COMBOS
from src.interfaces.blueprint.sessions import Sessions, UnknownSessionError
from src.pipeline.blueprint.grid import StrategyGrid, strategy_grid
from src.pipeline.blueprint.paths import (
    Decision,
    NeedsBoardError,
    PathError,
    Step,
    encode_action,
    match_action,
    replay,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.engine.search.heads_up_session import HeadsUpHand
    from src.engine.solver.policy.source import ScorableBlueprint


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


class Bucket(BaseModel):
    """One bucket's strategy at a spot.

    `strategy` is null exactly when `trained` is false: this server refuses to
    emit the uniform that an allocated-but-unvisited row would otherwise read
    as, and a client must be able to tell "never visited" from "plays uniform".
    There is no visit count: it read 0 on every row a PCS run ever served.
    """

    trained: bool
    strategy: list[float] | None


class NodeGrid(BaseModel):
    """The strategy at one spot, for every combo the board allows.

    ``buckets`` is keyed by STRING because JSON object keys are strings, and a
    client that had to guess whether "41" meant the int or the str would get it
    wrong exactly once.
    """

    street: str
    board: list[str]
    actor: int
    actions: list[str]
    """-1 where the board blocks the combo, so index i is always ALL_COMBOS[i]."""
    combo_buckets: list[int]
    blocked: int
    trained_buckets: int
    buckets: dict[str, Bucket]


class Edge(BaseModel):
    """One action on the menu: its path token, its type, and what it costs."""

    token: str
    type: str
    amount: float


class Spot(BaseModel):
    """One column of a line: somebody acted, and what else they could have.

    `options` carries the whole menu, not only what was taken, so a client can
    offer the alternatives at a past spot without a round trip per column.
    """

    kind: Literal["spot"] = "spot"
    street: str
    actor: int
    chosen: str
    options: list[Edge]
    pot: float
    stack: float


class Dealt(BaseModel):
    """The cards a street turned over, as the line crossed into it."""

    kind: Literal["dealt"] = "dealt"
    street: str
    cards: list[str]
    pot: float


class NeedsCards(BaseModel):
    """The line has reached a street and stopped, waiting to be dealt.

    An answer rather than a refusal, because the caller is not wrong: a line
    that crosses to the flop HAS no strategy until someone says which flop, and
    a client that is about to ask for three cards needs the line leading up to
    the question drawn first.
    """

    street: str
    #: Board cards the line needs in total by this street, and how many it has.
    needed: int
    have: int


class SolverNode(BaseModel):
    """One spot in the tree.

    `grid` is null when the hand is over (`terminal`) and when the board is
    short of what the line needs (`pending`) -- two different silences, which is
    why they are two fields rather than one null.
    """

    path: str
    terminal: bool
    board: list[str]
    grid: NodeGrid | None
    children: list[Edge] = Field(default_factory=list)
    """Which seat holds the button, so a client can NAME the seats rather than
    number them: replay pins it, and a seat index alone cannot be labelled."""
    button: int = 0
    """What is in the middle here, and what the player to act has left -- so a
    client can label this spot the way it labels every past one."""
    pot: float = 0.0
    stack: float | None = None
    """Every spot and deal between the start of the hand and here."""
    line: list[Spot | Dealt] = Field(default_factory=list)
    pending: NeedsCards | None = None


class HandEvent(BaseModel):
    """One move in the hand log."""

    seat: int
    actor: str
    action: str
    amount: float
    street: str
    """Whether the bot had NO strategy here and played uniform-random. A human
    explores far off the self-play distribution, so misses are common, and one
    must not be read as a bad blueprint."""
    untrained: bool
    """Null until the hand is over -- see :class:`Hand`."""
    mix: list[tuple[str, float]] | None


class Hand(BaseModel):
    """A hand in progress.

    `bot_hole_cards` and every `mix` are null until `over`, and that is not
    politeness: a client that received them could show them, and a sit-down
    where you can see the opponent's cards measures nothing at all.
    """

    session: str
    over: bool
    street: str
    board: list[str]
    pot: float
    stacks: list[float]
    human_seat: int
    button: int
    to_act: int | None
    hole_cards: list[str]
    bot_hole_cards: list[str] | None
    legal: list[Edge]
    payoff: float | None
    showdown: bool
    bot_decisions: int
    bot_untrained_decisions: int
    log: list[HandEvent]


class BlueprintRun(BaseModel):
    """What is loaded here, so a client can label what it is looking at."""

    run: str
    starting_stack: int
    small_blind: int
    big_blind: int
    combos: int
    # Every starting stack this server holds, deepest first. The chipzen box loads
    # the seat's whole depth ladder, so one process can answer for 100 bb down to
    # 3 bb; `starting_stack` above is the DEFAULT (deepest) and `/api/node?stack=`
    # selects any of these. A single-run server reports its one stack here.
    depths: list[int] = []
    # `stack -> run id`, so a client can say WHICH blueprint a depth came from
    # rather than implying the deepest run trained them all.
    runs_by_depth: dict[int, str] = {}


class Combos(BaseModel):
    """The canonical combo order the grid indexes into. Fetch once."""

    combos: list[str]


class LeftSession(BaseModel):
    """Confirmation that a play session was dropped.

    The session lives where the blueprint does, so leaving is a REQUEST rather
    than a local forget -- which is also why the console's proxy holds no state
    and a console restart does not lose a hand in progress.
    """

    session: str
    dropped: bool


def grid_payload(grid: StrategyGrid) -> NodeGrid:
    """The wire shape of a grid."""
    return NodeGrid(
        street=grid.street,
        board=[_card_text(card) for card in grid.board],
        actor=grid.actor,
        actions=list(grid.actions),
        combo_buckets=list(grid.combo_buckets),
        blocked=grid.blocked,
        trained_buckets=grid.trained_buckets,
        buckets={
            str(bucket): Bucket(
                trained=entry.trained,
                strategy=list(entry.strategy) if entry.strategy else None,
            )
            for bucket, entry in grid.buckets.items()
        },
    )


def _edge(action) -> Edge:
    """One action on the wire."""
    return Edge(token=encode_action(action), type=str(action.type), amount=action.amount)


def _line_payload(line: tuple[Step, ...]) -> list[Spot | Dealt]:
    """The walk, as the columns a client draws it in."""
    return [
        Spot(
            street=step.street,
            actor=step.actor,
            chosen=step.chosen,
            options=[_edge(action) for action in step.options],
            pot=step.pot,
            stack=step.stack,
        )
        if isinstance(step, Decision)
        else Dealt(
            street=step.street,
            cards=[_card_text(card) for card in step.cards],
            pot=step.pot,
        )
        for step in line
    ]


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


def hand_payload(hand: HeadsUpHand, session_id: str) -> Hand:
    """One hand's visible state.

    ``session_id`` is a PARAMETER rather than something the caller attaches
    afterwards. Each of the three endpoints that answer with a hand used to do
    `{"session": session_id, **hand_payload(hand)}`, so each had to remember --
    and a hand that shipped without one is silently unusable, because every
    subsequent action is addressed by it. Required here, forgetting is a type
    error.

    The bot's hole cards are withheld until the hand is over, which is not
    politeness: a client that received them could show them, and a sit-down where
    you can see the opponent's cards measures nothing at all.

    ``untrained`` on a bot decision is the field that matters most. A human
    explores far off the self-play distribution, so misses are common, and a bot
    playing uniform-random because it has NO strategy there must not be read as a
    bad blueprint.
    """
    state = hand.state
    return Hand(
        session=session_id,
        over=hand.is_over,
        street=str(state.street),
        board=[_card_text(card) for card in state.board],
        pot=state.pot,
        stacks=list(state.stacks),
        human_seat=hand.human_seat,
        button=hand.button,
        to_act=None if hand.is_over else state.current_player,
        hole_cards=[_card_text(card) for card in state.hole_cards[hand.human_seat]],
        bot_hole_cards=(
            [_card_text(card) for card in state.hole_cards[1 - hand.human_seat]]
            if hand.is_over
            else None
        ),
        legal=[
            Edge(token=encode_action(action), type=str(action.type), amount=action.amount)
            for action in hand.legal_actions()
        ],
        payoff=hand.human_payoff() if hand.is_over else None,
        showdown=hand.showdown,
        bot_decisions=hand.bot_decisions,
        bot_untrained_decisions=hand.bot_untrained_decisions,
        log=[
            HandEvent(
                seat=event.seat,
                actor=event.actor,
                action=event.action_type,
                amount=event.amount,
                street=event.street,
                untrained=event.untrained,
                # Revealed only at the end: the mix is what the bot WOULD have
                # done, and seeing it mid-hand is seeing the opponent's strategy.
                mix=[(name, weight) for name, weight in event.mix]
                if (hand.is_over and event.mix)
                else None,
            )
            for event in hand.log
        ],
    )


def create_app(
    load_blueprint: Callable[[], ScorableBlueprint | dict[int, ScorableBlueprint]],
    *,
    run_id: str = "unknown",
    run_ids: dict[int, str] | None = None,
) -> FastAPI:
    """Build the app around a blueprint, or a LADDER of them, loaded now.

    Eagerly, not lazily: a server that loads on first request answers its
    readiness check before it can serve anything, and the first caller pays a
    minute with no way to tell that from a hang. Which is also what makes
    reaching this server at all the readiness signal the deploy waits on.

    ``load_blueprint`` may return one blueprint or a ``{starting_stack: blueprint}``
    ladder. The chipzen box already stages every depth for the seat, so serving
    them all here costs the load, not another copy of the deepest run --
    measured 09-29 at 0.77 GB for one 100 bb blueprint under `play_only`.

    PLAY stays on the deepest rung deliberately. A session is stateful and its
    depth would have to be pinned for the whole hand; the chart is a read and can
    switch freely. Widening `Sessions` is a separate change with a separate
    failure mode.
    """
    loaded = load_blueprint()
    ladder: dict[int, ScorableBlueprint] = (
        dict(loaded) if isinstance(loaded, dict) else {loaded.config.game.starting_stack: loaded}
    )
    if not ladder:
        raise ValueError("create_app needs at least one blueprint")
    # Deepest first: it is the default, and the order a client should offer.
    depths = sorted(ladder, reverse=True)
    blueprint = ladder[depths[0]]
    names = dict(run_ids or {})
    sessions = Sessions(blueprint)

    def _pick(stack: int | None) -> ScorableBlueprint | None:
        """The blueprint for a stack, or None if this server does not hold it."""
        if stack is None:
            return blueprint
        return ladder.get(stack)

    app = FastAPI(title=f"blueprint server — {run_id}", docs_url="/api/docs")

    @app.get("/api/health")
    def _health() -> JSONResponse:
        """That this process is up, and what it is holding.

        The deploy polls this to know the load finished, so it must be the
        cheapest endpoint here -- no table reads, no session walk.
        """
        return JSONResponse({"run": run_id, "sessions": len(sessions)})

    @app.get("/api/run")
    def _run() -> JSONResponse:
        """What is loaded here, so a client can label what it is looking at."""
        config = blueprint.config
        return JSONResponse(
            BlueprintRun(
                run=run_id,
                starting_stack=config.game.starting_stack,
                small_blind=config.game.small_blind,
                big_blind=config.game.big_blind,
                combos=len(_COMBO_LABELS),
                depths=depths,
                runs_by_depth={stack: names.get(stack, run_id) for stack in depths},
            ).model_dump()
        )

    @app.get("/api/combos")
    def _combos() -> JSONResponse:
        """The canonical combo order the grid indexes into. Fetch once."""
        return JSONResponse(Combos(combos=list(_COMBO_LABELS)).model_dump())

    @app.get("/api/node")
    def _node(
        path: str = "", board: str = "", average: bool = True, stack: int | None = None
    ) -> JSONResponse:
        """The strategy at one spot, for every combo the board allows.

        ``stack`` selects a rung of the depth ladder; omitted means the deepest.
        A stack this server does not hold is a 422 that NAMES what it has, because
        the alternative -- silently answering from the deepest blueprint -- would
        label a 100 bb range as a 6 bb one and no reader could tell.
        """
        chosen = _pick(stack)
        if chosen is None:
            return JSONResponse(
                {"error": f"no blueprint at starting_stack={stack}; have {depths}"},
                status_code=422,
            )
        try:
            cards = parse_board(board)
            node = replay(chosen, path, cards)
        except NeedsBoardError as stopped:
            # Not a refusal: a line that crosses to the flop has no strategy
            # until someone says which flop. The walk so far comes back so the
            # client can draw the line leading up to the question it must ask.
            return JSONResponse(
                SolverNode(
                    path=path,
                    terminal=False,
                    board=[_card_text(card) for card in cards],
                    grid=None,
                    line=_line_payload(stopped.line),
                    pending=NeedsCards(
                        street=stopped.street, needed=stopped.needed, have=stopped.have
                    ),
                ).model_dump()
            )
        except PathError as error:
            return JSONResponse({"error": str(error)}, status_code=422)

        if node.actor is None:
            return JSONResponse(
                SolverNode(
                    path=path,
                    terminal=True,
                    board=[_card_text(card) for card in node.state.board],
                    grid=None,
                    line=_line_payload(node.line),
                ).model_dump()
            )
        grid = strategy_grid(chosen, node, use_average=average)
        return JSONResponse(
            SolverNode(
                path=path,
                terminal=False,
                board=[_card_text(card) for card in node.state.board],
                grid=grid_payload(grid),
                # The children are here so a client can walk the tree without
                # guessing which sizes are legal: the menu is a function of
                # the chip configuration, not of the action model alone.
                children=[_edge(action) for action in node.legal_actions],
                line=_line_payload(node.line),
                button=node.button,
                pot=node.state.pot,
                stack=node.state.stacks[node.actor],
            ).model_dump()
        )

    @app.post("/api/play")
    def _start(request: StartPlay) -> JSONResponse:
        """Deal a hand. The button alternates unless the caller pins it."""
        try:
            session_id, hand = sessions.start(
                human_seat=request.human_seat, button=request.button, seed=request.seed
            )
        except ValueError as error:
            return JSONResponse({"error": str(error)}, status_code=422)
        return JSONResponse(hand_payload(hand, session_id).model_dump())

    @app.get("/api/play/{session_id}")
    def _hand(session_id: str) -> JSONResponse:
        try:
            return JSONResponse(hand_payload(sessions.get(session_id), session_id).model_dump())
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
        return JSONResponse(hand_payload(hand, session_id).model_dump())

    @app.delete("/api/play/{session_id}")
    def _leave(session_id: str) -> JSONResponse:
        sessions.drop(session_id)
        return JSONResponse(LeftSession(session=session_id, dropped=True).model_dump())

    return app
