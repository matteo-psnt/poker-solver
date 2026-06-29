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
once at construction -- but the app can be asked to REPLACE it, which is what
``/api/load`` is for.

Switching used to mean a new process, and in practice a new deploy: an SSH
script that re-synced the code, re-ran ``uv sync``, rewrote the unit's
environment and restarted it, about three minutes. None of that has anything to
do with which run is loaded. Everything a swap actually needs is already on the
box -- the share is mounted, the runs directory is known -- so it is done in
process here, and the deploy script goes back to being for deploys.
"""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from src.core.game.state import Card
from src.engine.search.range_inference import ALL_COMBOS
from src.interfaces.blueprint.idle import IdleWatch
from src.interfaces.blueprint.sessions import Sessions, UnknownSessionError
from src.pipeline.blueprint.grid import StrategyGrid, strategy_grid
from src.pipeline.blueprint.paths import PathError, encode_action, match_action, replay

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
    """

    trained: bool
    strategy: list[float] | None
    reach_count: int


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


class SolverNode(BaseModel):
    """One spot in the tree. `grid` is null exactly when `terminal`."""

    path: str
    terminal: bool
    board: list[str]
    grid: NodeGrid | None
    children: list[Edge] = Field(default_factory=list)


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
    """The run being swapped in, while one is. Null when nothing is loading."""
    loading: str | None
    """False on a server handed a blueprint directly -- a laptop, a test."""
    can_switch: bool = False


class BlueprintLoad(BaseModel):
    """The 202 from asking for a swap. The work outlives the request."""

    run: str
    loading: bool


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
                reach_count=entry.reach_count,
            )
            for bucket, entry in grid.buckets.items()
        },
    )


class StartPlay(BaseModel):
    """A request to sit down. ``seed`` is for replaying a hand you want to study."""

    human_seat: int = 0
    button: int | None = None
    seed: int | None = None


class SubmitAction(BaseModel):
    """The human's move, as the same token the tree browser speaks."""

    token: str


class LoadRun(BaseModel):
    """Which run to serve instead, and optionally which checkpoint of it."""

    run: str
    at: int | None = None


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


class _Held:
    """The one run this server is currently answering for.

    A mutable holder rather than three closure variables, because ``/api/load``
    replaces all three together and a handler must never see a blueprint from
    one run beside sessions from another.
    """

    def __init__(self, blueprint: ScorableBlueprint, run_id: str) -> None:
        self.blueprint = blueprint
        self.run_id = run_id
        self.sessions = Sessions(blueprint)
        #: Set while a swap is in flight, so every reader can say so.
        self.loading: str | None = None
        #: Why the last swap failed, kept until the next one is attempted.
        self.error: str | None = None


def create_app(
    load_blueprint: Callable[[], ScorableBlueprint],
    *,
    run_id: str = "unknown",
    idle_timeout_seconds: float = 0.0,
    load_run: Callable[[str, int | None], tuple[ScorableBlueprint, str]] | None = None,
) -> FastAPI:
    """Build the app around one blueprint, loaded now.

    Eagerly, not lazily: a server that loads on first request answers its
    readiness check before it can serve anything, and the first caller pays a
    minute with no way to tell that from a hang.

    ``idle_timeout_seconds`` of 0 means stay up, which is what a laptop and a
    test want. On the hosted box it is what turns "nobody is here" into a
    stopped VM, via the systemd unit that escalates this process exiting.

    ``load_run`` is what makes ``/api/load`` possible: given a run id it stages
    and builds that run, returning the blueprint and the id it resolved to.
    OPTIONAL on purpose -- a test and a laptop serve a solver they built
    themselves and have no share to stage from, and for them the endpoint
    refuses with a sentence rather than 404ing as though it did not exist.
    """
    held = _Held(load_blueprint(), run_id)
    idle = IdleWatch(idle_timeout_seconds)
    # One swap at a time. A second caller is told the server is busy rather than
    # queued behind a minute of loading it cannot see.
    swapping = threading.Lock()

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
                "run": held.run_id,
                "ready": held.loading is None,
                "loading": held.loading,
                "can_switch": load_run is not None,
                "last_error": held.error,
                "idle_seconds": round(idle.idle_seconds(), 1),
                "idle_timeout_seconds": idle.timeout_seconds,
                "sessions": len(held.sessions),
            }
        )

    @app.get("/api/run")
    def _run() -> JSONResponse:
        """What is loaded here, so a client can label what it is looking at.

        Answers throughout a swap, describing the run STILL loaded and naming
        the one coming. It needs no guard now that the new blueprint is built
        before the old is released -- but it is worth saying why it once did:
        this is the endpoint a client polls to watch a load, so a null blueprint
        during one turned every switch into a stream of 500s on the page
        watching it.
        """
        config = held.blueprint.config
        return JSONResponse(
            BlueprintRun(
                run=held.run_id,
                starting_stack=config.game.starting_stack,
                small_blind=config.game.small_blind,
                big_blind=config.game.big_blind,
                combos=len(_COMBO_LABELS),
                loading=held.loading,
                can_switch=load_run is not None,
            ).model_dump()
        )

    def _swap(
        resolve: Callable[[str, int | None], tuple[ScorableBlueprint, str]],
        run: str,
        at_iteration: int | None,
    ) -> None:
        """Do the load. Runs on its own thread; never raises to the caller.

        Takes the resolver rather than closing over it: `/api/load` refuses
        before starting this thread when the server has none, and passing it in
        is what makes that refusal the only way in. Closed over, the guarantee
        was a `# type: ignore` on a call that is None on a server started with a
        blueprint handed to it directly.

        The new blueprint is BUILT before the old one is let go, so this server
        keeps answering from the run it already has for the whole minute-plus a
        load takes -- including the staging copy, which is thousands of small
        files off an SMB share and by far the longest part.

        The first version dropped the old one first, to avoid holding two tables
        at once. That was reasoning from a memory limit that does not exist here:
        the static table is allocated at full size and is FLAT IN ITERATION
        COUNT, so a 150M-iteration run costs exactly what a 30M one does, and the
        box idles at 14 of 15 GB free with one loaded. What it bought instead was
        a window where `held.blueprint` was None while `/api/run` -- the endpoint
        a client polls to WATCH the swap -- read straight through it.
        """
        try:
            blueprint, resolved = resolve(run, at_iteration)
            # Sessions die with the blueprint they were dealt from. Carrying one
            # across would leave a half-played hand whose next bot action comes
            # from a different run -- silently, and mid-hand.
            held.sessions.clear()
            held.blueprint = blueprint
            held.run_id = resolved
            held.sessions = Sessions(blueprint)
        except Exception as error:  # noqa: BLE001 -- reported, not swallowed
            held.error = f"{type(error).__name__}: {error}"
        finally:
            # Released only once the swap is fully installed, so a second caller
            # can never interleave with a half-applied one.
            held.loading = None
            swapping.release()

    @app.post("/api/load")
    def _load(request: LoadRun) -> JSONResponse:
        """Start replacing the loaded run, and return at once.

        202, not 200: the work takes about a minute and this answers in
        milliseconds. Holding the request open for the whole load would put a
        60-second HTTP call behind Caddy, the console's proxy and a browser tab,
        each with its own timeout and none of them 60 seconds -- so the load
        would keep succeeding while the caller saw a failure. Progress is on
        ``/api/health``, which is why that endpoint stays cheap and answering
        throughout.
        """
        if load_run is None:
            return JSONResponse(
                {
                    "error": "This server cannot switch runs: it was started with a "
                    "blueprint handed to it directly rather than a way to resolve one."
                },
                status_code=422,
            )
        if not swapping.acquire(blocking=False):
            return JSONResponse({"error": f"Already loading {held.loading}."}, status_code=409)

        held.loading = request.run
        held.error = None
        # `daemon`, so an idle expiry or a Ctrl-C during a load does not hang the
        # process waiting for a minute of numpy to finish.
        threading.Thread(
            target=_swap,
            args=(load_run, request.run, request.at),
            name="blueprint-swap",
            daemon=True,
        ).start()
        return JSONResponse(
            BlueprintLoad(run=request.run, loading=True).model_dump(), status_code=202
        )

    @app.get("/api/combos")
    def _combos() -> JSONResponse:
        """The canonical combo order the grid indexes into. Fetch once."""
        return JSONResponse(Combos(combos=list(_COMBO_LABELS)).model_dump())

    @app.get("/api/node")
    def _node(path: str = "", board: str = "", average: bool = True) -> JSONResponse:
        """The strategy at one spot, for every combo the board allows."""
        try:
            cards = parse_board(board)
            node = replay(held.blueprint, path, cards)
            if node.actor is None:
                return JSONResponse(
                    SolverNode(
                        path=path,
                        terminal=True,
                        board=[_card_text(card) for card in node.state.board],
                        grid=None,
                    ).model_dump()
                )
            grid = strategy_grid(held.blueprint, node, use_average=average)
            return JSONResponse(
                SolverNode(
                    path=path,
                    terminal=False,
                    board=[_card_text(card) for card in node.state.board],
                    grid=grid_payload(grid),
                    # The children are here so a client can walk the tree without
                    # guessing which sizes are legal: the menu is a function of
                    # the chip configuration, not of the action model alone.
                    children=[
                        Edge(
                            token=encode_action(action),
                            type=str(action.type),
                            amount=action.amount,
                        )
                        for action in node.legal_actions
                    ],
                ).model_dump()
            )
        except PathError as error:
            return JSONResponse({"error": str(error)}, status_code=422)

    @app.post("/api/play")
    def _start(request: StartPlay) -> JSONResponse:
        """Deal a hand. The button alternates unless the caller pins it."""
        try:
            session_id, hand = held.sessions.start(
                human_seat=request.human_seat, button=request.button, seed=request.seed
            )
        except ValueError as error:
            return JSONResponse({"error": str(error)}, status_code=422)
        return JSONResponse(hand_payload(hand, session_id).model_dump())

    @app.get("/api/play/{session_id}")
    def _hand(session_id: str) -> JSONResponse:
        try:
            return JSONResponse(
                hand_payload(held.sessions.get(session_id), session_id).model_dump()
            )
        except UnknownSessionError:
            return JSONResponse({"error": _GONE}, status_code=404)

    @app.post("/api/play/{session_id}/action")
    def _act(session_id: str, request: SubmitAction) -> JSONResponse:
        """Take the human's action, then auto-play to their next turn."""
        try:
            hand = held.sessions.get(session_id)
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
        held.sessions.drop(session_id)
        return JSONResponse(LeftSession(session=session_id, dropped=True).model_dump())

    return app
