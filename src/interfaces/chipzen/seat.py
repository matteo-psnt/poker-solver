"""One blueprint, sat at one Chipzen table.

Split in two on purpose. :class:`BlueprintSeat` decides, and knows nothing about
sockets -- hand it the dict from a ``turn_request`` and it answers with the dict
that goes back, which is the whole contract and is testable without an account.
:func:`run_seat` is the socket half, and is deliberately thin: it imports their
SDK, which is an optional dependency, and does nothing a failure there could make
interesting.

The resolver follows ``resolver.enabled``, which ships on. It was defaulted OFF
here on the old off-tree collapse -- an exploiter beat blueprint+resolver by far
more than it beat the bare blueprint -- and that measurement is now void.
`3565aec` ungated `_sync_board` from `_diverged`, and off-tree LBR over 4,000
hands then put the shipped arm (alpha=0.35) at -781.6 against the bare
blueprint's -253.8: paired **-527.8, t=-2.84**, i.e. the resolver now HELPS
exactly where it used to hurt. Arena play is that setting -- roughly one off-tree
opponent action per hand -- so following the config is the measured choice and
``--no-resolver`` is the escape hatch.

One caveat that survives the fix: we reconstruct statelessly each turn and never
call ``observe``, so the resolver's range inference starts cold every decision.
The measurement above was made the same way, so it prices that in.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit

from src.interfaces.chipzen.adapter import (
    Spot,
    TableScale,
    reconstruct,
    table_scale,
    wire_action,
)
from src.interfaces.chipzen.ladder import DepthLadder
from src.interfaces.chipzen.protocol import GameConfig, ProtocolError, TurnState
from src.interfaces.errors import CommandError

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.game.actions import Action
    from src.engine.solver.policy.source import ScorableBlueprint

logger = logging.getLogger(__name__)

# The clock differs by 15x between paths -- 30 s on casual and the rated queue,
# 2 s on ranked challenges and tournaments -- so one constant cannot be right for
# both. A fixed 900 ms was measured at max 1695 ms against the 2 s clock: SAFE by
# the letter and 85% of it, which is luck rather than margin.
#
# BUDGET IS NOT A CAP -- the resolver spends all of it and then overshoots, and
# the overshoot is ADDITIVE rather than proportional: 900 ms budget ran 1695 ms
# worst (+795), 9000 ms ran 9380 ms worst (+380). So the headroom to keep is a
# constant, not a percentage, and a fraction alone wastes most of a long clock.
#
# Compute is cheap, but the CLOCK is not compute -- overrunning it forfeits. At
# 90% the seat took 27 s of a 30 s clock on every decision, ran 42 hands, then
# lost the socket and was refused on reconnect (403) with the match abandoned.
# Nine seconds per decision had completed a match cleanly the run before, so
# half the clock is a correction back toward the setting that demonstrably
# worked while still buying ~1.6x that thinking time.
#
# Whether more resolver time plays better is STILL not measured: the
# -527 mbb/hand justifying the resolver was taken near its shipped 300 ms.
TIGHT_CLOCK_MS = 2000
CLOCK_FRACTION = 0.50
OVERSHOOT_ALLOWANCE_MS = 800

# What a fixed default costs when nothing says otherwise. Assumes the TIGHT
# clock: `turn_timeout_ms` rides `match_start` on the relaxed paths and is
# omitted on the fast ones, so silence means fast. Guessing the generous clock
# is the guess that auto-folds.
DEFAULT_BUDGET_MS = max(50, int(TIGHT_CLOCK_MS * CLOCK_FRACTION) - OVERSHOOT_ALLOWANCE_MS)


# Warming compiles code paths; it does not need to think. Sizing it from the
# match budget cost a forfeited match: a 30 s clock gave a 9 s budget, the
# resolver spent all of it INSIDE `on_match_start` on the event loop thread, the
# lobby heartbeat starved, and the reconnect hit `duplicate_participant` against
# our own still-live socket. Fifty ms compiles exactly the same code.
WARM_BUDGET_MS = 50

# Zeroing the residue of early iterations measured 940.1 -> 854.0 mbb/hand on the
# programme gate (three seeds). 0.10 measured WORSE, so this is a verified point
# rather than a direction. See `engine/solver/policy/threshold.py`.
DEFAULT_POLICY_THRESHOLD = 0.02

# Depth bands, in big blinds, for the per-decision census. The edges are where
# the game changes shape rather than round numbers: below 10 bb a stack goes in
# preflop and the tree is still offering pot control, and 60% of the trained
# depth is where `SHALLOW_FRACTION` already draws the line.
DEPTH_BANDS = ((10.0, "<=10bb"), (25.0, "<=25bb"), (50.0, "<=50bb"))


def surface_sdk_logs(level: int | None = None) -> None:
    """Let the SDK's own logger through at the level we are running at.

    `configure_logging` attaches to the `src` package logger and cuts
    propagation, so `chipzen`'s records reach only Python's last-resort handler
    -- WARNING and above. That is why a 42-hand match reported
    `reconnect budget exhausted (...)` and NOT the three
    `reconnecting in Xs (attempt N/3; REASON)` lines before it, each of which
    carries the reason the socket closed. Without them a disconnect can only be
    guessed at, and one was: `closed without match_end` and a websocket
    exception read identically from the outside.

    Borrows our handler when there is one, and otherwise only lowers the level
    and leaves propagation alone -- under pytest that is what puts the records
    in front of `caplog` rather than nowhere.
    """
    ours = logging.getLogger("src")
    theirs = logging.getLogger("chipzen")
    theirs.setLevel(level if level is not None else (ours.level or logging.INFO))
    if ours.handlers and not theirs.handlers:
        for handler in ours.handlers:
            theirs.addHandler(handler)
        theirs.propagate = False


def _self_seat(match_info: dict[str, Any]) -> int:
    """Our seat, from the ``seats`` entry flagged ``is_self``.

    There is no ``your_seat`` on ``match_start`` -- the SDK derives ours the same
    way. Reading a field they do not send meant the seat was always built as 0
    and only corrected on the first ``decide``, so ``warm()`` ran as the wrong
    seat half the time.
    """
    for entry in match_info.get("seats") or ():
        if entry.get("is_self"):
            return int(entry.get("seat", 0))
    return 0


def budget_for(clock_ms: int | None) -> int:
    """A per-decision budget for the clock the server says it is enforcing.

    ``None`` means the frame did not carry one, which is itself the signal for a
    fast-clock match -- so it resolves to the tight assumption rather than to
    anything roomier.
    """
    clock = int(clock_ms) if clock_ms else TIGHT_CLOCK_MS
    return max(50, int(clock * CLOCK_FRACTION) - OVERSHOOT_ALLOWANCE_MS)


@dataclass
class SeatTally:
    """What a seat has seen, for the log line at the end of a match.

    ``off_tree`` counts opponent actions we had to snap to a legal size -- the
    drift between the table and the tree, and the first number to look at when
    arena results and home scores disagree.

    It is a per-hand MAXIMUM summed across hands, not a running total. Every turn
    replays that hand's whole history, so each reconstruction reports the count
    for the hand so far; adding those up charged the same action once per
    remaining decision and read 66-of-66 on the first live match. Within a hand
    the count only grows, so the last one seen is the true one.
    """

    decisions: int = 0
    truncated: int = 0
    fallbacks: int = 0
    per_hand: dict[int, int] = field(default_factory=dict)
    #: Decisions taken after the blinds escalated away from the seated level.
    escalated: int = 0
    #: Effective depth in big blinds, per hand. The blueprint is cut for ONE
    #: depth and an elimination match sweeps through many -- match 3 ran nine of
    #: twenty hands between 4.5 and 8 bb against a tree built for 100.
    depth_by_hand: dict[int, float] = field(default_factory=dict)
    #: DECISIONS per depth band, which is what sizes a ladder -- `depth_by_hand`
    #: counts a 40-decision hand and a 1-decision hand the same. Keyed by
    #: `DEPTH_BANDS`, plus the trained band for everything above them.
    by_band: dict[str, int] = field(default_factory=dict)
    #: Decisions taken below the trained depth. The replay that answers them is
    #: built from `blueprint.config.game.starting_stack` whatever the table
    #: holds, so each one is a 100 bb strategy fielded in a spot that is not one.
    out_of_tree: int = 0
    #: Times the ladder moved to a different rung. Zero on a one-rung ladder.
    rung_switches: int = 0

    @property
    def off_tree(self) -> int:
        return sum(self.per_hand.values())

    def saw(self, hand: int, off_tree: int) -> None:
        """Record this hand's off-tree count, keeping the largest seen for it."""
        self.per_hand[hand] = max(self.per_hand.get(hand, 0), off_tree)

    def summary(self) -> str:
        return (
            f"{self.decisions} decisions over {len(self.per_hand)} hands, "
            f"{self.off_tree} off-tree opponent actions, "
            f"{self.truncated} truncated replays, {self.fallbacks} safe defaults"
            + (f", {self.escalated} past a blind escalation" if self.escalated else "")
            + (
                f", depth {min(self.depth_by_hand.values()):.0f}-"
                f"{max(self.depth_by_hand.values()):.0f} bb"
                if self.depth_by_hand
                else ""
            )
            + (
                f", {self.out_of_tree}/{self.decisions} decisions below the "
                f"trained depth ({self.band_census()})"
                if self.out_of_tree
                else ""
            )
            + (f", {self.rung_switches} rung switches" if self.rung_switches else "")
        )

    def band_census(self) -> str:
        """Decisions per depth band, deepest first -- what a ladder has to cover."""
        return " ".join(f"{band}:{count}" for band, count in self.by_band.items())


@dataclass
class BlueprintSeat:
    """A blueprint answering Chipzen turns for one match.

    ``scale`` is fixed at ``match_start`` and never re-derived: the blinds
    escalate mid-match, and re-reading them per hand would silently re-denominate
    the table under us. What escalation actually changes is the effective depth,
    which is reported by :attr:`depth_matches` and is a property of the match, not
    of one hand.
    """

    blueprint: ScorableBlueprint
    config: GameConfig
    scale: TableScale
    seat: int
    #: Every rung available to this match. `blueprint` and `scale` are whichever
    #: one the CURRENT hand selected; a one-rung ladder behaves exactly as before.
    ladder: DepthLadder | None = None
    use_resolver: bool | None = None
    budget_ms: int = DEFAULT_BUDGET_MS
    tally: SeatTally = field(default_factory=SeatTally)
    #: Highest big blind seen this match. NOT the last one posted: a big blind
    #: shorter than the level is a player all-in for less, and reading that as
    #: the level divides the effective stack by too small a number and reports a
    #: shallow hand as a deep one. Measured live -- `Blinds escalated to 52
    #: (seated at 100)` is a short post, not a blind level that went backwards.
    blind_level: int = 0

    @classmethod
    def for_match(
        cls,
        blueprint: ScorableBlueprint | DepthLadder,
        match_info: dict[str, Any],
        seat: int,
        *,
        use_resolver: bool | None = None,
        budget_ms: int | None = None,
    ) -> BlueprintSeat:
        """Build a seat from ``match_start``, refusing a table we cannot denominate.

        ``budget_ms=None`` sizes the budget from ``match_start.turn_timeout_ms``,
        the clock this match enforces; absent means the fast one. An explicit
        value overrides it.
        """
        seats = match_info.get("seats") or ()
        config = GameConfig.parse(match_info["game_config"], seats=len(seats) or None)
        # `turn_timeout_ms`, THEIR name -- `decision_timeout_ms` appears nowhere
        # in the SDK and always read as None, so this sized nothing.
        clock = match_info.get("turn_timeout_ms")
        budget = budget_ms if budget_ms is not None else budget_for(clock)
        logger.info(
            "Clock %s ms; per-decision budget %s ms.",
            clock if clock else f"unstated (assuming {TIGHT_CLOCK_MS})",
            budget,
        )
        # A match STARTS at full stacks, so the deepest rung is the right one to
        # open with whatever the ladder holds; escalation moves it down from there.
        if isinstance(blueprint, DepthLadder):
            ladder: DepthLadder | None = blueprint
            chosen = blueprint.deepest.blueprint
        else:
            ladder = None
            chosen = blueprint
        scale = table_scale(config, chosen)
        if not scale.depth_matches:
            logger.warning(
                "Table is %.1f bb deep; this blueprint was trained at %.1f bb. "
                "Every spot below its depth is extrapolation.",
                scale.their_depth,
                scale.our_depth,
            )
        seated = cls(
            blueprint=chosen,
            config=config,
            scale=scale,
            seat=seat,
            ladder=ladder,
            use_resolver=use_resolver,
            budget_ms=budget,
        )
        seated.warm()
        return seated

    def warm(self) -> None:
        """Take one throwaway decision, so the first real one is not the slow one.

        Measured live: 4,109 ms for a match's opening decision and ~118 ms for
        every one after -- numba compiling and caches filling on first use. That
        is comfortable on the 30 s casual clock and fatal on the 2,000 ms ranked
        and tournament one, where hand one would auto-fold. Doing it here spends
        the cost inside ``match_start``, which has no per-decision clock on it.

        Never raises: a seat that cannot warm is still a seat that can play.
        """
        opening = {
            "hand_number": 0,
            "phase": "preflop",
            "board": [],
            "your_hole_cards": ["Ah", "Kd"],
            "pot": self.config.small_blind + self.config.big_blind,
            "your_stack": self.config.starting_stack - self.config.small_blind,
            "opponent_stacks": [self.config.starting_stack - self.config.big_blind],
            "to_call": self.config.big_blind - self.config.small_blind,
            "min_raise": 2 * self.config.big_blind,
            "max_raise": self.config.starting_stack,
            "action_history": [
                {
                    "seat": self.seat,
                    "action": "post_small_blind",
                    "amount": self.config.small_blind,
                    "phase": "preflop",
                    "is_timeout": False,
                },
                {
                    "seat": 1 - self.seat,
                    "action": "post_big_blind",
                    "amount": self.config.big_blind,
                    "phase": "preflop",
                    "is_timeout": False,
                },
            ],
        }
        started = time.perf_counter()
        budget, self.budget_ms = self.budget_ms, WARM_BUDGET_MS
        try:
            self.decide_frame(opening)
        except Exception:
            logger.exception("Warm-up decision failed; play continues cold.")
        finally:
            self.budget_ms = budget
            # The throwaway must not show up as a real decision.
            self.tally = SeatTally()
        logger.info("Warmed the decision path in %.0f ms.", (time.perf_counter() - started) * 1000)

    def decide_frame(self, state_payload: dict[str, Any]) -> dict[str, Any]:
        """The ``turn_action`` payload answering one ``turn_request.state``.

        Never raises, for anything. A decision that does not arrive is scored as
        a timeout fold, so a spot we cannot name is answered with the cheapest
        legal action instead and counted in the tally.

        The catches are deliberately bare: they were `(AdapterError,
        ProtocolError, ValueError)`, which is narrower than the promise -- the
        resolver and the numba kernels beneath `_choose` can surface a `KeyError`
        or an `IndexError`, and a seat index outside 0-1 would index a two-slot
        list. Each of those escaped into the SDK, which folded the hand under
        `safe_mode` and left the contract reading stronger than it was.
        """
        self.tally.decisions += 1
        try:
            turn = TurnState.parse(state_payload)
        except ProtocolError:
            logger.exception("Unreadable turn_request; folding.")
            self.tally.fallbacks += 1
            return {"action": "fold", "params": {}}

        try:
            spot = reconstruct(self.blueprint, turn, self.seat, self.scale)
        except Exception:
            logger.exception("Could not replay hand %s; passing.", turn.hand_number)
            self.tally.fallbacks += 1
            return self._pass(turn)

        self._note_depth(turn)
        self._note_escalation(turn)
        self.tally.saw(turn.hand_number, spot.off_tree)
        if spot.truncated:
            self.tally.truncated += 1
            logger.warning("Replay of hand %s did not land on our seat; passing.", turn.hand_number)
            self.tally.fallbacks += 1
            return self._pass(turn)

        try:
            chosen = self._choose(spot)
            return wire_action(chosen, turn, spot)
        except Exception:
            logger.exception("No usable action for hand %s; passing.", turn.hand_number)
            self.tally.fallbacks += 1
            return self._pass(turn)

    #: Below this fraction of the trained depth, a spot is a different game
    #: rather than a slightly shallower one -- at 5 bb the whole stack goes in
    #: preflop while the tree is still offering pot control.
    SHALLOW_FRACTION = 0.6

    def _depth(self, turn: TurnState) -> float | None:
        """Effective depth in big blinds, against the level rather than the post."""
        posted = turn.big_blind()
        if posted:
            self.blind_level = max(self.blind_level, posted)
        # The seated level is a FLOOR: blinds escalate and never come back, so a
        # posting below it is a short all-in rather than a level.
        level = max(self.blind_level, self.config.big_blind)
        effective = turn.effective_stack(self.seat)
        if effective is None or not level:
            return None
        return effective / level

    def _select_rung(self, depth: float, hand: int) -> None:
        """Point `blueprint` and `scale` at the rung that fits this hand.

        Depth is the EFFECTIVE stack, which is fixed once a hand starts, so this
        is stable within a hand and re-deriving it per turn cannot move the spot
        under us mid-hand.
        """
        if self.ladder is None:
            return
        rung = self.ladder.select(depth)
        if rung.blueprint is self.blueprint:
            return
        previous = self.scale.our_depth
        self.blueprint = rung.blueprint
        self.scale = table_scale(self.config, rung.blueprint)
        self.tally.rung_switches += 1
        logger.info(
            "Hand %s is %.1f bb: switching from the %.0f bb rung to the %.0f bb one.",
            hand,
            depth,
            previous,
            rung.depth,
        )

    def _census(self, depth: float) -> None:
        """Count this decision into its depth band, and against the trained one."""
        band = next(
            (name for edge, name in DEPTH_BANDS if depth <= edge),
            f">{DEPTH_BANDS[-1][0]:.0f}bb",
        )
        self.tally.by_band[band] = self.tally.by_band.get(band, 0) + 1
        # Against the SELECTED rung, so the count means "still off-tree after the
        # ladder had its say" rather than "off the deepest tree we own".
        if depth < self.scale.our_depth * self.SHALLOW_FRACTION:
            self.tally.out_of_tree += 1

    def _note_depth(self, turn: TurnState) -> None:
        """Record how deep this hand actually is, and say so when it is not ours.

        The blueprint is cut for ONE depth; an elimination match sweeps through
        many. Measured on a real 20-hand match: hands 1-11 ran 96-99 bb and hands
        12-19 ran 4.5-8 bb, all answered by a tree built for 100 bb. Nothing here
        fixes that -- it needs a ladder of blueprints -- but the seat should not
        be the last to know.
        """
        depth = self._depth(turn)
        if depth is None:
            return
        self._select_rung(depth, turn.hand_number)
        self._census(depth)
        first = turn.hand_number not in self.tally.depth_by_hand
        self.tally.depth_by_hand[turn.hand_number] = depth
        if first and depth < self.scale.our_depth * self.SHALLOW_FRACTION:
            logger.warning(
                "Hand %s is %.1f bb effective against a blueprint cut for %.0f bb. "
                "Playing it anyway, extrapolating.",
                turn.hand_number,
                depth,
                self.scale.our_depth,
            )

    def _note_escalation(self, turn: TurnState) -> None:
        """Say so when the blinds have moved off the level we were seated at.

        Their COMMON-PITFALLS #11: tournaments and longer matches escalate, the
        example breaking at hand 30 on 200/400. Nothing here can FIX it -- the
        blueprint is cut for one depth and a shallower table needs a shallower
        tree -- but `depth_matches` is computed once at `match_start`, so without
        this the seat plays a 100 bb strategy into a 25 bb spot and says nothing.
        Every match observed so far ran 8-20 hands at a flat level, which is why
        it has never fired.
        """
        big_blind = turn.big_blind()
        if big_blind is None or big_blind == self.config.big_blind:
            return
        self.tally.escalated += 1
        if self.tally.escalated == 1:
            # `(your_stack + pot) / bb` printed 385 bb on a 100 bb table: it is
            # our remaining stack plus the WHOLE pot, over one seat, and misses
            # what we have already committed. Effective depth is the quantity.
            depth = self._depth(turn) or 0.0
            logger.warning(
                "Blinds escalated to %s (seated at %s): about %.0f bb deep now, "
                "against a blueprint cut for %.0f. Play continues, extrapolating.",
                big_blind,
                self.config.big_blind,
                depth,
                self.scale.our_depth,
            )

    def _choose(self, spot: Spot) -> Action:
        """Ask the blueprint (or the resolver) what to do at ``spot``.

        ``use_resolver=None`` defers to ``resolver.enabled``, which is where the
        decision belongs -- one switch, not two that can disagree.
        """
        from src.engine.search.agent import BlueprintAgent  # noqa: PLC0415 -- see below

        agent = BlueprintAgent(self.blueprint, use_resolver=self.use_resolver)
        return agent.act(spot.state, time_budget_ms=self.budget_ms)

    @staticmethod
    def _pass(turn: TurnState) -> dict[str, Any]:
        """Check if it is free, otherwise fold -- the cheapest way to stay legal."""
        if turn.to_call <= 0:
            return {"action": "check", "params": {}}
        return {"action": "fold", "params": {}}


def sdk_state_payload(state: Any) -> dict[str, Any]:
    """Their SDK's typed ``GameState`` back into the wire dict we parse.

    The SDK hands ``decide()`` a parsed object; our adapter is written against
    the published payload so it can be tested from the spec's own examples. One
    conversion here is cheaper than two representations everywhere else.
    """
    return {
        "hand_number": state.hand_number,
        "phase": state.phase,
        "board": [str(card) for card in state.board],
        "your_hole_cards": [str(card) for card in state.hole_cards],
        "pot": state.pot,
        "your_stack": state.your_stack,
        "opponent_stacks": list(state.opponent_stacks),
        "to_call": state.to_call,
        "min_raise": state.min_raise,
        "max_raise": state.max_raise,
        "action_history": list(state.action_history),
        # Authoritative, and not derivable from the numbers above.
        "valid_actions": list(getattr(state, "valid_actions", ()) or ()),
    }


# Their queue entry EXPIRES -- `matchmaking/status` reports
# `queue_ttl_seconds: 60` -- so being queued is a thing you keep doing, not a
# thing you did. Re-joining at half the TTL leaves a whole period of slack for a
# slow round trip.
_QUEUE_TTL_FRACTION = 0.5
_QUEUE_MIN_PERIOD_S = 5.0


def _rest_base(lobby_url: str) -> str:
    """The REST origin behind a lobby WebSocket URL.

    Derived rather than mapped from ``env``: the SDK already owns that mapping
    and a second copy of it is a second thing to get wrong when they add a host.
    """
    parts = urlsplit(lobby_url)
    return f"{'https' if parts.scheme in ('wss', 'https') else 'http'}://{parts.netloc}"


async def _keep_queued(
    base: str,
    token: str,
    in_flight: Callable[[], int],
) -> None:
    """Keep asking for a match for as long as we are not in one.

    THE SEAT DOES NOT PLAY WITHOUT THIS. Holding the lobby only makes the bot
    reachable; `POST matchmaking/join` is what makes it wanted, and their SDK
    never calls it. Measured 08-26: nine hours connected, healthy, zero matches.

    Silent by design once steady -- it logs the queue's state only when that
    state changes, because a line every half minute forever is how a log stops
    being read.
    """
    import httpx  # noqa: PLC0415 -- only the live socket path needs a client

    headers = {"Authorization": f"Bearer {token}"}
    period = _QUEUE_MIN_PERIOD_S
    last_state: str | None = None

    async with httpx.AsyncClient(base_url=base, headers=headers, timeout=15.0) as http:
        while True:
            try:
                # Never queue while playing: a second match would be dealt into
                # the same process against the same in-memory table.
                if in_flight() > 0:
                    state = "playing"
                else:
                    status = (await http.get("/api/external-api/matchmaking/status")).json()
                    state = str(status.get("status", "unknown"))
                    ttl = float(status.get("queue_ttl_seconds") or 0.0)
                    period = max(_QUEUE_MIN_PERIOD_S, ttl * _QUEUE_TTL_FRACTION)
                    if state == "idle":
                        reply = await http.post("/api/external-api/matchmaking/join", json={})
                        reply.raise_for_status()
                        state = "queued"
            except Exception as exc:  # noqa: BLE001 -- a seat must outlive its queue
                # Their side being down must never take the socket with it: the
                # lobby can still be handed a challenge while the queue is out.
                if last_state != "error":
                    logger.warning("queue: unreachable (%s); still holding the lobby", exc)
                    last_state = "error"
            else:
                if state != last_state:
                    logger.info("queue: %s", state)
                    last_state = state
            await asyncio.sleep(period)


def run_seat(
    blueprint_factory,
    *,
    bot_id: str | None,
    token: str | None,
    env: str,
    use_resolver: bool | None = None,
    budget_ms: int | None = None,
    max_matches: int | None = None,
    seek_matches: bool = True,
) -> None:
    """Hold a seat on Chipzen until interrupted, or for ``max_matches`` matches.

    ``blueprint_factory`` is called HERE, before the lobby is dialled, and not at
    ``match_start``. It used to be the other way round, reasoning that their
    handshake would not wait for a load -- but the server starts a turn's clock
    when it SENDS the request, and `on_match_start` runs while that clock is
    already running. Measured on the box: 3.28 s to load the 300M checkpoint plus
    0.40 s to warm, which is the whole of the 4.1 s first decision seen in two
    live matches. Loading before connecting spends it where nothing is timing us.

    ``bot_id`` and ``token`` may each be ``None``, which hands that one to the
    SDK's own ``chipzen.toml`` discovery rather than to a default.
    """
    try:
        import importlib  # noqa: PLC0415 -- see below

        # Resolved by name so the checker behaves the same whether or not the
        # optional extra is installed: a plain `import chipzen` is an unresolved
        # import without it and a typed call site with it, and no single
        # suppression is correct in both. What the static check would have bought
        # is bought better by `tests/interfaces/chipzen/test_sdk_contract.py`,
        # which asserts against the SDK actually installed on this box.
        chipzen = importlib.import_module("chipzen")
    except ImportError as exc:
        raise CommandError(
            "The Chipzen SDK is not installed. `uv sync --extra chipzen`, or "
            "`pip install chipzen-bot`."
        ) from exc

    surface_sdk_logs()
    started = time.perf_counter()
    blueprint = blueprint_factory()
    # Warm EVERY rung against its OWN game -- a 1:1 table, so no real match is
    # needed to compile the decision path. What gets compiled does not depend on
    # the denomination, so the per-match warm that follows costs nothing. Every
    # rung rather than only the deepest: a match ESCALATES into the shallow ones,
    # and a rung first touched mid-match would pay its compile on a live clock.
    rungs = (
        [rung.blueprint for rung in blueprint.rungs]
        if isinstance(blueprint, DepthLadder)
        else [blueprint]
    )
    for one in rungs:
        game = one.config.game
        BlueprintSeat.for_match(
            one,
            {
                "game_config": {
                    "variant": "nlhe",
                    "starting_stack": game.starting_stack,
                    "small_blind": game.small_blind,
                    "big_blind": game.big_blind,
                    "ante": 0,
                    "num_players": 2,
                }
            },
            seat=0,
            use_resolver=use_resolver,
            budget_ms=budget_ms,
        )
    logger.info(
        "%d rung(s) loaded and warm in %.1f s, before dialling the lobby.",
        len(rungs),
        time.perf_counter() - started,
    )

    # Matches in flight. A COUNT, not a flag: the SDK runs each dispatched match
    # in its own task and a bracket can overlap two. The queue keeper reads it to
    # know when to stop asking for more.
    playing = 0

    class _Seat(chipzen.ChipzenBot):
        """Their lifecycle, our blueprint. Deliberately almost empty."""

        def __init__(self) -> None:
            self._seat: BlueprintSeat | None = None

        def on_match_start(self, match_info: dict) -> None:
            nonlocal playing
            playing += 1
            # Cheap now: the table is already in memory, so this is a config
            # parse and one warm decision, both inside the clock the server
            # started when it sent the first turn.
            self._seat = BlueprintSeat.for_match(
                blueprint,
                match_info,
                seat=_self_seat(match_info),
                use_resolver=use_resolver,
                budget_ms=budget_ms,
            )

        def decide(self, state: Any) -> Any:
            if self._seat is None:
                # No match_start reached us (a reconnect mid-match can do this).
                # Build the seat from what the turn itself carries.
                self.on_match_start({"game_config": _implied_config(state)})
            assert self._seat is not None
            self._seat.seat = state.your_seat
            frame = self._seat.decide_frame(sdk_state_payload(state))
            # Their Action is (action, amount) and builds the nested `params`
            # itself in `to_wire`; handing it our params dict is a TypeError.
            return chipzen.Action(
                action=frame["action"], amount=int(frame.get("params", {}).get("amount", 0))
            )

        def on_hand_result(self, result: dict) -> None:
            # One line per hand, at INFO, so a match leaves a record of HOW it
            # was won rather than only that it was. This is what makes a live
            # match reviewable against the replay on their site.
            logger.info("Hand: %s", json.dumps(result, default=str, sort_keys=True))

        def on_match_end(self, results: dict) -> None:
            nonlocal playing
            playing = max(0, playing - 1)
            # Logged whole rather than picked apart: the tally says the client
            # worked, and only this says whether the blueprint WON. Their
            # `match_end` shape is not in the specs we hold, so reading it as
            # JSON is both the report and the way to learn what is in it.
            if self._seat is not None:
                logger.info("Match over: %s", self._seat.tally.summary())
            logger.info("Result: %s", json.dumps(results, default=str, sort_keys=True))
            self._seat = None

    async def _hold() -> None:
        keeper = None
        if seek_matches:
            # Resolved the way the SDK resolves it, so the queue and the lobby
            # cannot end up pointed at different environments or bots. Missing
            # credentials are NOT raised here: `run_external_bot` below refuses
            # them with a message written for this, and a seat that cannot dial
            # the lobby has a bigger problem than an empty queue.
            config = chipzen.load_chipzen_config()
            seat_id = bot_id or (config.bot_id if config is not None else None)
            conn = (
                chipzen.connect_to_chipzen(seat_id, cast("Any", env), config=config)
                if seat_id
                else None
            )
            queue_token = (token if token is not None else conn.token) if conn else None
            if conn is not None and queue_token:
                keeper = asyncio.create_task(
                    _keep_queued(_rest_base(conn.url), queue_token, lambda: playing)
                )
            else:
                logger.warning("queue: no credentials to join with; expecting a challenge")
        try:
            await chipzen.run_external_bot(
                # The CLASS, not an instance: the SDK runs each dispatched match
                # in its own task and calls this per match. `_Seat` keeps mutable
                # per-match state, so one shared instance would let a second
                # match's `match_start` overwrite the first's table -- and its
                # `match_end` blank it, dropping the first back onto the
                # reconnect path mid-hand. A tournament bracket is exactly where
                # that happens.
                _Seat,
                bot_id=bot_id,
                # Their `env` is a Literal of three names. Narrowed by the
                # command's `choices=`, which is where a wrong one should be
                # refused -- with a usage message rather than a stack trace three
                # frames into an SDK.
                env=cast("Any", env),
                token=token,
                max_matches=max_matches,
            )
        finally:
            if keeper is not None:
                keeper.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await keeper

    asyncio.run(_hold())


def _implied_config(state: Any) -> dict[str, Any]:
    """A ``game_config`` inferred from a turn, for a reconnect that missed one.

    The blinds are recoverable from the synthetic entries every history opens
    with; the starting stack is not, so it is taken as the chips currently in
    play, which is right on hand one and an underestimate afterwards. Only the
    blind ratio feeds the scale, so an underestimate costs a depth warning rather
    than a mis-sized bet.
    """
    blinds = {
        entry.get("action"): int(entry.get("amount", 0))
        for entry in state.action_history
        if entry.get("action", "").startswith("post_")
    }
    small = blinds.get("post_small_blind", 0)
    big = blinds.get("post_big_blind", max(small * 2, 1))
    return {
        "variant": "nlhe",
        "starting_stack": state.your_stack + state.pot,
        "small_blind": small or max(big // 2, 1),
        "big_blind": big,
        "num_players": len(state.opponent_stacks) + 1,
    }
