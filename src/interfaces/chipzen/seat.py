"""One blueprint, sat at one Chipzen table.

Split in two on purpose. :class:`BlueprintSeat` decides, and knows nothing about
sockets -- hand it the dict from a ``turn_request`` and it answers with the dict
that goes back, which is the whole contract and is testable without an account.
:func:`run_seat` is the socket half, and is deliberately thin: it imports their
SDK, which is an optional dependency, and does nothing a failure there could make
interesting.

The resolver is OFF by default here, against ``resolver.enabled``'s own default.
Arena play is off-tree by construction -- opponents bet sizes our action model
never cut -- and off-tree is exactly where the resolver has been measured to
collapse: an exploiter beats blueprint+resolver by more than it beats the bare
blueprint. Reconstructing statelessly each turn would also starve its range
inference, which never sees a hand's actions in order. ``--resolver`` turns it
back on for anyone who wants to measure that again.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, cast

from src.interfaces.chipzen.adapter import (
    AdapterError,
    Spot,
    TableScale,
    reconstruct,
    table_scale,
    wire_action,
)
from src.interfaces.chipzen.protocol import GameConfig, ProtocolError, TurnState
from src.interfaces.errors import CommandError

if TYPE_CHECKING:
    from src.core.game.actions import Action
    from src.engine.solver.policy.source import ScorableBlueprint

logger = logging.getLogger(__name__)

# Their ranked and tournament clocks are 2000 ms round-trip. Leave room for the
# frame to get there and back, and for the resolver to be cut off rather than
# blown through -- a decision that lands late is a fold the server made for us.
DEFAULT_BUDGET_MS = 1200


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
        )


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
    use_resolver: bool = False
    budget_ms: int = DEFAULT_BUDGET_MS
    tally: SeatTally = field(default_factory=SeatTally)

    @classmethod
    def for_match(
        cls,
        blueprint: ScorableBlueprint,
        match_info: dict[str, Any],
        seat: int,
        *,
        use_resolver: bool = False,
        budget_ms: int = DEFAULT_BUDGET_MS,
    ) -> BlueprintSeat:
        """Build a seat from ``match_start``, refusing a table we cannot denominate."""
        config = GameConfig.parse(match_info["game_config"])
        scale = table_scale(config, blueprint)
        if not scale.depth_matches:
            logger.warning(
                "Table is %.1f bb deep; this blueprint was trained at %.1f bb. "
                "Every spot below its depth is extrapolation.",
                scale.their_depth,
                scale.our_depth,
            )
        seated = cls(
            blueprint=blueprint,
            config=config,
            scale=scale,
            seat=seat,
            use_resolver=use_resolver,
            budget_ms=budget_ms,
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
        try:
            self.decide_frame(opening)
        except Exception:
            logger.exception("Warm-up decision failed; play continues cold.")
        finally:
            # The throwaway must not show up as a real decision.
            self.tally = SeatTally()
        logger.info("Warmed the decision path in %.0f ms.", (time.perf_counter() - started) * 1000)

    def decide_frame(self, state_payload: dict[str, Any]) -> dict[str, Any]:
        """The ``turn_action`` payload answering one ``turn_request.state``.

        Never raises for a spot it cannot read. A decision that does not arrive
        is scored as a timeout fold, so a spot we cannot name is answered with
        the cheapest legal action instead and counted in the tally.
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
        except (AdapterError, ProtocolError, ValueError):
            logger.exception("Could not replay hand %s; passing.", turn.hand_number)
            self.tally.fallbacks += 1
            return self._pass(turn)

        self.tally.saw(turn.hand_number, spot.off_tree)
        if spot.truncated:
            self.tally.truncated += 1
            logger.warning("Replay of hand %s did not land on our seat; passing.", turn.hand_number)
            self.tally.fallbacks += 1
            return self._pass(turn)

        try:
            chosen = self._choose(spot)
            return wire_action(chosen, turn, spot)
        except (AdapterError, ValueError):
            logger.exception("No usable action for hand %s; passing.", turn.hand_number)
            self.tally.fallbacks += 1
            return self._pass(turn)

    def _choose(self, spot: Spot) -> Action:
        """Ask the blueprint (or the resolver) what to do at ``spot``."""
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


def run_seat(
    blueprint_factory,
    *,
    bot_id: str | None,
    token: str | None,
    env: str,
    use_resolver: bool = False,
    budget_ms: int = DEFAULT_BUDGET_MS,
    max_matches: int | None = None,
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
        import asyncio  # noqa: PLC0415 -- deferred with the optional SDK below
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

    started = time.perf_counter()
    blueprint = blueprint_factory()
    # Warm against the blueprint's OWN game -- a 1:1 table, so no real match is
    # needed to compile the decision path. What gets compiled does not depend on
    # the denomination, so the per-match warm that follows costs nothing.
    game = blueprint.config.game
    BlueprintSeat.for_match(
        blueprint,
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
        "Blueprint loaded and warm in %.1f s, before dialling the lobby.",
        time.perf_counter() - started,
    )

    class _Seat(chipzen.ChipzenBot):
        """Their lifecycle, our blueprint. Deliberately almost empty."""

        def __init__(self) -> None:
            self._seat: BlueprintSeat | None = None

        def on_match_start(self, match_info: dict) -> None:
            # Cheap now: the table is already in memory, so this is a config
            # parse and one warm decision, both inside the clock the server
            # started when it sent the first turn.
            self._seat = BlueprintSeat.for_match(
                blueprint,
                match_info,
                seat=int(match_info.get("your_seat", 0)),
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
            # Logged whole rather than picked apart: the tally says the client
            # worked, and only this says whether the blueprint WON. Their
            # `match_end` shape is not in the specs we hold, so reading it as
            # JSON is both the report and the way to learn what is in it.
            if self._seat is not None:
                logger.info("Match over: %s", self._seat.tally.summary())
            logger.info("Result: %s", json.dumps(results, default=str, sort_keys=True))
            self._seat = None

    asyncio.run(
        chipzen.run_external_bot(
            _Seat(),
            bot_id=bot_id,
            # Their `env` is a Literal of three names. Narrowed by the command's
            # `choices=`, which is where a wrong one should be refused -- with a
            # usage message rather than a stack trace three frames into an SDK.
            env=cast("Any", env),
            token=token,
            max_matches=max_matches,
        )
    )


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
