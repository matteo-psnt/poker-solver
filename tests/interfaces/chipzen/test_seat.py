"""A seat must answer every turn, including the ones it cannot understand.

The server scores a decision that does not arrive as a timeout, and a timeout is
a fold -- so the property under test throughout is that ``decide_frame`` returns
a legal frame for any input at all, and records what it could not do rather than
raising. Anything else loses a match to an exception.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.interfaces.chipzen.protocol import TurnState
from src.interfaces.chipzen.seat import (
    CLOCK_FRACTION,
    MAX_BUDGET_MS,
    OVERSHOOT_ALLOWANCE_MS,
    TIGHT_CLOCK_MS,
    WARM_BUDGET_MS,
    BlueprintSeat,
    budget_for,
    sdk_state_payload,
    surface_sdk_logs,
)
from tests.interfaces.chipzen.test_adapter import (
    BB,
    BIG_BLIND_ENTRY,
    SB,
    SMALL_BLIND_ENTRY,
    STACK,
    raise_entry,
    turn_payload,
)
from tests.test_helpers import build_trained_test_solver

MATCH_INFO = {
    "game_config": {
        "variant": "nlhe",
        "starting_stack": 800,
        "small_blind": 100,
        "big_blind": 200,
        "num_players": 2,
    }
}

LEGAL = {"fold", "check", "call", "raise"}


@dataclass(frozen=True)
class SdkCard:
    """Stands in for their SDK's ``Card``, which ``str()`` renders back to notation."""

    text: str

    def __str__(self) -> str:
        return self.text


@pytest.fixture(scope="module")
def blueprint():
    return build_trained_test_solver(iterations=4)


@pytest.fixture
def seat(blueprint):
    """The BARE blueprint, explicitly.

    Most of this file is about the adapter and the fallbacks, and the resolver
    would only make each decision slow and non-deterministic. What the default
    IS gets its own test rather than riding along in every other one.
    """
    return BlueprintSeat.for_match(blueprint, MATCH_INFO, seat=0, use_resolver=False)


class TestSeatConstruction:
    def test_a_matching_table_scales_cleanly(self, seat):
        assert seat.scale.factor == 2
        assert seat.scale.depth_matches

    def test_a_shallower_table_still_seats(self, blueprint, caplog):
        shallow = {"game_config": {**MATCH_INFO["game_config"], "starting_stack": 400}}
        built = BlueprintSeat.for_match(blueprint, shallow, seat=0, use_resolver=False)
        assert not built.scale.depth_matches
        assert "trained at" in caplog.text

    def test_the_resolver_follows_the_config_by_default(self, blueprint):
        """`None` defers to `resolver.enabled` -- one switch, not two.

        It was hard-defaulted False on the old off-tree collapse; `3565aec`
        ungated the shadow board-sync and the shipped arm then measured 528
        mbb/hand AHEAD of the bare blueprint off-tree.
        """
        built = BlueprintSeat.for_match(blueprint, MATCH_INFO, seat=0, use_resolver=False)
        assert built.use_resolver is False
        assert BlueprintSeat.__dataclass_fields__["use_resolver"].default is None


class TestBudget:
    """One constant cannot serve clocks that differ by 15x.

    Casual and the rated queue allow 30 s; ranked challenges and tournaments
    allow 2 s. The overshoot is ADDITIVE, not proportional -- 900 ms ran 1695 ms
    worst (+795) and 9000 ms ran 9380 ms (+380) -- so the headroom kept is a
    constant rather than a fraction.

    A roomy clock no longer buys all of that room: `MAX_BUDGET_MS` caps it, so
    that a match on the generous clock cannot hold the GIL long enough to time
    out a concurrent match on the tight one.
    """

    def test_a_stated_clock_sizes_the_budget(self, blueprint):
        """THEIR name is `turn_timeout_ms`; `decision_timeout_ms` is not a field
        the SDK ever sends, so pinning it made this test assert nothing."""
        from src.interfaces.chipzen.seat import MAX_BUDGET_MS

        relaxed = {**MATCH_INFO, "turn_timeout_ms": 30_000}
        built = BlueprintSeat.for_match(blueprint, relaxed, seat=0, use_resolver=False)
        assert built.budget_ms == MAX_BUDGET_MS

    def test_an_unstated_clock_assumes_the_tight_one(self, blueprint):
        """`decision_timeout_ms` is absent on exactly the fast-clock matches."""
        built = BlueprintSeat.for_match(blueprint, MATCH_INFO, seat=0, use_resolver=False)
        assert built.budget_ms == 200

    def test_an_explicit_budget_wins(self, blueprint):
        relaxed = {**MATCH_INFO, "turn_timeout_ms": 30_000}
        built = BlueprintSeat.for_match(
            blueprint, relaxed, seat=0, use_resolver=False, budget_ms=250
        )
        assert built.budget_ms == 250

    @pytest.mark.parametrize(
        ("clock", "expected"),
        [(2000, 200), (30_000, 900), (None, 200), (0, 200), (100, 50)],
    )
    def test_the_rule_across_clocks(self, clock, expected):
        assert budget_for(clock) == expected

    def test_the_worst_case_stays_clear_of_the_tight_clock(self):
        """Budget plus the measured additive overshoot must fit the clock."""
        assert budget_for(None) + OVERSHOOT_ALLOWANCE_MS <= TIGHT_CLOCK_MS * CLOCK_FRACTION

    def test_a_long_clock_still_buys_more_than_the_measured_setting(self):
        """The resolver's 528 mbb/hand gain was measured near a 300 ms budget.

        The cap has to stay clear of THAT, not of the clock: spending 14.2 s was
        never measured to beat 300 ms, it was only what the clock allowed.
        """
        assert budget_for(30_000) > 300
        assert budget_for(30_000) > budget_for(TIGHT_CLOCK_MS)

    @pytest.mark.parametrize("clock", [2000, 5000, 30_000])
    def test_the_worst_case_never_exceeds_half_the_clock(self, clock):
        """At 90% the seat took 27 s of a 30 s clock and lost the match on a
        refused reconnect. The margin is the point, not the leftover."""
        assert budget_for(clock) + OVERSHOOT_ALLOWANCE_MS <= clock * 0.5


class TestSdkLogsAreVisible:
    """A disconnect must diagnose itself rather than be inferred an hour later.

    A 42-hand match ended with `reconnect budget exhausted (...)` and nothing
    else, because `configure_logging` cuts propagation on the `src` logger and
    `chipzen`'s records fell through to Python's last-resort handler at WARNING.
    The three `reconnecting in Xs (attempt N/3; REASON)` lines that carry the
    close reason were dropped, so `closed without match_end` and a websocket
    exception were indistinguishable from the outside.
    """

    def test_the_sdk_logger_is_lowered_to_our_level(self):
        surface_sdk_logs(logging.INFO)
        assert logging.getLogger("chipzen").level == logging.INFO

    def test_an_info_record_from_the_sdk_survives(self, caplog):
        """WARNING already got through; INFO is the one that was being lost."""
        surface_sdk_logs(logging.INFO)
        with caplog.at_level(logging.INFO, logger="chipzen"):
            logging.getLogger("chipzen").info(
                "reconnecting in 1.0s (attempt 1/3; closed without match_end)"
            )
        assert "closed without match_end" in caplog.text

    def test_our_handler_is_borrowed_when_there_is_one(self):
        """On the box `configure_logging` has run, so records go to our stream."""
        ours = logging.getLogger("src")
        theirs = logging.getLogger("chipzen")
        added = logging.NullHandler()
        ours.addHandler(added)
        theirs.handlers.clear()
        try:
            surface_sdk_logs(logging.INFO)
            assert added in theirs.handlers
            assert theirs.propagate is False, "borrowed, so it must not double-print"
        finally:
            ours.removeHandler(added)
            theirs.handlers.clear()
            theirs.propagate = True


class TestEffectiveDepth:
    """The mismatch that actually matters, measured off a real match.

    Match 3 ran twenty hands: 1-11 at 96-99 bb and 12-19 at 4.5-8 bb, every one
    answered by a tree cut for 100 bb. Nine of twenty hands -- 45% -- were a
    different game. Nothing here fixes it; a ladder of blueprints does. This is
    about the seat knowing, and saying.
    """

    def shortstacked(self, mine, theirs):
        """A preflop turn where the two seats hold uneven stacks."""
        return turn_payload(
            hand_number=12,
            your_stack=mine - SB,
            opponent_stacks=[theirs - BB],
            pot=SB + BB,
        )

    def test_the_depth_is_the_shorter_stack(self, seat):
        """Nobody can win or lose more than the shorter stack."""
        seat.decide_frame(self.shortstacked(mine=19_300, theirs=700))
        assert seat.tally.depth_by_hand[12] == pytest.approx(700 / BB)

    def test_an_even_table_is_its_own_depth(self, seat):
        seat.decide_frame(turn_payload())
        assert seat.tally.depth_by_hand[1] == pytest.approx(STACK / BB)

    def test_a_shallow_hand_warns_once(self, seat, caplog):
        """The test table is only 4 bb, so "shallow" here is under 2.4 bb."""
        assert seat.scale.our_depth == 4.0
        with caplog.at_level(logging.WARNING, logger="src.interfaces.chipzen.seat"):
            for _ in range(3):
                seat.decide_frame(self.shortstacked(mine=1500, theirs=300))
        assert caplog.text.count("effective against a blueprint") == 1

    def test_a_hand_at_our_depth_says_nothing(self, seat, caplog):
        with caplog.at_level(logging.WARNING, logger="src.interfaces.chipzen.seat"):
            seat.decide_frame(turn_payload())
        assert "effective against a blueprint" not in caplog.text

    def test_the_summary_reports_the_range(self, seat):
        seat.decide_frame(turn_payload())
        seat.decide_frame(self.shortstacked(mine=1500, theirs=300))
        summary = seat.tally.summary()
        assert "depth" in summary
        assert "bb" in summary

    def test_committed_chips_count_toward_the_hand_start_stack(self):
        """`your_stack` is what is LEFT; the depth is what was there."""
        turn = TurnState.parse(
            turn_payload(
                your_stack=STACK - 600,
                opponent_stacks=[STACK - 600],
                action_history=[
                    SMALL_BLIND_ENTRY,
                    BIG_BLIND_ENTRY,
                    raise_entry(0, 600),
                    raise_entry(1, 600),
                ],
            )
        )
        assert turn.committed(0) == 600
        assert turn.effective_stack(0) == STACK
        assert turn.depth_in_blinds(0) == pytest.approx(STACK / BB)

    def test_no_opponent_stack_means_no_claim(self):
        turn = TurnState.parse(turn_payload(opponent_stacks=[]))
        assert turn.effective_stack(0) is None
        assert turn.depth_in_blinds(0) is None


class TestBlindEscalation:
    """A shallower table than the one we sat down at must not pass unremarked.

    Their COMMON-PITFALLS #11 -- tournaments and longer matches escalate, their
    example breaking at hand 30 on 200/400. Nothing here fixes it: the blueprint
    is cut for one depth. But `depth_matches` is computed once at `match_start`,
    so before this the seat played a 100 bb strategy into a 25 bb spot silently.
    """

    def escalated_turn(self, big_blind):
        small = big_blind // 2
        return turn_payload(
            hand_number=30,
            pot=small + big_blind,
            to_call=big_blind - small,
            min_raise=2 * big_blind,
            action_history=[
                {**SMALL_BLIND_ENTRY, "amount": small},
                {**BIG_BLIND_ENTRY, "amount": big_blind},
            ],
        )

    def test_the_seated_level_is_not_flagged(self, seat):
        seat.decide_frame(self.escalated_turn(200))
        assert seat.tally.escalated == 0

    def test_a_raised_level_is_counted_and_warned_once(self, seat, caplog):
        with caplog.at_level(logging.WARNING, logger="src.interfaces.chipzen.seat"):
            for _ in range(3):
                seat.decide_frame(self.escalated_turn(800))
        assert seat.tally.escalated == 3
        assert caplog.text.count("Blinds escalated") == 1, "warn once, count every time"

    def test_the_summary_says_so(self, seat):
        seat.decide_frame(self.escalated_turn(800))
        assert "past a blind escalation" in seat.tally.summary()

    def test_a_quiet_match_says_nothing_about_it(self, seat):
        seat.decide_frame(turn_payload())
        assert "escalation" not in seat.tally.summary()


class TestWarmUp:
    """The opening decision must not be the slow one.

    Measured live: 4,109 ms for a match's first decision, ~118 ms thereafter.
    Comfortable on the 30 s casual clock, an auto-fold on the 2,000 ms ranked and
    tournament one.
    """

    def test_warming_does_not_spend_the_match_budget(self, blueprint):
        """It compiles code paths; it does not need to think.

        Sizing it from the budget forfeited a live match: a 30 s clock gave a
        9 s budget, the resolver spent all of it inside `on_match_start` on the
        event loop, the lobby heartbeat starved, and the reconnect collided with
        our own still-live socket as `duplicate_participant`. The budget is
        capped now, but WARM_BUDGET_MS must still be below it rather than equal.
        """
        relaxed = {**MATCH_INFO, "turn_timeout_ms": 30_000}
        seen: list[int] = []
        original = BlueprintSeat.decide_frame

        def record(self, payload):
            seen.append(self.budget_ms)
            return original(self, payload)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(BlueprintSeat, "decide_frame", record)
            built = BlueprintSeat.for_match(blueprint, relaxed, seat=0, use_resolver=False)

        assert seen == [WARM_BUDGET_MS], "the warm decision must not use the match budget"
        assert built.budget_ms == MAX_BUDGET_MS, "and the match budget must survive it"

    def test_a_seat_arrives_already_warm(self, blueprint, caplog):
        with caplog.at_level(logging.INFO, logger="src.interfaces.chipzen.seat"):
            BlueprintSeat.for_match(blueprint, MATCH_INFO, seat=0, use_resolver=False)
        assert "Warmed the decision path" in caplog.text

    def test_the_throwaway_leaves_no_trace_in_the_tally(self, seat):
        """Otherwise every match would report a phantom first decision."""
        assert seat.tally.decisions == 0
        assert seat.tally.fallbacks == 0
        assert seat.tally.per_hand == {}

    def test_the_first_real_decision_is_still_answered(self, seat):
        assert seat.decide_frame(turn_payload())["action"] in LEGAL
        assert seat.tally.decisions == 1

    def test_a_seat_that_cannot_warm_still_plays(self, blueprint, monkeypatch, caplog):
        """Warming is best-effort; a failure there must not cost us the match."""
        monkeypatch.setattr(
            BlueprintSeat,
            "decide_frame",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("cold")),
        )
        built = BlueprintSeat.for_match(blueprint, MATCH_INFO, seat=0, use_resolver=False)
        assert "play continues cold" in caplog.text
        assert built.tally.decisions == 0


class TestDecide:
    def test_the_opening_spot_produces_a_legal_frame(self, seat):
        frame = seat.decide_frame(turn_payload())
        assert frame["action"] in LEGAL
        assert seat.tally.decisions == 1
        assert seat.tally.fallbacks == 0

    def test_a_raise_it_chooses_is_within_the_servers_bounds(self, seat):
        turn = turn_payload()
        for _ in range(20):
            frame = seat.decide_frame(turn)
            if frame["action"] == "raise":
                amount = frame["params"]["amount"]
                assert turn["min_raise"] <= amount <= turn["max_raise"]

    def test_a_spot_facing_a_bet_still_answers(self, seat):
        frame = seat.decide_frame(
            turn_payload(
                your_hole_cards=["Qs", "Qc"],
                to_call=400,
                action_history=[SMALL_BLIND_ENTRY, BIG_BLIND_ENTRY, raise_entry(0, 600)],
            )
        )
        assert frame["action"] in LEGAL

    def test_an_off_tree_opponent_size_is_tallied(self, blueprint):
        built = BlueprintSeat.for_match(blueprint, MATCH_INFO, seat=1, use_resolver=False)
        built.decide_frame(
            turn_payload(
                your_hole_cards=["Qs", "Qc"],
                to_call=330,
                action_history=[SMALL_BLIND_ENTRY, BIG_BLIND_ENTRY, raise_entry(0, 530)],
            )
        )
        assert built.tally.off_tree == 1

    def test_the_same_hand_answered_twice_counts_its_drift_once(self, blueprint):
        """Every turn replays the whole hand, so summing per-turn double-counts.

        Read 66-of-66 on the first live match -- one charge per remaining
        decision for the same handful of snapped actions.
        """
        built = BlueprintSeat.for_match(blueprint, MATCH_INFO, seat=1, use_resolver=False)
        payload = turn_payload(
            your_hole_cards=["Qs", "Qc"],
            to_call=330,
            action_history=[SMALL_BLIND_ENTRY, BIG_BLIND_ENTRY, raise_entry(0, 530)],
        )
        for _ in range(5):
            built.decide_frame(payload)
        assert built.tally.decisions == 5
        assert built.tally.off_tree == 1

    def test_two_hands_each_contribute_their_own(self, blueprint):
        built = BlueprintSeat.for_match(blueprint, MATCH_INFO, seat=1, use_resolver=False)
        for hand in (1, 2):
            built.decide_frame(
                turn_payload(
                    hand_number=hand,
                    your_hole_cards=["Qs", "Qc"],
                    to_call=330,
                    action_history=[SMALL_BLIND_ENTRY, BIG_BLIND_ENTRY, raise_entry(0, 530)],
                )
            )
        assert built.tally.off_tree == 2


class TestItNeverRaises:
    def test_an_unparseable_payload_folds(self, seat):
        assert seat.decide_frame({"nonsense": True}) == {"action": "fold", "params": {}}
        assert seat.tally.fallbacks == 1

    def test_a_history_with_no_blinds_passes_rather_than_raising(self, seat):
        frame = seat.decide_frame(turn_payload(action_history=[]))
        assert frame["action"] in LEGAL
        assert seat.tally.fallbacks == 1

    def test_a_free_pass_checks_and_a_costly_one_folds(self, seat):
        """The fallback is the cheapest legal action, not always a fold."""
        assert seat.decide_frame(turn_payload(action_history=[], to_call=0))["action"] == "check"
        assert seat.decide_frame(turn_payload(action_history=[], to_call=50))["action"] == "fold"

    def test_a_turn_for_the_seat_that_is_not_ours_passes(self, blueprint):
        """Their frame says it is our turn; if our replay disagrees, do not guess."""
        built = BlueprintSeat.for_match(blueprint, MATCH_INFO, seat=1, use_resolver=False)
        frame = built.decide_frame(turn_payload())
        assert frame["action"] in LEGAL
        assert built.tally.truncated == 1

    @pytest.mark.parametrize(
        "broken",
        [
            {"phase": "showdown"},
            {"your_hole_cards": ["Ah"]},
            {"your_hole_cards": ["Zz", "Kd"]},
        ],
    )
    def test_a_malformed_field_folds_rather_than_propagating(self, seat, broken):
        frame = seat.decide_frame(turn_payload(**broken))
        assert frame["action"] in LEGAL


class TestSdkConversion:
    def test_a_typed_state_becomes_the_wire_dict(self):
        """Their SDK hands ``decide()`` cards as objects; the wire wants strings."""
        state = SimpleNamespace(
            hand_number=3,
            phase="flop",
            board=[SdkCard("2c"), SdkCard("7d"), SdkCard("9h")],
            hole_cards=[SdkCard("Ah"), SdkCard("Kd")],
            pot=400,
            your_stack=600,
            opponent_stacks=[600],
            to_call=0,
            min_raise=200,
            max_raise=600,
            action_history=[SMALL_BLIND_ENTRY],
        )
        payload = sdk_state_payload(state)
        assert payload["board"] == ["2c", "7d", "9h"]
        assert payload["your_hole_cards"] == ["Ah", "Kd"]
        assert payload["hand_number"] == 3
        assert payload["action_history"] == [SMALL_BLIND_ENTRY]


class TestDepthAgainstTheLevelRatherThanThePost:
    """A big blind shorter than the level is a player all-in, not a level.

    `big_blind()` reads the posted amount, so a short all-in blind divides the
    effective stack by too small a number and reports the shallowest hands in a
    match as the deepest. Live evidence: `Blinds escalated to 52 (seated at
    100)` -- a level that appeared to go backwards, which cannot happen.
    """

    def short_blind_turn(self, posted):
        """Both seats 400 deep behind a big blind posted for ``posted``."""
        return turn_payload(
            hand_number=7,
            your_stack=300,
            opponent_stacks=[400 - posted],
            pot=SB + posted,
            action_history=[SMALL_BLIND_ENTRY, {**BIG_BLIND_ENTRY, "amount": posted}],
        )

    def test_a_short_post_does_not_deepen_the_table(self, seat):
        seat.decide_frame(self.short_blind_turn(posted=50))
        # 400 chips at the real 200 level is 2 bb. Read off the post it is 8 bb,
        # which is deeper than the whole table and cannot be true.
        assert seat.tally.depth_by_hand[7] == pytest.approx(2.0)

    def test_a_full_post_is_unaffected(self, seat):
        seat.decide_frame(self.short_blind_turn(posted=BB))
        assert seat.tally.depth_by_hand[7] == pytest.approx(2.0)

    def test_the_level_only_ever_rises(self, seat):
        seat.decide_frame(self.short_blind_turn(posted=400))
        assert seat.blind_level == 400
        seat.decide_frame(self.short_blind_turn(posted=50))
        assert seat.blind_level == 400, "a short post is not a level"


class TestTheDepthCensus:
    """Decisions per depth band -- the number that sizes a ladder.

    `depth_by_hand` counts a 40-decision hand and a 1-decision hand alike, so it
    cannot say what share of PLAY happens off the trained depth.
    """

    def shallow_turn(self):
        return turn_payload(hand_number=3, your_stack=200, opponent_stacks=[300])

    def test_every_decision_lands_in_a_band(self, seat):
        for _ in range(3):
            seat.decide_frame(turn_payload())
        assert sum(seat.tally.by_band.values()) == 3

    def test_a_shallow_decision_is_counted_out_of_tree(self, seat):
        seat.decide_frame(self.shallow_turn())
        assert seat.tally.out_of_tree == 1

    def test_a_decision_at_the_trained_depth_is_not(self, seat):
        seat.decide_frame(turn_payload())
        assert seat.tally.out_of_tree == 0
        assert "below the trained depth" not in seat.tally.summary()

    def test_the_summary_reports_the_share_and_the_census(self, seat):
        seat.decide_frame(turn_payload())
        seat.decide_frame(self.shallow_turn())
        summary = seat.tally.summary()
        assert "1/2 decisions below the trained depth" in summary
        assert "<=10bb:2" in summary


class TestTheLadderInPlay:
    """A seat handed several rungs plays the one that fits the hand.

    The table is 4 bb, so a hand whose effective stack halves is a 2 bb hand and
    must be answered by the 2 bb rung. Selection is per HAND, and the effective
    stack is fixed once a hand starts, so it cannot move under a hand in play.
    """

    @pytest.fixture
    def laddered(self, blueprint):
        from src.interfaces.chipzen.ladder import DepthLadder

        # The BLUEPRINT's blinds are 50/100 while the TABLE's are 100/200, so a
        # 200-chip blueprint is the 2 bb rung against the table's 4 bb.
        shallow = build_trained_test_solver(iterations=4, starting_stack=200)
        return BlueprintSeat.for_match(
            DepthLadder([blueprint, shallow]), MATCH_INFO, seat=0, use_resolver=False
        )

    def shortstacked(self, mine, theirs, hand=9):
        return turn_payload(
            hand_number=hand,
            your_stack=mine - SB,
            opponent_stacks=[theirs - BB],
            pot=SB + BB,
        )

    def test_a_full_stack_opens_on_the_deepest_rung(self, laddered):
        assert laddered.scale.our_depth == 4.0
        laddered.decide_frame(turn_payload())
        assert laddered.tally.rung_switches == 0

    def test_a_halved_stack_drops_to_the_shallow_rung(self, laddered):
        laddered.decide_frame(self.shortstacked(mine=STACK // 2, theirs=STACK // 2))
        assert laddered.scale.our_depth == 2.0
        assert laddered.tally.rung_switches == 1

    def test_it_switches_back_when_the_stacks_come_back(self, laddered):
        laddered.decide_frame(self.shortstacked(mine=STACK // 2, theirs=STACK // 2, hand=9))
        laddered.decide_frame(turn_payload(hand_number=10))
        assert laddered.scale.our_depth == 4.0
        assert laddered.tally.rung_switches == 2

    def test_the_rung_is_chosen_once_within_one_hand(self, laddered):
        # Three turns of ONE hand at one depth: the rung is chosen once and the
        # spot cannot move under the hand in play.
        for _ in range(3):
            laddered.decide_frame(self.shortstacked(mine=STACK // 2, theirs=STACK // 2))
        assert laddered.tally.rung_switches == 1

    def test_the_summary_reports_the_switches(self, laddered):
        laddered.decide_frame(self.shortstacked(mine=STACK // 2, theirs=STACK // 2))
        assert "1 rung switches" in laddered.tally.summary()

    def test_a_laddered_seat_still_answers_legally(self, laddered):
        frame = laddered.decide_frame(self.shortstacked(mine=STACK // 2, theirs=STACK // 2))
        assert frame["action"] in LEGAL


class TestTheBudgetCeiling:
    """No decision may hold the GIL long enough to time out a concurrent one.

    Two matches can be in flight, and although the SDK decides each in its own
    thread, none of the numba kernels under a decision sets `nogil=True` -- so
    concurrent decisions serialise. A 30 s-clock match at half its clock would
    hold the interpreter for 14.2 s and forfeit any 2 s fixture decision waiting
    behind it.
    """

    def test_a_generous_clock_is_capped(self):
        from src.interfaces.chipzen.seat import MAX_BUDGET_MS

        assert budget_for(30_000) == MAX_BUDGET_MS

    def test_a_concurrent_pair_still_fits_the_tight_clock(self):
        from src.interfaces.chipzen.seat import MAX_BUDGET_MS

        # Measured overshoot is ~+35 ms and a frame is ~120 ms, so the pair is
        # the capped decision plus the tight one plus two frames.
        pair = MAX_BUDGET_MS + 35 + budget_for(TIGHT_CLOCK_MS) + 35 + 2 * 120
        assert pair < TIGHT_CLOCK_MS

    def test_the_tight_clock_is_untouched_by_the_cap(self):
        # The cap must only bind on the roomy clock; the fast path was already
        # well under it.
        assert budget_for(TIGHT_CLOCK_MS) == max(
            50, int(TIGHT_CLOCK_MS * CLOCK_FRACTION) - OVERSHOOT_ALLOWANCE_MS
        )


class TestResolverOnlyOffTree:
    """The resolver runs only where its edge was measured.

    Its 528 mbb/hand is an OFF-TREE gain. On tree it measured -203.6 +/- 133.1
    -- nothing -- while DOUBLING the stack-off rate, because its local tree ends
    at the next street and CFR then converges to shipping whenever equity is
    ahead (DEC-0013). This gate keeps that cost off the on-tree spots.
    """

    @staticmethod
    def _spy(monkeypatch) -> list:
        """Record the `use_resolver` each decision asks for."""
        import src.engine.search.agent as agent_module
        from src.core.game.actions import Action, ActionType

        seen: list = []

        class _Spy:
            def __init__(self, _blueprint, *, use_resolver=None, rng=None):
                seen.append(use_resolver)

            def act(self, _state, time_budget_ms=None):
                return Action(type=ActionType.FOLD)

        monkeypatch.setattr(agent_module, "BlueprintAgent", _Spy)
        return seen

    @staticmethod
    def _spot(seat, off_tree: int):
        from src.core.game.state import FULL_DECK
        from src.interfaces.chipzen.adapter import Spot

        state = seat.blueprint.rules.create_initial_state(
            starting_stack=seat.blueprint.config.game.starting_stack,
            hole_cards=((FULL_DECK[0], FULL_DECK[1]), (FULL_DECK[2], FULL_DECK[3])),
            button=0,
        )
        return Spot(state=state, scale=seat.scale, seat=0, off_tree=off_tree, truncated=False)

    def test_an_off_tree_spot_gets_the_resolver(self, blueprint, monkeypatch) -> None:
        seat = BlueprintSeat.for_match(blueprint, MATCH_INFO, seat=0, resolver_only_off_tree=True)
        seen = self._spy(monkeypatch)
        seat._choose(self._spot(seat, off_tree=2))
        assert seen == [True]

    def test_an_on_tree_spot_does_not(self, blueprint, monkeypatch) -> None:
        seat = BlueprintSeat.for_match(blueprint, MATCH_INFO, seat=0, resolver_only_off_tree=True)
        seen = self._spy(monkeypatch)
        seat._choose(self._spot(seat, off_tree=0))
        assert seen == [False]

    def test_no_resolver_still_wins(self, blueprint, monkeypatch) -> None:
        # Two switches that can disagree is the bug this avoids: an explicit
        # `--no-resolver` must not be re-enabled by an off-tree spot.
        seat = BlueprintSeat.for_match(
            blueprint, MATCH_INFO, seat=0, use_resolver=False, resolver_only_off_tree=True
        )
        seen = self._spy(monkeypatch)
        seat._choose(self._spot(seat, off_tree=5))
        assert seen == [False]

    def test_it_is_off_by_default(self, blueprint, monkeypatch) -> None:
        # Unset must still defer to `resolver.enabled`, not to this gate.
        seat = BlueprintSeat.for_match(blueprint, MATCH_INFO, seat=0)
        seen = self._spy(monkeypatch)
        seat._choose(self._spot(seat, off_tree=0))
        assert seen == [None]
