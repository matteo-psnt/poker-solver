"""A seat must answer every turn, including the ones it cannot understand.

The server scores a decision that does not arrive as a timeout, and a timeout is
a fold -- so the property under test throughout is that ``decide_frame`` returns
a legal frame for any input at all, and records what it could not do rather than
raising. Anything else loses a match to an exception.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from src.interfaces.chipzen.seat import BlueprintSeat, sdk_state_payload
from tests.interfaces.chipzen.test_adapter import (
    BIG_BLIND_ENTRY,
    SMALL_BLIND_ENTRY,
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
    return BlueprintSeat.for_match(blueprint, MATCH_INFO, seat=0)


class TestSeatConstruction:
    def test_a_matching_table_scales_cleanly(self, seat):
        assert seat.scale.factor == 2
        assert seat.scale.depth_matches

    def test_a_shallower_table_still_seats(self, blueprint, caplog):
        shallow = {"game_config": {**MATCH_INFO["game_config"], "starting_stack": 400}}
        built = BlueprintSeat.for_match(blueprint, shallow, seat=0)
        assert not built.scale.depth_matches
        assert "trained at" in caplog.text

    def test_the_resolver_is_off_unless_asked_for(self, seat):
        """Off-tree play is where the resolver has been measured to collapse."""
        assert seat.use_resolver is False


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
        built = BlueprintSeat.for_match(blueprint, MATCH_INFO, seat=1)
        built.decide_frame(
            turn_payload(
                your_hole_cards=["Qs", "Qc"],
                to_call=330,
                action_history=[SMALL_BLIND_ENTRY, BIG_BLIND_ENTRY, raise_entry(0, 530)],
            )
        )
        assert built.tally.off_tree == 1


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
        built = BlueprintSeat.for_match(blueprint, MATCH_INFO, seat=1)
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
