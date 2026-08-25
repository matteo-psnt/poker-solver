"""Their turn, our infoset -- and the two chip currencies that must not mix.

We cannot hold a Chipzen account and a live match open in CI, so this file is the
whole safety net for the half of the client that can be wrong quietly. The
payloads are shaped exactly as the published protocol documents them, and the
invariant every sizing test circles is the same one: whatever our tree says, the
amount that goes back on the wire is one the server will accept.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.core.game.actions import Action, ActionType
from src.core.game.state import Street
from src.interfaces.chipzen.adapter import (
    AdapterError,
    opponent_hole_cards,
    reconstruct,
    table_scale,
    wire_action,
)
from src.interfaces.chipzen.protocol import GameConfig, ProtocolError, TurnState
from tests.test_helpers import build_test_solver, build_trained_test_solver, make_test_config

# The test blueprint plays 50/100 with 400 behind -- 4 big blinds. A Chipzen
# table at double the denomination is the same game in bigger chips.
FACTOR = 2
SB, BB, STACK = 100, 200, 800

SMALL_BLIND_ENTRY = {
    "seat": 0,
    "action": "post_small_blind",
    "amount": SB,
    "phase": "preflop",
    "is_timeout": False,
}
BIG_BLIND_ENTRY = {
    "seat": 1,
    "action": "post_big_blind",
    "amount": BB,
    "phase": "preflop",
    "is_timeout": False,
}


@pytest.fixture(scope="module")
def blueprint():
    """A real static blueprint, barely trained -- the strategy is irrelevant here."""
    return build_trained_test_solver(iterations=4)


@pytest.fixture(scope="module")
def config():
    return GameConfig.parse(
        {
            "variant": "nlhe",
            "starting_stack": STACK,
            "small_blind": SB,
            "big_blind": BB,
            "num_players": 2,
        }
    )


@pytest.fixture(scope="module")
def scale(config, blueprint):
    return table_scale(config, blueprint)


def turn_payload(**overrides):
    """A ``turn_request.state`` for the small blind, first to act preflop."""
    payload = {
        "hand_number": 1,
        "phase": "preflop",
        "board": [],
        "your_hole_cards": ["Ah", "Kd"],
        "pot": SB + BB,
        "your_stack": STACK - SB,
        "opponent_stacks": [STACK - BB],
        "to_call": BB - SB,
        "min_raise": 2 * BB,
        "max_raise": STACK,
        "action_history": [SMALL_BLIND_ENTRY, BIG_BLIND_ENTRY],
    }
    payload.update(overrides)
    return payload


def raise_entry(seat: int, amount: int, phase: str = "preflop"):
    """A player raise. ``amount`` is their total wager for the round."""
    return {
        "seat": seat,
        "action": "raise",
        "amount": amount,
        "phase": phase,
        "is_timeout": False,
    }


def simple_entry(seat: int, action: str, amount: int = 0, phase: str = "preflop"):
    return {"seat": seat, "action": action, "amount": amount, "phase": phase, "is_timeout": False}


class TestParsing:
    def test_a_turn_carries_its_cards_and_amounts(self):
        turn = TurnState.parse(turn_payload())
        assert [str(card) for card in turn.hole_cards] == ["[ A ♥ ]", "[ K ♦ ]"]
        assert turn.to_call == BB - SB
        assert turn.action_history[0].is_synthetic

    def test_the_button_is_whoever_posted_the_small_blind(self):
        assert TurnState.parse(turn_payload()).button_seat() == 0
        swapped = turn_payload(
            action_history=[
                {**SMALL_BLIND_ENTRY, "seat": 1},
                {**BIG_BLIND_ENTRY, "seat": 0},
            ]
        )
        assert TurnState.parse(swapped).button_seat() == 1

    def test_a_history_with_no_blind_refuses_to_guess(self):
        with pytest.raises(ProtocolError, match="cannot locate the button"):
            TurnState.parse(turn_payload(action_history=[])).button_seat()

    def test_round_wager_is_the_latest_total_for_the_phase(self):
        turn = TurnState.parse(
            turn_payload(
                action_history=[SMALL_BLIND_ENTRY, BIG_BLIND_ENTRY, raise_entry(0, 600)],
            )
        )
        assert turn.round_wager(0) == 600
        assert turn.round_wager(1) == BB

    def test_a_wager_from_an_earlier_street_does_not_leak(self):
        turn = TurnState.parse(
            turn_payload(
                phase="flop",
                board=["2c", "7d", "9h"],
                action_history=[
                    SMALL_BLIND_ENTRY,
                    BIG_BLIND_ENTRY,
                    simple_entry(0, "call", BB),
                    simple_entry(1, "check", 0),
                ],
            )
        )
        assert turn.round_wager(0) == 0

    @pytest.mark.parametrize(
        ("payload", "because"),
        [
            ({"num_players": 6}, "heads-up"),
            ({"big_blind": 50}, "below small_blind"),
        ],
    )
    def test_a_table_we_cannot_play_is_refused(self, payload, because):
        base = {
            "variant": "nlhe",
            "starting_stack": STACK,
            "small_blind": SB,
            "big_blind": BB,
            "num_players": 2,
        }
        with pytest.raises(ProtocolError, match=because):
            GameConfig.parse({**base, **payload})

    def test_a_table_with_an_ante_is_refused(self):
        """`GameRules` takes two blinds and nothing else; an ante is a game we
        cannot represent, so every pot would be the wrong size silently."""
        with pytest.raises(ProtocolError, match="ante"):
            GameConfig.parse(
                {
                    "variant": "nlhe",
                    "starting_stack": STACK,
                    "small_blind": SB,
                    "big_blind": BB,
                    "ante": 25,
                    "num_players": 2,
                }
            )

    @pytest.mark.parametrize(
        ("broken", "because"),
        [
            ({"board": ["2c", "2c", "9h"], "phase": "flop"}, "a board card twice"),
            ({"your_hole_cards": ["Ah", "Ah"]}, "the same card twice in hand"),
            (
                {"board": ["Ah", "7d", "9h"], "phase": "flop"},
                "a hole card already on the board",
            ),
        ],
    )
    def test_a_card_that_repeats_is_refused(self, broken, because):
        """Probed live: each of these came back with a confident action.

        A real server never sends one, which is exactly why nothing downstream
        checks -- the bucket for a state that cannot exist looks like any other.
        """
        with pytest.raises(ProtocolError, match="repeats"):
            TurnState.parse(turn_payload(**broken))

    def test_a_bad_card_names_itself(self):
        with pytest.raises(ProtocolError, match="'Zz' is not a card"):
            TurnState.parse(turn_payload(your_hole_cards=["Zz", "Kd"]))

    def test_a_missing_required_field_names_itself(self):
        payload = turn_payload()
        del payload["pot"]
        with pytest.raises(ProtocolError, match="'pot'"):
            TurnState.parse(payload)


class TestScale:
    def test_a_whole_multiple_of_our_blinds_is_the_factor(self, scale):
        assert scale.factor == FACTOR
        assert scale.depth_matches

    def test_conversion_round_trips_our_chips(self, scale):
        assert scale.to_theirs(200) == 400
        assert scale.to_ours(400) == 200

    def test_their_chips_round_to_nearest(self, scale):
        assert scale.to_ours(0) == 0
        assert scale.to_ours(FACTOR) == 1
        # Half a chip of ours rounds up, not toward zero -- so a size that sits
        # between two of our rungs never silently becomes a call.
        assert scale.to_ours(FACTOR // 2) == 1
        assert scale.to_ours(FACTOR // 2 - 1) == 0

    def test_blinds_that_are_not_a_multiple_are_refused(self, blueprint):
        odd = GameConfig.parse(
            {
                "variant": "nlhe",
                "starting_stack": 1000,
                "small_blind": 75,
                "big_blind": 150,
                "num_players": 2,
            }
        )
        with pytest.raises(AdapterError, match="not a whole multiple"):
            table_scale(odd, blueprint)

    def test_a_depth_we_did_not_train_is_flagged_not_refused(self, blueprint):
        """A shallower table still plays; the caller decides what to do about it."""
        shallow = GameConfig.parse(
            {
                "variant": "nlhe",
                "starting_stack": STACK // 2,
                "small_blind": SB,
                "big_blind": BB,
                "num_players": 2,
            }
        )
        assert not table_scale(shallow, blueprint).depth_matches


class TestReconstruct:
    def test_the_opening_spot_is_the_small_blind_to_act(self, blueprint, scale):
        spot = reconstruct(blueprint, TurnState.parse(turn_payload()), seat=0, scale=scale)
        assert spot.state.street is Street.PREFLOP
        assert spot.state.current_player == 0
        assert not spot.truncated
        assert spot.off_tree == 0

    def test_our_hole_cards_land_in_our_seat(self, blueprint, scale):
        spot = reconstruct(blueprint, TurnState.parse(turn_payload()), seat=0, scale=scale)
        assert [str(card) for card in spot.state.hole_cards[0]] == ["[ A ♥ ]", "[ K ♦ ]"]

    def test_the_big_blind_facing_a_raise_is_reconstructed_at_its_own_seat(self, blueprint, scale):
        payload = turn_payload(
            your_hole_cards=["Qs", "Qc"],
            your_stack=STACK - BB,
            opponent_stacks=[STACK - 600],
            to_call=400,
            action_history=[SMALL_BLIND_ENTRY, BIG_BLIND_ENTRY, raise_entry(0, 600)],
        )
        spot = reconstruct(blueprint, TurnState.parse(payload), seat=1, scale=scale)
        assert spot.state.current_player == 1
        assert not spot.truncated
        assert [str(card) for card in spot.state.hole_cards[1]] == ["[ Q ♠ ]", "[ Q ♣ ]"]

    def test_a_flop_spot_carries_the_board(self, blueprint, scale):
        payload = turn_payload(
            phase="flop",
            board=["2c", "7d", "9h"],
            pot=2 * BB,
            to_call=0,
            min_raise=BB,
            max_raise=STACK - BB,
            action_history=[
                SMALL_BLIND_ENTRY,
                BIG_BLIND_ENTRY,
                simple_entry(0, "call", BB),
                simple_entry(1, "check", 0),
            ],
        )
        spot = reconstruct(blueprint, TurnState.parse(payload), seat=1, scale=scale)
        assert spot.state.street is Street.FLOP
        assert len(spot.state.board) == 3
        assert not spot.truncated

    def test_the_opponents_hand_never_collides_with_known_cards(self, blueprint, scale):
        payload = turn_payload(phase="flop", board=["2c", "7d", "9h"], to_call=0)
        spot = reconstruct(blueprint, TurnState.parse(payload), seat=0, scale=scale)
        seen = {card.mask for card in spot.state.board}
        seen |= {card.mask for card in spot.state.hole_cards[0]}
        assert not seen & {card.mask for card in spot.state.hole_cards[1]}

    def test_an_on_tree_raise_costs_nothing(self, blueprint, scale):
        """The control for the snap test below: a size we do offer is free."""
        payload = turn_payload(
            your_hole_cards=["Qs", "Qc"],
            to_call=400,
            action_history=[SMALL_BLIND_ENTRY, BIG_BLIND_ENTRY, raise_entry(0, 600)],
        )
        spot = reconstruct(blueprint, TurnState.parse(payload), seat=1, scale=scale)
        assert spot.off_tree == 0

    def test_an_off_tree_raise_is_snapped_and_counted(self, blueprint, scale):
        """A size our action model does not offer still replays, and says so.

        530 of their chips is 265 of ours -- between the rungs the action model
        cuts, so it has to land on one of them and the drift is recorded.
        """
        payload = turn_payload(
            your_hole_cards=["Qs", "Qc"],
            to_call=330,
            action_history=[SMALL_BLIND_ENTRY, BIG_BLIND_ENTRY, raise_entry(0, 530)],
        )
        spot = reconstruct(blueprint, TurnState.parse(payload), seat=1, scale=scale)
        assert spot.off_tree == 1
        assert not spot.truncated
        assert spot.state.current_player == 1

    def test_reconstruction_is_stable_across_repeated_turns(self, blueprint, scale):
        """The same frame must name the same infoset every time it is answered."""
        payload = turn_payload(
            your_hole_cards=["Qs", "Qc"],
            to_call=330,
            action_history=[SMALL_BLIND_ENTRY, BIG_BLIND_ENTRY, raise_entry(0, 530)],
        )
        turn = TurnState.parse(payload)
        first = reconstruct(blueprint, turn, seat=1, scale=scale)
        second = reconstruct(blueprint, turn, seat=1, scale=scale)
        assert first.state.betting_history == second.state.betting_history
        assert first.off_tree == second.off_tree


class TestWireAction:
    @pytest.fixture
    def spot(self, blueprint, scale):
        return reconstruct(blueprint, TurnState.parse(turn_payload()), seat=0, scale=scale)

    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            (Action(ActionType.FOLD), "fold"),
            (Action(ActionType.CHECK), "check"),
            (Action(ActionType.CALL), "call"),
        ],
    )
    def test_unsized_actions_pass_straight_through(self, action, expected, spot):
        turn = TurnState.parse(turn_payload())
        assert wire_action(action, turn, spot) == {"action": expected, "params": {}}

    def test_an_all_in_is_their_max_raise(self, spot):
        turn = TurnState.parse(turn_payload())
        frame = wire_action(Action(ActionType.ALL_IN, 350), turn, spot)
        assert frame == {"action": "raise", "params": {"amount": turn.max_raise}}

    def test_a_raise_is_priced_from_their_numbers(self, spot):
        """Our increment, their chips, on top of what we already have in."""
        turn = TurnState.parse(turn_payload())
        frame = wire_action(Action(ActionType.RAISE, 200), turn, spot)
        assert frame["params"]["amount"] == SB + (BB - SB) + 200 * FACTOR

    def test_every_legal_action_produces_a_size_the_server_accepts(self, blueprint, spot):
        """The invariant that matters: nothing we send can be out of bounds."""
        turn = TurnState.parse(turn_payload())
        legal = blueprint.rules.get_legal_actions(spot.state, action_model=blueprint.action_model)
        assert legal
        for action in legal:
            frame = wire_action(action, turn, spot)
            if frame["action"] == "raise":
                assert turn.min_raise <= frame["params"]["amount"] <= turn.max_raise

    def test_an_oversized_raise_is_clamped_rather_than_rejected(self, spot):
        turn = TurnState.parse(turn_payload())
        frame = wire_action(Action(ActionType.RAISE, 10_000), turn, spot)
        assert frame["params"]["amount"] == turn.max_raise

    def test_an_undersized_raise_is_lifted_to_the_minimum(self, spot):
        turn = TurnState.parse(turn_payload())
        frame = wire_action(Action(ActionType.BET, 1), turn, spot)
        assert frame["params"]["amount"] == turn.min_raise

    def test_raising_where_it_is_not_offered_refuses(self, spot):
        turn = TurnState.parse(turn_payload(min_raise=0, max_raise=0))
        with pytest.raises(AdapterError, match="raising is not offered"):
            wire_action(Action(ActionType.RAISE, 200), turn, spot)


class TestValidActions:
    """Their `valid_actions` is authoritative and is NOT derivable from the sizes.

    Measured on the first live hand against PluriBot: facing an all-in the frame
    still carried a positive `max_raise` while offering only fold and call, and
    four raises were rejected -- after which the SERVER picked a safe default and
    the blueprint's choice was discarded.
    """

    @pytest.fixture
    def spot(self, blueprint, scale):
        return reconstruct(blueprint, TurnState.parse(turn_payload()), seat=0, scale=scale)

    @pytest.fixture
    def facing_an_all_in(self):
        """Positive max_raise, but only fold and call on offer."""
        return TurnState.parse(
            turn_payload(to_call=600, max_raise=STACK, valid_actions=["fold", "call"])
        )

    def test_a_raise_becomes_a_call_when_raising_is_not_offered(self, facing_an_all_in, spot):
        frame = wire_action(Action(ActionType.RAISE, 200), facing_an_all_in, spot)
        assert frame == {"action": "call", "params": {}}

    def test_an_all_in_becomes_a_call_too(self, facing_an_all_in, spot):
        frame = wire_action(Action(ActionType.ALL_IN, 350), facing_an_all_in, spot)
        assert frame == {"action": "call", "params": {}}

    def test_a_raise_becomes_a_check_when_nothing_is_owed(self, spot):
        turn = TurnState.parse(turn_payload(to_call=0, valid_actions=["check"]))
        assert wire_action(Action(ActionType.BET, 50), turn, spot)["action"] == "check"

    def test_a_frame_without_the_field_is_not_filtered(self, spot):
        """A recorded fixture predating the field still gets a real answer."""
        turn = TurnState.parse(turn_payload())
        assert turn.valid_actions == ()
        assert wire_action(Action(ActionType.RAISE, 200), turn, spot)["action"] == "raise"

    def test_an_offered_raise_still_goes_out_as_one(self, spot):
        turn = TurnState.parse(turn_payload(valid_actions=["fold", "call", "raise"]))
        assert wire_action(Action(ActionType.RAISE, 200), turn, spot)["action"] == "raise"


class TestPlaceholders:
    def test_two_cards_outside_the_dead_set(self):
        turn = TurnState.parse(turn_payload(board=["2c", "7d", "9h"]))
        dead = (*turn.board, *turn.hole_cards)
        drawn = opponent_hole_cards(dead)
        assert len({card.mask for card in drawn}) == 2
        assert not {card.mask for card in drawn} & {card.mask for card in dead}


class TestTheRealTable:
    """The denomination and depth this will actually meet, not a scaled-down one.

    Everything above runs at 4 bb and ``factor`` 2, where ``to_ours`` is exact
    and the tree is too shallow to reach a raise cap. Chipzen's own protocol
    example is 5/10 with 1000 -- ``factor`` 5, where rounding is lossy, and 100
    bb, which is the depth the production blueprint is cut for. This exercises
    both at once, across a street boundary, from the shipped fixture.
    """

    @pytest.fixture(scope="class")
    def hundred_bb(self):
        """1/2 blinds, 200 behind -- 100 bb, same as their table in our chips."""
        config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=200)
        solver, _storage = build_test_solver(config)
        return solver

    @pytest.fixture(scope="class")
    def recorded(self):
        path = Path(__file__).parent / "fixtures" / "turn_preflop_sb.json"
        return json.loads(path.read_text())

    @pytest.fixture(scope="class")
    def real(self, hundred_bb, recorded):
        config = GameConfig.parse(recorded["game_config"])
        scale = table_scale(config, hundred_bb)
        turn = TurnState.parse(recorded["state"])
        spot = reconstruct(hundred_bb, turn, seat=recorded["seat"], scale=scale)
        return turn, spot

    def test_their_denomination_is_a_clean_factor_of_ours(self, real):
        _turn, spot = real
        assert spot.scale.factor == 5
        assert spot.scale.depth_matches
        assert spot.scale.their_depth == 100.0

    def test_rounding_is_lossy_at_this_factor(self, real):
        """The case factor 2 cannot produce: a size strictly between two rungs."""
        _turn, spot = real
        assert spot.scale.to_ours(15) == 3
        assert spot.scale.to_ours(12) == 2
        assert spot.scale.to_ours(13) == 3

    def test_a_preflop_raise_and_call_replays_onto_the_flop(self, real):
        _turn, spot = real
        assert spot.state.street is Street.FLOP
        assert len(spot.state.board) == 3
        assert spot.state.current_player == 1
        assert spot.off_tree == 0
        assert not spot.truncated

    def test_every_legal_action_is_priced_inside_their_bounds(self, hundred_bb, real):
        turn, spot = real
        legal = hundred_bb.rules.get_legal_actions(spot.state, action_model=hundred_bb.action_model)
        assert legal
        for action in legal:
            frame = wire_action(action, turn, spot)
            if frame["action"] == "raise":
                assert turn.min_raise <= frame["params"]["amount"] <= turn.max_raise

    def test_an_all_in_is_every_chip_the_server_will_take(self, real):
        turn, spot = real
        frame = wire_action(Action(ActionType.ALL_IN, 194), turn, spot)
        assert frame["params"]["amount"] == turn.max_raise
