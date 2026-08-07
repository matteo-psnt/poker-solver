"""Their frame, our infoset -- and the two chip currencies that must not mix.

The API key is granted by hand and every endpoint that plays is gated, so this
file is the whole safety net for the half of the client that can be wrong
quietly. Payloads are shaped exactly as their OpenAPI documents them, and the
invariant every sizing test circles is the same one: whatever our tree says, the
amount that goes back on the wire is one the server will accept.
"""

from __future__ import annotations

import pytest

from src.core.game.actions import Action, ActionType
from src.interfaces.gtowizard.adapter import (
    AdapterError,
    opponent_hole_cards,
    reconstruct,
    round_wagers,
    table_scale,
    wire_amount,
)
from src.interfaces.gtowizard.protocol import Frame
from tests.test_helpers import build_trained_test_solver

# The test blueprint plays 50/100 with 400 behind -- 4 big blinds. A table at
# double the denomination is the same game in bigger chips.
FACTOR = 2
SB, BB, STACK = 100, 200, 800
HERO = 1
VILLAIN = 0

GAME = {
    "game_id": 1,
    "game_name": "test",
    "game_format": "heads-up",
    "starting_stack": STACK,
    "blinds": [SB, BB],
    "stack_reset_per_hand": True,
}


def frame(
    street: str = "preflop",
    history: list[str] | None = None,
    board: str = "",
    *,
    raise_min: int = 2 * BB,
    raise_max: int = STACK,
    legal: list[str] | None = None,
    game: dict | None = None,
) -> Frame:
    return Frame.parse(
        {
            "hand_id": 7,
            "game": game or GAME,
            "game_state": {
                "street": street,
                "common_pot": 0,
                "total_pot": SB + BB,
                "board_cards": board,
                "is_hand_over": False,
                "players": [
                    {"name": "villain", "stack": 1, "position": "?", "hole_cards": None},
                    {"name": "hero", "stack": 1, "position": "?", "hole_cards": "AhKd"},
                ],
                "legal_actions": legal if legal is not None else ["f", "c", "b"],
                "raise_range": {"min": raise_min, "max": raise_max},
                "action_history": history or [],
                "has_gto_wizard_folded": False,
                "winnings": None,
                "aivat_score": None,
            },
        }
    )


@pytest.fixture(scope="module")
def blueprint():
    """A real static blueprint, barely trained -- the strategy is irrelevant here."""
    return build_trained_test_solver(iterations=4)


@pytest.fixture(scope="module")
def scale(blueprint):
    return table_scale(frame().game, blueprint)


class TestTableScale:
    def test_a_whole_multiple_of_our_blinds_is_the_factor(self, scale) -> None:
        assert scale.factor == FACTOR
        assert scale.depth_matches

    def test_their_chips_convert_both_ways(self, scale) -> None:
        assert scale.to_ours(400) == 200
        assert scale.to_theirs(200) == 400

    def test_a_ragged_blind_ratio_is_refused_not_approximated(self, blueprint) -> None:
        # 150/250 is not a whole multiple of 50/100, so every bet size would sit
        # between two of our tree's rungs.
        odd = {**GAME, "blinds": [150, 250]}
        with pytest.raises(AdapterError, match="whole"):
            table_scale(frame(game=odd).game, blueprint)

    def test_blinds_that_scale_unevenly_are_refused(self, blueprint) -> None:
        uneven = {**GAME, "blinds": [50, 200]}
        with pytest.raises(AdapterError, match="unevenly"):
            table_scale(frame(game=uneven).game, blueprint)

    def test_a_depth_we_did_not_train_is_visible_rather_than_silent(self, blueprint) -> None:
        # THE guard that fires against the real benchmark: their game is 200 bb
        # and every blueprint we have is cut for 100.
        deep = {**GAME, "starting_stack": STACK * 2}
        assert not table_scale(frame(game=deep).game, blueprint).depth_matches


class TestRoundWagers:
    def test_the_blinds_open_the_preflop_round(self) -> None:
        turn = frame().turn
        button = turn.button_seat()
        wagers = round_wagers(turn, frame().game)
        assert wagers[button] == SB
        assert wagers[1 - button] == BB

    def test_a_raise_to_is_read_as_a_total_not_an_increment(self) -> None:
        # Villain opened to 600. That is their whole round wager, not 600 more.
        turn = frame(history=["b600"]).turn
        assert max(round_wagers(turn, frame().game)) == 600

    def test_a_call_matches_the_high_wager(self) -> None:
        turn = frame(history=["b600", "c", "_"], street="flop", board="2s7c9d").turn
        # A new street opens at nothing, whatever went in before it.
        assert round_wagers(turn, frame().game) == [0, 0]

    def test_a_later_street_opens_at_nothing(self) -> None:
        turn = frame(history=["c", "k", "_", "b300"], street="flop", board="2s7c9d").turn
        assert max(round_wagers(turn, frame().game)) == 300


class TestReconstruct:
    def test_an_unacted_hand_lands_on_our_seat_to_act(self, blueprint, scale) -> None:
        spot = reconstruct(blueprint, frame(), scale)
        assert not spot.truncated
        assert spot.state.current_player == spot.seat == HERO

    def test_a_villain_open_lands_on_us_with_it_replayed(self, blueprint, scale) -> None:
        spot = reconstruct(blueprint, frame(history=["b600"]), scale)
        assert not spot.truncated
        assert spot.state.current_player == HERO
        # Their 600 is our 300, and we are facing it.
        assert spot.state.to_call > 0

    def test_the_opponents_stand_in_cards_never_collide(self) -> None:
        turn = frame(street="flop", board="2s7c9d").turn
        hero = turn.players[HERO].hole_cards
        theirs = opponent_hole_cards((*turn.board, *hero))
        masks = {card.mask for card in (*turn.board, *hero, *theirs)}
        assert len(masks) == 3 + len(hero) + len(theirs)


class TestWireAmount:
    def test_a_raise_goes_back_as_a_cumulative_total(self, blueprint, scale) -> None:
        # Villain opened to 600 (our 300). We raise by 50 of ours = 100 theirs,
        # so the wire carries 600 + 100 = 700, not the bare 100.
        this = frame(history=["b600"], raise_max=STACK)
        spot = reconstruct(blueprint, this, scale)
        amount = wire_amount(Action(ActionType.RAISE, 50), this.turn, this.game, spot)
        assert amount == 700

    def test_an_all_in_is_their_maximum(self, blueprint, scale) -> None:
        this = frame(history=["b600"], raise_max=STACK)
        spot = reconstruct(blueprint, this, scale)
        assert wire_amount(Action(ActionType.ALL_IN, 400), this.turn, this.game, spot) == STACK

    def test_a_size_outside_their_range_is_clamped_not_rejected(self, blueprint, scale) -> None:
        this = frame(history=["b600"], raise_min=1200, raise_max=1500)
        spot = reconstruct(blueprint, this, scale)
        for ours in (1, 10_000):
            amount = wire_amount(Action(ActionType.RAISE, ours), this.turn, this.game, spot)
            assert 1200 <= amount <= 1500

    def test_betting_where_it_is_not_offered_is_an_error_not_a_zero(self, blueprint, scale) -> None:
        this = frame(history=["b600"], raise_max=0, legal=["f", "c"])
        spot = reconstruct(blueprint, this, scale)
        with pytest.raises(AdapterError, match="not offered"):
            wire_amount(Action(ActionType.RAISE, 200), this.turn, this.game, spot)
