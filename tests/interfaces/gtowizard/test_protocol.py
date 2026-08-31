"""The wire has no seats in it, so this is where they are pinned.

Their ``action_history`` is a flat list of ``f``/``c``/``k``/``bX`` with ``_``
between streets and nothing saying who acted. Everything downstream -- the
button, the blinds, whose wager a ``bX`` is -- hangs off the alternation rule in
:meth:`Turn.button_seat`, and getting it backwards would field a strategy from
the wrong seat while every number still looked plausible.
"""

from __future__ import annotations

import pytest

from src.interfaces.gtowizard.protocol import Frame, ProtocolError

GAME = {
    "game_id": 1,
    "game_name": "HUNL 200BB",
    "game_format": "heads-up",
    "starting_stack": 20000,
    "blinds": [50, 100],
    "stack_reset_per_hand": True,
}
HERO = 1
VILLAIN = 0


def frame(
    street: str,
    history: list[str],
    board: str = "",
    *,
    legal_actions: list[str] | None = None,
    **overrides: object,
) -> Frame:
    state = {
        "street": street,
        "common_pot": 150,
        "total_pot": 150,
        "board_cards": board,
        "is_hand_over": False,
        "players": [
            {"name": "GTO Wizard AI", "stack": 19900, "position": "BB", "hole_cards": None},
            {"name": "us", "stack": 19950, "position": "BTN", "hole_cards": "AhKd"},
        ],
        "legal_actions": ["f", "c", "b"] if legal_actions is None else legal_actions,
        "raise_range": {"min": 200, "max": 20000},
        "action_history": history,
        "has_gto_wizard_folded": False,
        "winnings": None,
        "aivat_score": None,
    }
    state.update(overrides)
    return Frame.parse({"hand_id": 7, "game": GAME, "game_state": state})


class TestSeats:
    @pytest.mark.parametrize(
        ("street", "history", "button"),
        [
            # Preflop the button acts FIRST, so an even count leaves it to us.
            ("preflop", [], HERO),
            ("preflop", ["b300"], VILLAIN),
            ("preflop", ["c", "b400"], HERO),
            ("preflop", ["b300", "b900"], HERO),
            # Postflop the button acts SECOND, so the parity flips.
            ("flop", ["c", "k", "_"], VILLAIN),
            ("flop", ["c", "k", "_", "b100"], HERO),
            ("turn", ["c", "k", "_", "k", "k", "_"], VILLAIN),
            ("river", ["c", "k", "_", "k", "k", "_", "k", "k", "_"], VILLAIN),
        ],
    )
    def test_the_button_is_read_off_whose_turn_it_is(
        self, street: str, history: list[str], button: int
    ) -> None:
        board = "2s7c9dJhQc"[: 2 * {"preflop": 0, "flop": 3, "turn": 4, "river": 5}[street]]
        assert frame(street, history, board).turn.button_seat() == button

    def test_the_hero_is_the_seat_holding_cards(self) -> None:
        assert frame("preflop", []).turn.hero_seat == HERO


class TestRounds:
    def test_the_history_splits_on_the_street_breaks(self) -> None:
        turn = frame("flop", ["c", "k", "_", "b100"], "2s7c9d").turn
        assert turn.rounds == (("c", "k"), ("b100",))

    def test_a_closed_street_leaves_an_empty_round_to_act_in(self) -> None:
        turn = frame("flop", ["c", "k", "_"], "2s7c9d").turn
        assert turn.rounds == (("c", "k"), ())

    def test_a_board_ahead_of_the_history_is_refused(self) -> None:
        # One break, so the history says flop; the frame says turn. Replaying
        # this would deal a card nobody has bet into.
        with pytest.raises(ProtocolError, match="street break"):
            frame("turn", ["c", "k", "_"], "2s7c9dJh").turn.street_index()


class TestGuards:
    def test_a_card_cannot_appear_twice(self) -> None:
        with pytest.raises(ProtocolError, match="repeats"):
            frame("flop", ["c", "k", "_"], "AhKd7c")

    def test_an_odd_length_card_string_is_not_silently_truncated(self) -> None:
        with pytest.raises(ProtocolError, match="two-character"):
            frame("flop", ["c", "k", "_"], "2s7c9")

    def test_a_table_that_is_not_heads_up_is_refused(self) -> None:
        with pytest.raises(ProtocolError, match="heads-up"):
            frame(
                "preflop",
                [],
                players=[
                    {"name": "a", "stack": 1, "position": "BB", "hole_cards": None},
                    {"name": "b", "stack": 1, "position": "SB", "hole_cards": "AhKd"},
                    {"name": "c", "stack": 1, "position": "BTN", "hole_cards": None},
                ],
            )

    def test_exactly_one_seat_may_hold_cards(self) -> None:
        with pytest.raises(ProtocolError, match="one seat"):
            frame(
                "preflop",
                [],
                players=[
                    {"name": "a", "stack": 1, "position": "BB", "hole_cards": "2s3s"},
                    {"name": "b", "stack": 1, "position": "BTN", "hole_cards": "AhKd"},
                ],
            )

    def test_a_missing_required_field_is_named(self) -> None:
        with pytest.raises(ProtocolError, match="game_state"):
            Frame.parse({"hand_id": 1, "game": GAME})


class TestAllows:
    def test_only_what_the_frame_names_is_allowed(self) -> None:
        turn = frame("preflop", [], legal_actions=["f", "c"]).turn
        assert turn.allows("f")
        assert not turn.allows("b")

    def test_an_empty_list_allows_nothing_at_all(self) -> None:
        # It once allowed everything, on the theory that a fixture predating the
        # field should still be answered. On a LIVE frame that turns "the server
        # named nothing" into "yes", and the resulting 4xx abandons a hand that
        # would otherwise have been scored.
        turn = frame("preflop", [], legal_actions=[]).turn
        assert not any(turn.allows(action) for action in ("f", "c", "k", "b"))


class TestGame:
    def test_the_depth_is_their_stack_in_their_blinds(self) -> None:
        assert frame("preflop", []).game.depth_in_blinds == 200.0
