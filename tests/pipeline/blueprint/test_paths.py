"""Paths name a spot, and replaying one is the only way to reach it.

The property that matters most here is the refusal. A path that no longer
describes a real line must raise, not resolve to something nearby -- the whole
reason to address nodes this way is that a stale path fails loudly where a stale
node id would quietly point at a different spot.
"""

from __future__ import annotations

import pytest

from src.core.game.actions import Action, ActionType
from src.core.game.state import Card
from src.pipeline.blueprint.paths import (
    PathError,
    encode_action,
    encode_path,
    parse_path,
    placeholder_hole_cards,
    replay,
)
from tests.test_helpers import build_trained_test_solver

FLOP = (Card.new("2c"), Card.new("7d"), Card.new("9h"))
TURN = (*FLOP, Card.new("4s"))


@pytest.fixture(scope="module")
def blueprint():
    """A real static blueprint, barely trained -- the strategy is irrelevant here."""
    return build_trained_test_solver(iterations=4)


class TestTokens:
    def test_sized_actions_carry_their_amount(self):
        assert encode_action(Action(ActionType.RAISE, 150)) == "r150"
        assert encode_action(Action(ActionType.BET, 50)) == "b50"

    def test_unsized_actions_are_a_bare_letter(self):
        assert encode_action(Action(ActionType.FOLD, 0)) == "f"
        assert encode_action(Action(ActionType.CHECK, 0)) == "x"
        assert encode_action(Action(ActionType.ALL_IN, 400)) == "A"

    def test_a_path_round_trips(self):
        actions = (Action(ActionType.RAISE, 150), Action(ActionType.CALL, 0))
        assert parse_path(encode_path(actions)) == ("r150", "c")

    def test_an_empty_path_is_the_start_of_the_hand(self):
        assert parse_path("") == ()
        assert parse_path("/") == ()

    @pytest.mark.parametrize(
        ("bad", "because"),
        [
            ("z", "does not start with an action"),
            ("b", "needs an amount"),
            ("rXY", "needs an amount"),
            ("c50", "takes no amount"),
        ],
    )
    def test_malformed_tokens_are_refused_before_any_state_is_built(self, bad, because):
        with pytest.raises(PathError, match=because):
            parse_path(bad)


class TestReplay:
    def test_the_empty_path_is_the_first_decision(self, blueprint):
        node = replay(blueprint, "")

        assert node.actor is not None
        assert node.legal_actions, "someone must be able to act at the root"
        assert node.board_consumed == 0

    def test_every_offered_action_can_be_taken(self, blueprint):
        """Whatever replay reports as legal must itself replay -- or the surface
        would advertise lines a caller cannot then follow."""
        root = replay(blueprint, "")

        for action in root.legal_actions:
            child = replay(blueprint, encode_action(action), board=TURN)
            assert child.state is not None

    def test_a_line_that_reaches_the_flop_consumes_three_cards(self, blueprint):
        root = replay(blueprint, "")
        call = next(a for a in root.legal_actions if a.type is ActionType.CALL)
        node = replay(blueprint, encode_path((call, Action(ActionType.CHECK, 0))), board=FLOP)

        assert node.board_consumed == 3
        assert node.state.board == FLOP

    def test_the_given_board_is_used_rather_than_dealt(self, blueprint):
        """Two replays of one path must agree, which random dealing would break."""
        root = replay(blueprint, "")
        call = next(a for a in root.legal_actions if a.type is ActionType.CALL)
        path = encode_path((call, Action(ActionType.CHECK, 0)))

        assert (
            replay(blueprint, path, FLOP).state.board == replay(blueprint, path, FLOP).state.board
        )

    def test_surplus_board_cards_are_ignored(self, blueprint):
        """So a caller can pin one runout and walk back and forth along a line."""
        node = replay(blueprint, "", board=TURN)

        assert node.board_consumed == 0


class TestRefusals:
    def test_an_action_that_is_not_on_offer_names_what_is(self, blueprint):
        with pytest.raises(PathError, match="On offer"):
            replay(blueprint, "b999999")

    def test_a_line_past_the_end_of_the_hand_is_refused(self, blueprint):
        with pytest.raises(PathError, match="already ended"):
            replay(blueprint, "f/f/f")

    def test_too_few_board_cards_says_how_many_were_needed(self, blueprint):
        root = replay(blueprint, "")
        call = next(a for a in root.legal_actions if a.type is ActionType.CALL)
        path = encode_path((call, Action(ActionType.CHECK, 0)))

        with pytest.raises(PathError, match="needs 3 board cards, but 0 were given"):
            replay(blueprint, path, board=())


class TestPlaceholderHoleCards:
    def test_they_never_collide_with_the_board(self):
        holes = placeholder_hole_cards(TURN)
        dealt = {card.mask for hand in holes for card in hand}

        assert len(dealt) == 4
        assert not dealt.intersection({card.mask for card in TURN})

    def test_a_full_board_still_leaves_room(self):
        river = (*TURN, Card.new("Ts"))

        assert len(placeholder_hole_cards(river)) == 2
