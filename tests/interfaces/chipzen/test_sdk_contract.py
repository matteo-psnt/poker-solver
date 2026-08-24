"""What we assume about their SDK, checked against the installed one.

Every assertion here was written after the assumption it pins turned out to be
wrong. `chipzen.Action` takes ``amount`` and builds the nested ``params`` itself,
so passing it our wire dict is a ``TypeError``; ``run_external_bot`` has
``max_matches``, not ``loop``; and the environment is ``prod``, never
``production``. All three would have failed on the first decision of the first
real match, where nothing we can run in CI would have seen them.

Skipped when the optional extra is absent, which is the default install -- so
this is a check that runs on the box that will actually hold the seat.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from src.interfaces.commands.chipzen_seat import DEFAULT_ENV, ENVIRONMENTS

chipzen = pytest.importorskip("chipzen", reason="the `chipzen` extra is not installed")


class TestTheNamesWeReachFor:
    @pytest.mark.parametrize("name", ["Action", "ChipzenBot", "run_external_bot"])
    def test_the_module_exports_it(self, name):
        assert hasattr(chipzen, name)


class TestAction:
    def test_it_takes_an_amount_and_not_our_params_dict(self):
        fields = {f.name for f in dataclasses.fields(chipzen.Action) if f.init}
        assert fields == {"action", "amount"}

    def test_a_raise_reaches_the_wire_as_nested_params(self):
        """`to_wire` is what the SDK sends; our amount has to survive into it."""
        assert chipzen.Action(action="raise", amount=60).to_wire() == {
            "action": "raise",
            "params": {"amount": 60},
        }

    def test_an_unsized_action_carries_empty_params(self):
        assert chipzen.Action(action="fold").to_wire() == {"action": "fold", "params": {}}


class TestWhatDecideIsHanded:
    """`sdk_state_payload` reads their `GameState`; these are the reads it makes.

    Both would fail the same silent way: a card that does not render back to
    their notation, or a history entry missing `phase`, is a `ProtocolError` the
    seat catches and answers with a fold -- on every hand, looking exactly like
    the "bot folds every decision" symptom in their own pitfalls guide.
    """

    def test_a_card_renders_back_to_wire_notation(self):
        assert str(chipzen.Card(rank="A", suit="h")) == "Ah"
        assert str(chipzen.Card(rank="T", suit="s")) == "Ts"

    def test_a_turn_request_keeps_its_action_history_keys_unrenamed(self):
        entry = {
            "seat": 0,
            "action": "raise",
            "amount": 30,
            "phase": "preflop",
            "is_timeout": False,
        }
        state = chipzen.GameState.from_turn_request(
            {
                "state": {
                    "hand_number": 1,
                    "phase": "preflop",
                    "board": [],
                    "your_hole_cards": ["Ah", "Kd"],
                    "pot": 15,
                    "your_stack": 995,
                    "opponent_stacks": [990],
                    "to_call": 5,
                    "min_raise": 20,
                    "max_raise": 995,
                    "action_history": [entry],
                }
            }
        )
        assert state.action_history == [entry]

    def test_the_payload_we_build_parses_as_our_own_protocol(self):
        """The round trip that matters: their object -> our dict -> our types."""
        from src.interfaces.chipzen.protocol import TurnState
        from src.interfaces.chipzen.seat import sdk_state_payload

        state = chipzen.GameState.from_turn_request(
            {
                "state": {
                    "hand_number": 2,
                    "phase": "flop",
                    "board": ["Ts", "7h", "2d"],
                    "your_hole_cards": ["Ah", "Kd"],
                    "pot": 60,
                    "your_stack": 970,
                    "opponent_stacks": [970],
                    "to_call": 0,
                    "min_raise": 10,
                    "max_raise": 970,
                    "action_history": [
                        {
                            "seat": 0,
                            "action": "post_small_blind",
                            "amount": 5,
                            "phase": "preflop",
                            "is_timeout": False,
                        }
                    ],
                }
            }
        )
        turn = TurnState.parse(sdk_state_payload(state))
        assert [str(card) for card in turn.board] == ["[ T ♠ ]", "[ 7 ♥ ]", "[ 2 ♦ ]"]
        assert turn.button_seat() == 0
        assert turn.hand_number == 2


class TestRunExternalBot:
    @pytest.fixture(scope="class")
    def parameters(self):
        return inspect.signature(chipzen.run_external_bot).parameters

    @pytest.mark.parametrize("name", ["bot_id", "token", "env", "max_matches"])
    def test_it_accepts_the_keyword_we_pass(self, name, parameters):
        assert name in parameters

    def test_it_has_no_loop_keyword(self, parameters):
        """The one we reached for first. `max_matches=None` is the forever case."""
        assert "loop" not in parameters


class TestEnvironments:
    def test_our_choices_are_the_ones_their_docstring_names(self):
        documented = inspect.getdoc(chipzen.run_external_bot) or ""
        for env in ENVIRONMENTS:
            assert f'``"{env}"``' in documented or f'"{env}"' in documented

    def test_production_is_not_an_environment(self):
        assert "production" not in ENVIRONMENTS
        assert DEFAULT_ENV in ENVIRONMENTS


class TestOurSeatSatisfiesTheirBaseClass:
    def test_decide_is_the_only_thing_they_require(self):
        """If they add an abstract method, our seat stops instantiating."""
        required = getattr(chipzen.ChipzenBot, "__abstractmethods__", frozenset())
        assert required == frozenset({"decide"})
