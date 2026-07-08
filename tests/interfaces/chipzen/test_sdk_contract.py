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


class TestTheMatchStartFrame:
    """The field names we read off ``match_start``, against the SDK's own fixture.

    Every one of these was invented. `num_players` and `decision_timeout_ms` and
    `your_seat` appear NOWHERE in the SDK -- requiring the first raised on every
    real match (swallowed by `safe_mode`, so the seat silently rebuilt itself
    from a guessed config), the second made the budget sizing inert, and the
    third pinned our seat to 0 until the first decide corrected it.
    """

    @pytest.fixture(scope="class")
    def match_start(self):
        conformance = pytest.importorskip("chipzen.conformance")
        return conformance._match_start()

    def test_the_table_size_is_in_seats_not_game_config(self, match_start):
        assert "num_players" not in match_start["game_config"]
        assert len(match_start["seats"]) == 2

    def test_the_clock_is_turn_timeout_ms(self, match_start):
        assert "decision_timeout_ms" not in match_start
        assert isinstance(match_start["turn_timeout_ms"], int)

    def test_our_seat_comes_from_the_is_self_flag(self, match_start):
        assert "your_seat" not in match_start
        assert [s["seat"] for s in match_start["seats"] if s.get("is_self")] == [0]

    def test_their_game_config_parses_as_ours(self, match_start):
        """The whole point: their real frame, through our parser, unmodified."""
        from src.interfaces.chipzen.protocol import GameConfig

        config = GameConfig.parse(match_start["game_config"], seats=len(match_start["seats"]))
        assert config.num_players == 2
        assert config.depth_in_blinds == 100.0

    def test_a_seat_builds_from_their_real_frame(self, match_start):
        """End to end: no invented field, no swallowed exception."""
        from src.interfaces.chipzen.seat import MAX_BUDGET_MS, _self_seat, budget_for

        assert _self_seat(match_start) == 0
        # Their frame carries a 5,000 ms clock -- a THIRD value beside the 2 s
        # ranked one and the 30 s queue one. Half of it less the overshoot
        # allowance is 1,700, but `MAX_BUDGET_MS` caps it at 900 so a decision
        # cannot hold the GIL long enough to time out a concurrent match.
        assert budget_for(match_start["turn_timeout_ms"]) == MAX_BUDGET_MS


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
