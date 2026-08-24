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
