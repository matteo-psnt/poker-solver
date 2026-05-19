"""Tests for the deterministic public-tree exact best response.

The load-bearing check is scalar equivalence: the vectorized engine must
reproduce, per hero combo, the exact ``best_response_value`` of the
:class:`RestrictedHUNL` reference game — the same betting tree, board plan,
and annulled measure walked one scalar state at a time. That pins the measure
handling (public chance weights, card-removal masking, annulment, fold and
showdown valuation) against the generic BR implementation already validated
on Kuhn/Leduc. Property tests add BR-dominates-on-policy and determinism.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.game.state import Card
from src.engine.search.range_inference import NUM_COMBOS, combo_index_for
from src.pipeline.evaluation.estimators.public_tree_br import (
    PublicBRConfig,
    PublicTreeBestResponse,
    compute_public_tree_br,
)
from src.pipeline.evaluation.reference.best_response import best_response_value, on_policy_value
from tests.pipeline.evaluation.restricted_hunl import RestrictedHUNL, blueprint_policy
from tests.test_helpers import build_trained_test_solver

CONFIG = PublicBRConfig(num_flops=2, num_turns=1, num_rivers=1, board_seed=3)
STACK = 400


def _combo(a: str, b: str) -> tuple[Card, Card]:
    return (Card.new(a), Card.new(b))


OPP_COMBOS = [
    _combo("Ah", "Kh"),
    _combo("Ac", "Qd"),
    _combo("Kd", "Kc"),
    _combo("Qs", "Qh"),
    _combo("Js", "Tc"),
    _combo("Td", "9c"),
    _combo("9s", "8s"),
    _combo("8h", "7d"),
    _combo("6c", "6d"),
    _combo("5h", "4h"),
    _combo("4c", "3s"),
    _combo("2h", "2c"),
    _combo("Ad", "5s"),
    _combo("Kh", "8c"),
    _combo("Qc", "7h"),
    _combo("3d", "2s"),
]
HERO_COMBOS = [_combo("As", "Ks"), _combo("7c", "2d"), _combo("Th", "9h")]


@pytest.fixture(scope="module")
def trained_solver():
    return build_trained_test_solver(4, starting_stack=STACK)


@pytest.fixture(scope="module")
def uniform_solver():
    return build_trained_test_solver(0, starting_stack=STACK)


def _assert_engine_matches_scalar(solver, hero_seat: int, button: int) -> None:
    engine = PublicTreeBestResponse(solver, CONFIG, starting_stack=STACK)
    for hero_combo in HERO_COMBOS:
        game = RestrictedHUNL(
            solver,
            engine._plan,
            hero_seat=hero_seat,
            hero_combo=hero_combo,
            opp_combos=OPP_COMBOS,
            button=button,
            starting_stack=STACK,
        )
        scalar = best_response_value(game, hero_seat, blueprint_policy(solver))
        reach = np.zeros(NUM_COMBOS)
        for combo in game.opp_combos:
            reach[combo_index_for(combo)] = 1.0
        values = engine.responder_values(hero_seat, button, reach)
        vectorized = values[combo_index_for(hero_combo)] / len(game.opp_combos)
        assert vectorized == pytest.approx(scalar, abs=1e-9)


@pytest.mark.timeout(30)
class TestScalarEquivalence:
    def test_uniform_blueprint_seat0(self, uniform_solver):
        _assert_engine_matches_scalar(uniform_solver, hero_seat=0, button=0)

    def test_uniform_blueprint_seat1(self, uniform_solver):
        _assert_engine_matches_scalar(uniform_solver, hero_seat=1, button=0)

    def test_trained_blueprint_seat0(self, trained_solver):
        _assert_engine_matches_scalar(trained_solver, hero_seat=0, button=0)

    def test_trained_blueprint_seat1_button1(self, trained_solver):
        _assert_engine_matches_scalar(trained_solver, hero_seat=1, button=1)


@pytest.mark.timeout(30)
def test_br_dominates_on_policy(trained_solver):
    """Per hero combo, exact BR value >= the blueprint's own on-policy value."""
    engine = PublicTreeBestResponse(trained_solver, CONFIG, starting_stack=STACK)
    policy = blueprint_policy(trained_solver)
    for hero_seat, hero_combo in ((0, HERO_COMBOS[0]), (1, HERO_COMBOS[2])):
        game = RestrictedHUNL(
            trained_solver,
            engine._plan,
            hero_seat=hero_seat,
            hero_combo=hero_combo,
            opp_combos=OPP_COMBOS,
            button=0,
            starting_stack=STACK,
            full_state_keys=True,
        )
        on_policy = on_policy_value(game, hero_seat, policy)
        reach = np.zeros(NUM_COMBOS)
        for combo in game.opp_combos:
            reach[combo_index_for(combo)] = 1.0
        values = engine.responder_values(hero_seat, 0, reach)
        best = values[combo_index_for(hero_combo)] / len(game.opp_combos)
        assert best >= on_policy - 1e-9


@pytest.mark.timeout(30)
def test_deterministic_and_nonnegative(trained_solver):
    result_a = compute_public_tree_br(trained_solver, CONFIG, starting_stack=STACK)
    result_b = compute_public_tree_br(trained_solver, CONFIG, starting_stack=STACK)
    assert result_a.exploitability_mbb == result_b.exploitability_mbb
    assert [r.value_mbb for r in result_a.seat_results] == [
        r.value_mbb for r in result_b.seat_results
    ]
    assert result_a.exploitability_mbb >= -1e-6
    assert result_a.nodes_visited > 0
    assert 0.0 <= result_a.missing_policy_mass <= 1.0


@pytest.mark.timeout(30)
def test_untrained_blueprint_is_all_fallback(uniform_solver):
    result = compute_public_tree_br(uniform_solver, CONFIG, starting_stack=STACK)
    assert result.missing_policy_mass == pytest.approx(1.0)
    assert result.exploitability_mbb >= -1e-6


class TestBranchProgress:
    """The bar an evaluation reports, and the one thing it may not cost.

    Scoring one checkpoint is ~10 minutes in which nothing about the process is
    observable from outside it. Flop branches are the outermost thing the walk
    counts — four walks of `--br-flops` each — so counting them is free, where a
    counter in the node recursion would not be.
    """

    def _engine(self, solver, on_branch=None):
        return PublicTreeBestResponse(solver, CONFIG, starting_stack=STACK, on_branch=on_branch)

    def test_progress_only_moves_forward(self, uniform_solver):
        seen: list[tuple[int, int]] = []
        engine = self._engine(uniform_solver, lambda done, total: seen.append((done, total)))
        engine.evaluate()

        assert seen, "nothing reported at all"
        assert [d for d, _ in seen] == sorted(d for d, _ in seen), "done went backwards"
        assert len({t for _, t in seen}) == 1, "the total must not move under the bar"

    def test_nothing_is_published_before_the_total_is_known(self, uniform_solver):
        """The first walk MEASURES the denominator. Publishing against a total of
        zero is a bar with no meaning, and against a growing one is a bar that
        slides backwards — so the first report is the honest 1-of-4."""
        seen: list[tuple[int, int]] = []
        engine = self._engine(uniform_solver, lambda done, total: seen.append((done, total)))
        engine.evaluate()

        first_done, first_total = seen[0]
        assert first_total > 0
        assert first_done == first_total // 4, "the first report IS one walk of four"

    def test_it_finishes_on_its_own_total(self, uniform_solver):
        """A branch skipped for having no reach is still a branch DONE — the
        earlier `continue` would have left the bar permanently short."""
        seen: list[tuple[int, int]] = []
        engine = self._engine(uniform_solver, lambda done, total: seen.append((done, total)))
        engine.evaluate()

        done, total = seen[-1]
        assert done == total == engine.branch_total

    def test_the_denominator_is_far_bigger_than_the_rung_it_replaced(self, uniform_solver):
        """The point of the change, stated as a number: this is a tiny 2-flop
        config and it still beats `1 rung` by an order of magnitude."""
        engine = self._engine(uniform_solver, lambda done, total: None)
        engine.evaluate()
        assert engine.branch_total >= 8

    def test_the_value_is_identical_with_and_without_a_bar(self, uniform_solver):
        """A progress hook that changes the number it measures is worthless."""
        watched = self._engine(uniform_solver, lambda done, total: None).evaluate()
        plain = self._engine(uniform_solver).evaluate()
        assert watched.exploitability_mbb == plain.exploitability_mbb
