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
from src.pipeline.evaluation.public_tree_br import (
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
