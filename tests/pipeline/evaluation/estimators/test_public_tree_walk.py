"""The compiled flop walk must reproduce the Python walk to the last bit.

The Python walk is the specification: every value, node count and tallied
mass the compiled one returns must equal it exactly, across seats, buttons,
board tiers, the conditional-chance game and the eval-time threshold.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.pipeline.evaluation.estimators.public_tree_br import (
    PublicBRConfig,
    PublicTreeBestResponse,
)
from tests.test_helpers import build_trained_test_solver

STACK = 400


def _evaluate(solver, config, *, compiled: bool):
    engine = PublicTreeBestResponse(solver, config, starting_stack=STACK)
    assert engine._compiled  # plain configs take the compiled path by default
    engine._compiled = compiled
    return engine.evaluate()


@pytest.mark.timeout(240)
@pytest.mark.parametrize(
    "config",
    [
        PublicBRConfig(num_flops=2, num_turns=2, num_rivers=2, board_seed=3),
        PublicBRConfig(
            num_flops=3, num_turns=3, num_rivers=2, board_seed=9, conditional_chance=True
        ),
        PublicBRConfig(num_flops=2, num_turns=2, num_rivers=3, board_seed=5, policy_threshold=0.1),
    ],
)
@pytest.mark.parametrize("iterations", [0, 40])
def test_compiled_walk_matches_the_python_walk(config, iterations):
    solver = build_trained_test_solver(iterations, starting_stack=STACK)
    python = _evaluate(solver, config, compiled=False)
    compiled = _evaluate(solver, config, compiled=True)
    assert compiled.exploitability_mbb == python.exploitability_mbb
    assert compiled.seat_results == python.seat_results
    assert compiled.nodes_visited == python.nodes_visited
    assert compiled.missing_policy_mass == python.missing_policy_mass


def test_modes_the_compiled_walk_does_not_cover_take_the_python_walk():
    solver = build_trained_test_solver(0, starting_stack=STACK)
    for flags in ({"decompose": True}, {"in_abstraction": True}):
        engine = PublicTreeBestResponse(
            solver, replace(PublicBRConfig(), **flags), starting_stack=STACK
        )
        assert not engine._compiled
