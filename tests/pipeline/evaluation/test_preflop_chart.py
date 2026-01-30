"""Tests for pipeline-side preflop chart data extraction.

The load-bearing property is encoder parity: the chart reads infosets through
the same ``encode_infoset_key`` the solver writes them with (real state, real
SPR bucket), so a key-schema or bucketing change can never silently blank the
chart again — these tests trip instead.
"""

import itertools

import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.actions import ActionType
from src.core.game.state import Card
from src.engine.solver.mccfr import MCCFRSolver
from src.pipeline.evaluation.preflop_chart import (
    _hand_classes,
    preflop_chart_data,
    preflop_open_sizes_bb,
)
from tests.test_helpers import (
    DummyCardAbstraction,
    build_test_storage,
    make_test_config,
    skew_preflop_infoset,
)

_SESSION_IDS = itertools.count()


@pytest.fixture
def blueprint():
    config = make_test_config(seed=42)
    storage = build_test_storage(session_id=f"preflop-chart-{next(_SESSION_IDS)}")
    solver = MCCFRSolver(
        action_model=ActionModel(config),
        card_abstraction=DummyCardAbstraction(),
        storage=storage,
        config=config,
    )
    yield solver
    storage.close()


def _initial_state(blueprint: MCCFRSolver):
    return blueprint.rules.create_initial_state(
        starting_stack=blueprint.config.game.starting_stack,
        hole_cards=((Card.new("As"), Card.new("Kd")), (Card.new("Qc"), Card.new("Jh"))),
        button=0,
    )


def _seed_hand(blueprint: MCCFRSolver, combo, actor=0):
    """Put all strategy mass on the first legal action for ``combo``'s class."""
    state = _initial_state(blueprint)
    legal = blueprint.rules.get_legal_actions(state, action_model=blueprint.action_model)
    skew_preflop_infoset(blueprint, state, actor=actor, combo=combo, action=legal[0])
    return legal[0]


def test_hand_classes_cover_all_169():
    classes = _hand_classes()
    assert len(classes) == 169
    assert len(set(classes)) == 169
    assert "AA" in classes and "AKs" in classes and "AKo" in classes and "32o" in classes


def test_trained_hand_appears_with_its_strategy(blueprint):
    seeded_action = _seed_hand(blueprint, (Card.new("Ah"), Card.new("Kh")))

    chart = preflop_chart_data(blueprint, position=0)

    assert "AKs" in chart.hands
    strategy = chart.hands["AKs"]
    seeded_idx = strategy.actions.index(seeded_action)
    assert strategy.probabilities[seeded_idx] == 1.0
    # Untrained classes are absent, not zero-filled.
    assert "72o" not in chart.hands
    assert chart.applied_raise is False
    assert chart.big_blind == blueprint.rules.big_blind


def test_encoder_parity_across_combo_representatives(blueprint):
    """Any combo of a class must hit the same infoset the chart queries.

    The chart picks its own representative combo per class; seeding through a
    *different* combo of the same class must still be found (keys collapse by
    class), proving both sides share the canonical encoder.
    """
    _seed_hand(blueprint, (Card.new("Ad"), Card.new("Kd")))  # AKs via diamonds

    chart = preflop_chart_data(blueprint, position=0)

    assert "AKs" in chart.hands


def test_unknown_raise_size_falls_back_to_unraised_node(blueprint):
    _seed_hand(blueprint, (Card.new("Ah"), Card.new("Kh")))

    base = preflop_chart_data(blueprint, position=0)
    fallback = preflop_chart_data(blueprint, position=0, open_raise_bb=999.0)

    assert fallback.applied_raise is False
    assert fallback.betting_sequence == base.betting_sequence
    assert "AKs" in fallback.hands


def test_in_tree_raise_advances_the_node(blueprint):
    open_sizes = preflop_open_sizes_bb(blueprint)
    assert open_sizes, "test config must define preflop open sizes"

    chart = preflop_chart_data(blueprint, position=1, open_raise_bb=open_sizes[0])

    assert chart.applied_raise is True
    assert chart.betting_sequence  # non-empty: the open raise is in the history
    assert chart.to_call > 0


def test_facing_raise_strategy_read_at_raised_node(blueprint):
    open_bb = preflop_open_sizes_bb(blueprint)[0]

    # Seed the BB's response infoset at the raised node via the raised state.
    state = _initial_state(blueprint)
    legal = blueprint.rules.get_legal_actions(state, action_model=blueprint.action_model)
    total = int(open_bb * blueprint.rules.big_blind)
    raise_action = next(
        a for a in legal if a.type == ActionType.RAISE and a.amount + state.to_call == total
    )
    raised = state.apply_action(raise_action, blueprint.rules)
    response = blueprint.rules.get_legal_actions(raised, action_model=blueprint.action_model)
    skew_preflop_infoset(
        blueprint, raised, actor=1, combo=(Card.new("Qh"), Card.new("Qs")), action=response[0]
    )

    chart = preflop_chart_data(blueprint, position=1, open_raise_bb=open_bb)

    assert chart.applied_raise is True
    assert "QQ" in chart.hands
    assert chart.hands["QQ"].probabilities[chart.hands["QQ"].actions.index(response[0])] == 1.0
