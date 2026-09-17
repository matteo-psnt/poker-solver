"""The strategy profile reads what the arrays hold, street by street."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.core.game.actions import ActionType
from src.engine.solver.infoset.index import preflop_hand_string_at
from src.pipeline.evaluation.policy_profile import (
    _DUMMY_HOLES,
    profile_policy,
    sb_first_in_fold,
)
from tests.test_helpers import build_trained_test_solver

STACK = 400


@pytest.fixture(scope="module")
def trained_solver():
    return build_trained_test_solver(4, starting_stack=STACK)


@pytest.fixture(scope="module")
def profile(trained_solver):
    return profile_policy(
        trained_solver.storage, trained_solver.rules, trained_solver.action_model, STACK
    )


def test_coverage_matches_the_storage(trained_solver, profile):
    storage = trained_solver.storage
    visited = sum(row["visited_rows"] for row in profile["streets"].values())
    rows = sum(row["rows"] for row in profile["streets"].values())
    assert rows == storage.num_infosets()
    assert visited == storage.num_touched_infosets()
    assert 0 < visited < rows


def test_entropies_are_normalised_and_bounded(profile):
    for street, row in profile["streets"].items():
        if "average_entropy" not in row:
            continue
        for key in ("average_entropy", "current_entropy"):
            q = row[key]
            assert 0.0 <= q["p10"] <= q["p50"] <= q["p90"] <= 1.0 + 1e-12, (street, key, q)
            assert 0.0 <= q["mean"] <= 1.0 + 1e-12
        assert 0.0 <= row["pure_fraction"] <= 1.0
        assert 0.0 <= row["no_positive_regret_fraction"] <= 1.0


def test_preflop_tables_are_distributions_over_all_classes(profile):
    nodes = profile["preflop"]
    labels = [node["label"] for node in nodes]
    assert labels[0] == "SB first in"
    assert any(label.startswith("BB vs open") for label in labels)
    for node in nodes:
        assert len(node["classes"]) == 169
        for probabilities in node["classes"].values():
            assert math.isclose(sum(probabilities), 1.0, abs_tol=1e-9)
        assert math.isclose(sum(node["mean_mix"].values()), 1.0, abs_tol=1e-9)
        assert math.isclose(sum(node["combo_weighted_mix"].values()), 1.0, abs_tol=1e-9)
        assert set(node["named_hands"]) >= {"AA", "72o"}


def test_untrained_rows_read_as_uniform(trained_solver):
    """A zero row is uniform in the profile because it is uniform when fielded."""
    cold = build_trained_test_solver(0, starting_stack=STACK)
    out = profile_policy(cold.storage, cold.rules, cold.action_model, STACK)
    root = out["preflop"][0]
    width = len(root["actions"])
    assert np.allclose(list(root["mean_mix"].values()), 1.0 / width)
    assert out["streets"]["preflop"]["visited_rows"] == 0


def test_bucket_spread_is_bounded(profile):
    for street, row in profile["streets"].items():
        if "bucket_spread" not in row:
            continue
        q = row["bucket_spread"]
        assert 0.0 <= q["p10"] <= q["p50"] <= q["p90"] <= 1.0 + 1e-12, (street, q)


def test_spread_separates_a_strategy_that_ignores_its_cards_from_one_that_does_not():
    """Entropy cannot see this, which is why it needed its own number.

    A node whose every bucket plays the SAME mixture has not learned its own hand
    strength -- air indistinguishable from nuts. Its rows can be exactly as mixed
    as a well-differentiated node's, so H is identical and only the spread across
    buckets tells them apart.
    """
    width, buckets = 4, 8
    identical = np.tile(np.linspace(1.0, float(width), width), (buckets, 1))
    differentiated = identical.copy()
    differentiated[::2] = differentiated[::2][:, ::-1]

    def spread_of(matrix):
        rows = matrix / matrix.sum(axis=1, keepdims=True)
        centre = rows.mean(axis=0, keepdims=True)
        return float(np.mean(0.5 * np.abs(rows - centre).sum(axis=1)))

    def entropy_of(matrix):
        rows = matrix / matrix.sum(axis=1, keepdims=True)
        return float(np.mean(-(rows * np.log(rows)).sum(axis=1) / math.log(width)))

    assert spread_of(identical) == pytest.approx(0.0, abs=1e-12)
    assert spread_of(differentiated) > 0.1
    # The blind spot this metric exists to cover: identical entropy, opposite meaning.
    assert entropy_of(identical) == pytest.approx(entropy_of(differentiated), abs=1e-12)


def _root_matrix(solver, stack):
    """The SB's first-decision rows, as a writable view, plus its fold column."""
    root = solver.rules.create_initial_state(
        starting_stack=stack, hole_cards=_DUMMY_HOLES, button=0
    )
    node_id = solver.storage.tree.node_id(root)
    legal = solver.rules.get_legal_actions(root, solver.action_model)
    fold_index = next(i for i, action in enumerate(legal) if action.type == ActionType.FOLD)
    matrix = solver.storage.tree.node_action_matrix(solver.storage.strategy_sum, node_id)
    return matrix, fold_index, len(legal)


def test_sb_fold_separates_an_abdication_from_a_strategy(trained_solver):
    """The failure this measure exists for, and exploitability cannot see it.

    Measured 09-17: r(SB fold, exploitability) = -0.855 over every profiled arm.
    The arm folding 91.8% of the button scored 3.45x BETTER than the one that
    plays, then lost the chip match to it by 3.23 sigma.
    """
    matrix, fold_index, width = _root_matrix(trained_solver, STACK)
    keep = matrix.copy()
    try:
        matrix[:] = 0.0
        matrix[:, fold_index] = 1.0
        folds = sb_first_in_fold(
            trained_solver.storage, trained_solver.rules, trained_solver.action_model, STACK
        )

        matrix[:] = 0.0
        matrix[:, (fold_index + 1) % width] = 1.0
        plays = sb_first_in_fold(
            trained_solver.storage, trained_solver.rules, trained_solver.action_model, STACK
        )
    finally:
        matrix[:] = keep

    assert folds == pytest.approx(1.0)
    assert plays == pytest.approx(0.0)


def test_sb_fold_is_combo_weighted_not_class_weighted(trained_solver):
    """A pair is 6 combos and an offsuit 12, so the 169 classes cannot count equally.

    Folding ONLY the 13 pairs is 78 of 1,326 combos = 5.88%, where counting
    classes would read 13/169 = 7.69%.
    """
    matrix, fold_index, width = _root_matrix(trained_solver, STACK)
    keep = matrix.copy()
    try:
        matrix[:] = 0.0
        matrix[:, (fold_index + 1) % width] = 1.0
        for index in range(matrix.shape[0]):
            name = preflop_hand_string_at(index)
            if len(name) == 2:  # a pair
                matrix[index] = 0.0
                matrix[index, fold_index] = 1.0
        measured = sb_first_in_fold(
            trained_solver.storage, trained_solver.rules, trained_solver.action_model, STACK
        )
    finally:
        matrix[:] = keep

    assert measured == pytest.approx(78 / 1326, abs=1e-9)
    assert measured != pytest.approx(13 / 169, abs=1e-4)
