"""The strategy profile reads what the arrays hold, street by street."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.pipeline.evaluation.policy_profile import profile_policy
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
