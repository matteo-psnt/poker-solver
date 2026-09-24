"""The three ways a naive drift number lies, each pinned.

Measured 09-24 on `production_nobr` s7, which is what this estimator exists to
report without a 950 MB download per rung:
  - a PURE row's drift is exactly 0.00000 -- frozen, not settled
  - thinly-reached rows freeze FIRST, so the unweighted median read 1.7-2.9x
    lower than the reach-weighted one on the river
  - drift can fall because rows are GOING pure, which is not convergence
"""

from __future__ import annotations

import numpy as np

from src.pipeline.evaluation.estimators.settling import (
    PURE_CUTOFF,
    _normalised,
    _weighted_median,
    measure_settling,
)


class _Node:
    def __init__(self, node_id, street):
        self.node_id = node_id
        self.street = street


class _Tree:
    """Two nodes on one street with DIFFERENT action counts, which is the shape
    that makes per-street stacking impossible."""

    def __init__(self, rows):
        self._rows = rows
        self.nodes = [_Node(0, "flop"), _Node(1, "flop")]

    def node_row_vector(self, array, node_id):
        return np.ones(self._rows[node_id].shape[0], dtype=bool)

    def node_action_matrix(self, array, node_id):
        return array[node_id]


class _Storage:
    def __init__(self, per_node):
        self.strategy_sum = per_node
        self.visited = per_node
        self.tree = _Tree(per_node)

    def num_infosets(self):
        return sum(r.shape[0] for r in self.strategy_sum.values())


def _storages(early_rows, late_rows):
    return _Storage(early_rows), _Storage(late_rows)


class TestAPureRowIsFrozenNotSettled:
    def test_a_pure_row_contributes_zero_drift(self):
        pure = np.array([[1000.0, 0.0]])
        early, late = _storages({0: pure, 1: pure}, {0: pure, 1: pure})
        out = measure_settling(early, late, from_iteration=100, at_iteration=200)
        assert out.streets["flop"].median_tv == 0.0
        assert out.streets["flop"].pure_fraction == 1.0

    def test_pure_fraction_is_reported_so_freezing_is_visible(self):
        """Drift falling BECAUSE rows went pure must be distinguishable."""
        mixed = np.array([[5.0, 5.0]])
        pure = np.array([[10.0, 0.0]])
        early, late = _storages({0: mixed, 1: mixed}, {0: pure, 1: pure})
        out = measure_settling(early, late, from_iteration=100, at_iteration=200)
        assert out.streets["flop"].pure_fraction == 1.0
        assert out.streets["flop"].median_tv > 0.0


class TestReachWeighting:
    def test_the_weighted_median_follows_the_traffic(self):
        """One heavy row that moved outweighs many weightless rows that did not."""
        values = np.array([0.0, 0.0, 0.0, 0.9])
        weights = np.array([1.0, 1.0, 1.0, 1000.0])
        assert _weighted_median(values, weights) == 0.9

    def test_the_unweighted_median_is_flattered_by_frozen_thin_rows(self):
        # Three frozen rows with no reach, one busy row that moved.
        thin = np.array([[1e-6, 0.0], [1e-6, 0.0], [1e-6, 0.0]])
        busy_early = np.array([[900.0, 100.0]])
        busy_late = np.array([[100.0, 900.0]])
        early, late = _storages({0: thin, 1: busy_early}, {0: thin, 1: busy_late})
        out = measure_settling(early, late, from_iteration=100, at_iteration=200)
        street = out.streets["flop"]
        assert street.median_tv == 0.0, "the typical row did not move"
        assert street.reach_weighted_tv > 0.5, "but the traffic did"


class TestPairing:
    def test_nodes_with_different_action_counts_both_report(self):
        """A street's rows pair node-for-node; stacking them raises instead."""
        two = np.array([[6.0, 4.0]])
        three = np.array([[3.0, 3.0, 4.0]])
        early, late = _storages({0: two, 1: three}, {0: two, 1: three})
        out = measure_settling(early, late, from_iteration=100, at_iteration=200)
        assert out.streets["flop"].rows == 2

    def test_a_node_whose_shape_changed_is_skipped_not_fatal(self):
        two = np.array([[6.0, 4.0]])
        three = np.array([[3.0, 3.0, 4.0]])
        early, late = _storages({0: two, 1: two}, {0: two, 1: three})
        out = measure_settling(early, late, from_iteration=100, at_iteration=200)
        assert out.streets["flop"].rows == 1


class TestNormalisation:
    def test_a_zero_row_reads_uniform_not_nan(self):
        assert _normalised(np.zeros((1, 4))).tolist() == [[0.25, 0.25, 0.25, 0.25]]

    def test_the_pure_cutoff_matches_the_policy_profile(self):
        from src.pipeline.evaluation.policy_profile import PURE_CUTOFF as PROFILE_CUTOFF

        assert PURE_CUTOFF == PROFILE_CUTOFF
