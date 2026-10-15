"""Pooling per-seed head-to-head rows into gates."""

from __future__ import annotations

import math

import pytest

from src.pipeline.evaluation.ledger.gates import pool_matches


def match(seed, edge, se, *, at=24000, rung_b=12000, ts="2026-10-15T16:00:00", **over):
    record = {
        "run_id": "run-a",
        "method": "blueprint_match",
        "checkpoint_iteration": at,
        "knobs": {
            "run_b": "run-a",
            "rung_b": rung_b,
            "base_seed": seed,
            "num_deals": 20000,
            "policy_threshold_a": 0.35,
            "policy_threshold_b": 0.35,
        },
        "results": {"edge_mbb_per_hand": edge, "edge_se_mbb": se, "opponent": "blueprint"},
        "timestamp": ts,
        "task_id": f"t{seed}",
        "experiment_id": "opens",
    }
    record.update(over)
    return record


def test_seeds_pool_to_the_mean_with_root_sum_of_squares_error():
    (gate,) = pool_matches([match(1, 10.0, 3.0), match(2, 20.0, 4.0)])
    assert gate.edge_mbb == pytest.approx(15.0)
    assert gate.se_mbb == pytest.approx(math.sqrt(9 + 16) / 2)
    assert gate.sigma == pytest.approx(15.0 / 2.5)
    assert [s.seed for s in gate.seeds] == [1, 2]


def test_a_rescored_seed_keeps_its_newest_row_and_is_counted():
    # The 16k-vs-12k gate on pcs-6584 recorded seeds 1-3 twice.
    (gate,) = pool_matches(
        [
            match(1, 10.0, 3.0, ts="2026-10-15T06:51:00"),
            match(1, -5.0, 3.0, ts="2026-10-15T06:52:00"),
        ]
    )
    assert gate.duplicates == 1
    assert gate.edge_mbb == pytest.approx(-5.0)


def test_different_rungs_are_different_gates_newest_first():
    gates = pool_matches(
        [
            match(1, 1.0, 1.0, at=16000, ts="2026-10-15T06:00:00"),
            match(1, 2.0, 1.0, at=24000, ts="2026-10-15T14:00:00"),
        ]
    )
    assert [g.at for g in gates] == [24000, 16000]


def test_rows_that_are_not_scored_matches_are_skipped():
    settle = {"run_id": "run-a", "method": "settle", "knobs": {}, "results": {}}
    unscored = match(1, None, None)
    assert pool_matches([settle, unscored]) == []
