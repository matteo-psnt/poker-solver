"""The bucket sweep must reuse ONE equity pass across every bucket count.

That is the whole reason the sweep exists: the equity pass is essentially all of
the cost, so a seven-point sweep should cost about one precompute rather than
seven. A refactor that moved ``compute_street_matrices`` inside the loop would
still return correct numbers — just ~7x slower — so the call count is asserted
directly rather than inferred from a timing.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.game.state import Street
from src.pipeline import services
from src.pipeline.services import abstraction as services_abstraction


@pytest.fixture
def synthetic_matrices():
    """One board, many hands spread across the equity range.

    Equities are deliberately spread so that more buckets genuinely resolve
    more: a degenerate all-equal input would make every bucket count look
    identical and the monotonicity assertion vacuous.
    """
    rng = np.random.default_rng(7)
    n_hands = 400
    equity = rng.uniform(0.0, 1.0, size=(1, n_hands)).astype(np.float32)
    weight = np.ones((1, n_hands), dtype=np.uint8)
    board_ids = np.array([0], dtype=np.int64)
    return board_ids, equity, weight, None


@pytest.fixture
def one_pass_service(monkeypatch, synthetic_matrices):
    """Patch out the equity pass, counting how many times it runs."""
    calls = {"n": 0}

    def fake_matrices(self, street, board_limit=None):
        calls["n"] += 1
        return synthetic_matrices

    monkeypatch.setattr(
        services_abstraction.PostflopPrecomputer,
        "compute_street_matrices",
        fake_matrices,
        raising=True,
    )
    return calls


COUNTS = [2, 4, 8, 16]


class TestSweepContract:
    def test_equity_pass_runs_exactly_once(self, one_pass_service):
        services.sweep_bucket_counts("quick_test", Street.RIVER, COUNTS)
        assert one_pass_service["n"] == 1, (
            "the sweep recomputed equities per bucket count; that is the cost it exists to avoid"
        )

    def test_returns_one_row_per_requested_count(self, one_pass_service):
        results = services.sweep_bucket_counts("quick_test", Street.RIVER, COUNTS)
        assert [r["requested_buckets"] for r in results] == COUNTS

    def test_each_row_reports_what_it_was_measured_at(self, one_pass_service):
        results = services.sweep_bucket_counts("quick_test", Street.RIVER, COUNTS)
        for row in results:
            assert row["num_buckets"] == row["requested_buckets"]
            assert row["occupied_buckets"] <= row["num_buckets"]

    def test_more_buckets_resolve_more(self, one_pass_service):
        """Sanity on the measurement itself, not just its plumbing."""
        results = services.sweep_bucket_counts("quick_test", Street.RIVER, COUNTS)
        explained = [r["variance_explained"] for r in results]
        within = [r["within_bucket_std"] for r in results]
        assert explained == sorted(explained), "variance explained must rise with bucket count"
        assert within == sorted(within, reverse=True), "within-bucket spread must fall"

    def test_writes_nothing_to_disk(self, one_pass_service, tmp_path, monkeypatch):
        """A sweep measures an abstraction; it must not leave one behind.

        An emitted artifact would be loadable by a run whose
        card_abstraction_hash never accounted for it.
        """
        monkeypatch.chdir(tmp_path)
        services.sweep_bucket_counts("quick_test", Street.RIVER, COUNTS)
        assert not list(tmp_path.rglob("metadata.json"))
        assert not (tmp_path / "data").exists()
