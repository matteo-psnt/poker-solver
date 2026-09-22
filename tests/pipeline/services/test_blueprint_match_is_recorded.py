"""A chip edge must land in the record, carrying who it was measured against.

The head-to-head gate used to write a file under run A's `evals/` instead,
because the `evals` table keys on ONE `run_id` and a match has two. A node
cannot write that file anywhere a reader will find it, so the gate the rules
make gate #1 had no cloud path at all. The resolution is that B's identity
lives in `results`, which is JSONB -- so the row is filed under A and still
says which rung, which abstraction and which recipe the other half was.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.pipeline.evaluation.ledger import tiers
from src.pipeline.services import scoring
from src.pipeline.services.scoring._shared import EvaluationOutput


@dataclass
class _Metadata:
    """Exactly the scalars `evaluate_and_record` copies onto a row."""

    run_id: str
    git_commit: str = "c" * 40
    git_dirty: bool = False
    git_branch: str = "main"
    code_snapshot: str | None = None
    config_name: str = "production"
    card_abstraction_hash: str = "a1542e88"
    action_config_hash: str = "45a36854"
    experiment_id: str | None = None
    arm: str | None = None
    parent_run_id: str | None = None
    config_hash: str = "cfg123"
    kernel: str = "static"


class _CapturingSink:
    def __init__(self) -> None:
        self.rows: list[tuple[str, dict, str]] = []

    def scored(self, eval_id, run_id, document, tier_digest) -> None:
        self.rows.append((run_id, document, tier_digest))


def _results(run_b="run-b", rung_b=4000, edge=-313.4, seed=7):
    return {
        "run_a": "run-a",
        "run_b": run_b,
        "a_mbb_per_hand": edge,
        "se_mbb": 97.1,
        "confidence_95_mbb": (-503.7, -123.1),
        "p_value": 0.0013,
        "num_deals": 2000,
        "num_hands": 4000,
        "infosets_a": 10,
        "infosets_b": 10,
        "seed": seed,
        "rung_b": rung_b,
        "card_abstraction_hash_b": "a1542e88",
        "action_config_hash_b": "45a36854",
        "config_name_b": "production",
        "git_commit_b": "d" * 40,
        "pair_samples_mbb": [1.0, -2.0],
    }


@pytest.fixture
def recorded(monkeypatch, tmp_path):
    """Run the orchestrator's blueprint_match branch over a synthetic result."""

    def _run(**overrides):
        sink = _CapturingSink()
        monkeypatch.setattr(
            scoring, "load_run_metadata", lambda run_dir, *a, **k: _Metadata(run_dir.name)
        )
        monkeypatch.setattr(
            scoring,
            "evaluate_blueprint_match",
            lambda *a, **k: EvaluationOutput(
                infosets=10, results=_results(**overrides), checkpoint_iteration=4000
            ),
        )
        payload = scoring.evaluate_and_record(
            tmp_path / "run-a",
            sink=sink,
            method="blueprint_match",
            vs_run_dir=tmp_path / "run-b",
            vs_at=4000,
            at_iteration=4000,
        )
        return payload, sink.rows

    return _run


class TestTheRowKnowsWhoItPlayed:
    def test_the_document_carries_the_opponent(self, recorded):
        _, ((_run_id, document, _digest),) = recorded()
        results = document["results"]
        assert results["run_b"] == "run-b"
        assert results["rung_b"] == 4000
        # Two arms can bucket hands differently and still play a legitimate
        # match; without this the edge is uninterpretable afterwards.
        assert results["card_abstraction_hash_b"] == "a1542e88"

    def test_the_method_is_recorded_as_the_match_it_was(self, recorded):
        payload, ((_run_id, document, _digest),) = recorded()
        assert payload.method == "blueprint_match"
        assert document["method"] == "blueprint_match"
        assert "blueprint_match" in payload.estimator

    def test_a_match_carries_no_exploitability(self, recorded):
        """It is not a bound, and a reader that treats it as one is reading a
        chip edge as a distance from equilibrium."""
        _, ((_run_id, document, _digest),) = recorded()
        assert "exploitability_mbb" not in document["results"]


class TestTheOpponentIsPartOfTheInstrument:
    """Head-to-head results are intransitive (A beats B beats C beats A), so
    two edges measured against different opponents are not two readings of one
    thing -- they must not pair, the way two board plans must not."""

    def test_a_different_opponent_is_a_different_tier(self, recorded):
        _, ((_a, _doc_a, first),) = recorded(run_b="run-b")
        _, ((_b, _doc_b, second),) = recorded(run_b="run-c")
        assert first != second

    def test_a_different_rung_of_the_same_opponent_is_a_different_tier(self, recorded):
        _, ((_a, _doc_a, first),) = recorded(rung_b=4000)
        _, ((_b, _doc_b, second),) = recorded(rung_b=4800)
        assert first != second

    def test_a_replicate_seed_is_a_different_tier(self, recorded):
        """Same as every other estimator: a seed picks the DEALS, so seeds are
        replicates to pool, not readings to average blindly."""
        _, ((_a, _doc_a, first),) = recorded(seed=7)
        _, ((_b, _doc_b, second),) = recorded(seed=17)
        assert first != second

    def test_the_knobs_are_exactly_the_pairing(self):
        knobs = tiers.build_blueprint_match_knobs(_results())
        assert knobs == {"run_b": "run-b", "rung_b": 4000, "num_deals": 2000, "base_seed": 7}
