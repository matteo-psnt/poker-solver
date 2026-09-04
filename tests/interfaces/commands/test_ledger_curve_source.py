"""`ledger`, `curve` and `arms` answer from either store, and must answer the same.

Compared across the whole record before the flip: 2,238 eval rows both ways,
none on one side only, all 2,238 IDENTICAL -- but only after two defects that
the COUNTS agreed through.
"""

from __future__ import annotations

import argparse
from typing import Any

from src.interfaces.commands import _base, arms, curve
from src.pipeline.evaluation import ledger as eval_ledger

DOCUMENT: dict[str, Any] = {
    "run_id": "run-a",
    "timestamp": "2026-09-01T00:00:00+00:00",
    "schema_version": 3,
    "method": "exact_br",
    "knobs": {"base_seed": 7},
    "checkpoint_iteration": 1000,
    "results": {
        "exploitability_mbb": 854.0,
        "std_error_mbb": 12.5,
        "num_hands": 4000,
        "pair_samples_mbb": [1.0, 2.0, 3.0],
        "decomposition": {"by_street": {"preflop": 900.0}},
    },
}


def test_the_index_row_is_derived_not_the_whole_document(monkeypatch):
    """`evals.payload` holds the DOCUMENT; the share's ledger holds what
    `ledger_row` derives from it. Handing a reader the document looks right and
    is not -- `results` then carries every knob and sample, and 2,224 of 2,238
    rows compared unequal while the counts matched exactly."""
    monkeypatch.setattr(_base, "eval_index_rows", _base.eval_index_rows)
    import src.adapters.postgres.queries as queries

    monkeypatch.setattr(queries, "eval_records", lambda _e, _r=None: [DOCUMENT])
    (row,) = _base.eval_index_rows(object())
    assert row["results"] == eval_ledger.ledger_row(DOCUMENT)["results"]
    assert "pair_samples_mbb" not in row["results"]
    assert "decomposition" not in row["results"]


def test_the_stamped_version_survives_the_crossing(monkeypatch):
    """The sink stored the UNSTAMPED document, so 37 rows had no
    `schema_version` while the file they mirror did."""
    import src.adapters.postgres.queries as queries

    monkeypatch.setattr(queries, "eval_records", lambda _e, _r=None: [DOCUMENT])
    (row,) = _base.eval_index_rows(object())
    assert row["schema_version"] == 3


def test_curve_asks_for_one_runs_evals_only(monkeypatch):
    """`curve_series` skips every record whose run_id is not the one asked
    about, so fetching all 2,238 documents to plot one run is 2,238 payloads
    over the wire for a handful of rows."""
    asked: list[Any] = []
    monkeypatch.setattr(curve.connect, "engine_from_environment", lambda: object())
    monkeypatch.setattr(curve, "resolve_run_id", lambda run, _e: run)
    monkeypatch.setattr(curve.queries, "rung_ladder", lambda _e, _r: [1000])
    monkeypatch.setattr(
        curve, "eval_index_rows", lambda _e, run_id=None: asked.append(run_id) or []
    )
    curve.run(argparse.Namespace(run="run-a", tier=0))
    assert asked == ["run-a"]


def test_curve_takes_its_ladder_from_the_claimed_rungs(monkeypatch):
    """The manifest names 151 rungs for a run with 23 completion markers -- it
    still advertises the 769 pruned today. `checkpoints` holds what was claimed
    as it was written, which is the truthful ladder."""
    monkeypatch.setattr(curve.connect, "engine_from_environment", lambda: object())
    monkeypatch.setattr(curve, "resolve_run_id", lambda run, _e: run)
    monkeypatch.setattr(curve.queries, "rung_ladder", lambda _e, _r: [1000, 2000])
    monkeypatch.setattr(curve, "eval_index_rows", lambda _e, run_id=None: [])
    payload = curve.run(argparse.Namespace(run="run-a", tier=0))
    assert payload.retained_iterations == [1000, 2000]


def test_arms_reads_the_database_when_one_is_configured(monkeypatch):
    """It rebuilt its index from the share's eval DOCUMENTS, which stopped being
    written on 09-03 -- so `--experiment` answered with a world that ended that
    day, and an arm scored since looked exactly like an arm never scored."""
    row = eval_ledger.ledger_row({**DOCUMENT, "experiment_id": "e1", "arm": "a1"})
    monkeypatch.setattr(arms.connect, "engine_from_environment", lambda: object())
    monkeypatch.setattr(arms, "eval_index_rows", lambda _e: [row])
    payload = arms.run(argparse.Namespace(experiment="e1", control=None))
    assert [a for tier in payload.result.tiers for a in tier.arms] == ["a1"]
