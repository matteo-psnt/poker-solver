"""`ledger --full` carries the document's results, not the index's summary."""

from __future__ import annotations

import argparse

from src.interfaces.commands import ledger as ledger_cmd
from src.interfaces.commands._base import ledger_for
from src.pipeline.evaluation import ledger as eval_ledger


def _record(root, run_id: str) -> None:
    run_dir = root / run_id
    run_dir.mkdir()
    eval_ledger.record_evaluation(
        run_dir=run_dir,
        payload={
            "infosets": 10,
            "results": {
                "exploitability_mbb": 1500.0,
                "std_error_mbb": 0.0,
                "num_hands": 0,
                "decomposition": {"by_street": {"preflop": 900.0, "flop": 600.0}},
            },
        },
        provenance=eval_ledger.RunProvenance(
            run_id=run_id,
            git_commit="cafebabe" * 5,
            git_dirty=False,
            config_name="quick_test",
            card_abstraction_hash="deadbeef",
            action_config_hash="beefcafe",
        ),
        method="exact_br",
        estimator="exact",
        knobs={"num_flops": 4, "num_turns": 2, "num_rivers": 2, "base_seed": 7},
    )


def _args(**over) -> argparse.Namespace:
    base = {
        "run": None,
        "experiment": None,
        "method": None,
        "since": None,
        "limit": 0,
        "full": False,
    }
    return argparse.Namespace(**{**base, **over})


def test_the_index_row_is_a_summary_and_full_is_the_document(tmp_path):
    _record(tmp_path, "run-a")
    ledger_for(tmp_path)

    summary = ledger_cmd._list(_args(), tmp_path).rows[0].results
    assert "decomposition" not in summary
    assert summary["exploitability_mbb"] == 1500.0

    full = ledger_cmd._list(_args(full=True), tmp_path).rows[0].results
    assert full["decomposition"]["by_street"] == {"preflop": 900.0, "flop": 600.0}
    assert full["exploitability_mbb"] == 1500.0
