"""`score` must name the tier a run's existing scores of this method sit in."""

from __future__ import annotations

from src.pipeline.evaluation import ledger as eval_ledger


def test_tier_label_reads_the_method_key_not_eval_method():
    """Pins the key this note filters on.

    `ledger_row` copies the document's keys verbatim, so the method lives under
    "method"; "eval_method" is the flag's name on the dispatch side. Filtering on
    the wrong one silently yields NO tiers, which reads exactly like a run that
    has never been scored -- the failure the note exists to prevent.
    """
    row = {
        "method": "exact_br",
        "knobs": {"num_flops": 4, "num_turns": 16, "num_rivers": 16, "seed": 7},
    }
    label = eval_ledger.tier_label(row)
    assert label.startswith("exact_br"), label
    assert "num_flops=4" in label, label
    assert "num_turns=16" in label, label
    # If Batch's document ever grows this key, `score._existing_tiers` has to choose.
    assert row.get("eval_method") is None
