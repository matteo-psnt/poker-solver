"""A ledger row must say WHICH checkpoint produced the number.

A run id alone does not identify a measurement: a long run is evaluated at
successive checkpoints, so two rows for one run were previously indistinguishable.
That is how an eval of a 10M-iteration checkpoint was once reported as the score of
a 16M-iteration run -- the reader had no field to check. It also made
``latest_record_for_run`` unable to select between two checkpoints of one run, so a
same-run comparison could not be expressed at all.
"""

import json
from pathlib import Path

from src.engine.solver.storage.helpers import CHECKPOINT_MANIFEST_FILE
from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.services import checkpoint_iteration_of

PROVENANCE = eval_ledger.RunProvenance(
    run_id="run-abc",
    git_commit="deadbee",
    git_dirty=False,
    config_name="production",
    card_abstraction_hash="hash-a",
    action_config_hash="hash-b",
    representation_version=1,
)


def _record(checkpoint_iteration: int | None, infosets: int, timestamp: str) -> dict:
    return eval_ledger.build_record(
        provenance=PROVENANCE,
        method="lbr",
        estimator="local_best_response",
        infosets=infosets,
        knobs={"base_seed": 1},
        results={"exploitability_mbb": 1.0, "pair_samples_mbb": [1.0, 2.0]},
        result_path=Path("/tmp/payload.json"),
        timestamp=timestamp,
        checkpoint_iteration=checkpoint_iteration,
    )


def test_record_carries_checkpoint_iteration_and_infosets():
    """``infosets`` was previously accepted by build_record and silently dropped."""
    record = _record(16_160_000, infosets=18_090_076, timestamp="2026-07-18T21:00:00")
    assert record["checkpoint_iteration"] == 16_160_000
    assert record["infosets"] == 18_090_076


def test_record_evaluation_threads_the_iteration_from_the_payload(tmp_path):
    ledger_path = tmp_path / "ledger.jsonl"
    _, record = eval_ledger.record_evaluation(
        run_dir=tmp_path / "run",
        payload={
            "infosets": 18_090_076,
            "checkpoint_iteration": 16_160_000,
            "results": {"exploitability_mbb": 1.0, "pair_samples_mbb": [1.0]},
        },
        provenance=PROVENANCE,
        method="lbr",
        estimator="local_best_response",
        knobs={"base_seed": 1},
        ledger_path=ledger_path,
    )
    assert record["checkpoint_iteration"] == 16_160_000


def test_two_checkpoints_of_one_run_are_selectable(tmp_path):
    """The comparison that was previously impossible to express."""
    ledger_path = tmp_path / "ledger.jsonl"
    eval_ledger.append_record(_record(10_000_000, 10_611_180, "2026-07-18T08:00:00"), ledger_path)
    eval_ledger.append_record(_record(16_160_000, 18_090_076, "2026-07-18T21:00:00"), ledger_path)

    older = eval_ledger.latest_record_for_run("run-abc", ledger_path, 10_000_000)
    newer = eval_ledger.latest_record_for_run("run-abc", ledger_path, 16_160_000)
    assert older is not None and newer is not None
    assert older["infosets"] == 10_611_180
    assert newer["infosets"] == 18_090_076


def test_unselected_lookup_still_returns_the_newest_row(tmp_path):
    """Existing callers that pass no iteration keep their behavior."""
    ledger_path = tmp_path / "ledger.jsonl"
    eval_ledger.append_record(_record(10_000_000, 10_611_180, "2026-07-18T08:00:00"), ledger_path)
    eval_ledger.append_record(_record(16_160_000, 18_090_076, "2026-07-18T21:00:00"), ledger_path)

    latest = eval_ledger.latest_record_for_run("run-abc", ledger_path)
    assert latest is not None
    assert latest["checkpoint_iteration"] == 16_160_000


def test_iteration_is_sourced_from_the_manifest(tmp_path):
    """The value stamped into evals comes from the atomically-committed manifest."""
    (tmp_path / CHECKPOINT_MANIFEST_FILE).write_text(
        json.dumps(
            {
                "iteration": 16_160_000,
                "zarr": "checkpoint-16160000.zarr",
                "key_mapping": "key_mapping-16160000.pkl",
                "action_signatures": "action_signatures-16160000.pkl",
            }
        )
    )
    assert checkpoint_iteration_of(tmp_path) == 16_160_000


def test_pre_manifest_runs_report_no_iteration(tmp_path):
    assert checkpoint_iteration_of(tmp_path) is None


def test_missing_checkpoint_iteration_never_matches_a_selector(tmp_path):
    """Pre-manifest rows carry None; they must not satisfy a specific request."""
    ledger_path = tmp_path / "ledger.jsonl"
    eval_ledger.append_record(_record(None, 10_611_180, "2026-07-18T08:00:00"), ledger_path)
    assert eval_ledger.latest_record_for_run("run-abc", ledger_path, 10_000_000) is None
