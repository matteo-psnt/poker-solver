"""Tests for the headless (non-interactive) CLI transport."""

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.interfaces.cli import headless
from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.services import (
    LBR_ESTIMATOR_LABEL,
    ROLLOUT_ESTIMATOR_LABEL,
    TrainingOutput,
)
from src.shared.jsonio import json_default


def test_json_default_coerces_numpy_scalar():
    """json_default should turn numpy scalars into plain floats for JSON."""
    assert json_default(np.float64(1.5)) == 1.5
    assert isinstance(json_default(np.float64(1.5)), float)


def test_json_default_falls_back_to_str():
    """Non-numeric objects should stringify rather than raise."""
    assert json_default(object()).startswith("<object")


def test_resolve_run_dir_prefers_direct_path(tmp_path):
    """An existing directory path should resolve to itself."""
    run = tmp_path / "run-a"
    run.mkdir()
    assert headless._resolve_run_dir(str(run), str(tmp_path / "other")) == run


def test_resolve_run_dir_resolves_id_under_runs_dir(tmp_path):
    """A bare run id should resolve under runs_dir."""
    (tmp_path / "run-b").mkdir()
    assert headless._resolve_run_dir("run-b", str(tmp_path)) == tmp_path / "run-b"


def test_resolve_run_dir_missing_raises_system_exit(tmp_path):
    """An unknown run should raise SystemExit with a helpful message."""
    with pytest.raises(SystemExit, match="Run not found"):
        headless._resolve_run_dir("nope", str(tmp_path))


def test_write_result_is_namespaced_by_op_and_no_clobber(tmp_path):
    """train and evaluate results must coexist under distinct filenames."""
    headless._write_result(tmp_path, {"op": "train", "run_id": "r"})
    headless._write_result(tmp_path, {"op": "evaluate", "run_id": "r"})

    train_json = json.loads((tmp_path / "train_result.json").read_text())
    eval_json = json.loads((tmp_path / "evaluate_result.json").read_text())
    assert train_json["op"] == "train"
    assert eval_json["op"] == "evaluate"


def test_main_train_json_stdout_is_clean(monkeypatch, tmp_path, capsys):
    """With --json, log noise must go to stderr and stdout must be parseable JSON."""
    out = TrainingOutput(
        run_id="run-xyz",
        runs_dir=str(tmp_path),
        config_name="quick_test",
        iterations=2000,
        num_infosets=10,
        runtime_seconds=5.0,
        iterations_per_second=400.0,
        storage_capacity=1000,
        status="completed",
    )

    def _fake_train(config_name, **kwargs):
        print("noisy training log line")  # must NOT land on stdout under --json
        return out

    monkeypatch.setattr(headless.services, "train", _fake_train)

    rc = headless.main(["train", "--config", "quick_test", "--json"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "noisy training log line" in captured.err
    assert "noisy training log line" not in captured.out
    payload = json.loads(captured.out)  # would raise if stdout were polluted
    assert payload["op"] == "train"
    assert payload["run_id"] == "run-xyz"
    assert (tmp_path / "run-xyz" / "train_result.json").exists()


def test_main_evaluate_defaults_to_lbr(monkeypatch, tmp_path, capsys):
    """Evaluate defaults to LBR and carries the LBR estimator label."""
    run_dir = tmp_path / "run-xyz"
    run_dir.mkdir()

    fake_out = SimpleNamespace(
        infosets=42,
        checkpoint_iteration=1000,
        results={"exploitability_mbb": 1.0, "std_error_mbb": 0.1},
    )
    monkeypatch.setattr(headless.services, "evaluate_run_lbr", lambda *a, **kw: fake_out)

    rc = headless.main(["evaluate", "--run", "run-xyz", "--runs-dir", str(tmp_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["method"] == "lbr"
    assert payload["estimator"] == LBR_ESTIMATOR_LABEL
    assert payload["infosets"] == 42


def test_main_evaluate_rollout_opt_in(monkeypatch, tmp_path, capsys):
    """--method rollout uses the legacy estimator and its label."""
    run_dir = tmp_path / "run-xyz"
    run_dir.mkdir()

    fake_out = SimpleNamespace(
        infosets=7,
        checkpoint_iteration=1000,
        results={"exploitability_mbb": 9.0, "std_error_mbb": 0.5},
    )
    monkeypatch.setattr(headless.services, "evaluate_run_rollout", lambda *a, **kw: fake_out)

    rc = headless.main(
        [
            "evaluate",
            "--run",
            "run-xyz",
            "--runs-dir",
            str(tmp_path),
            "--method",
            "rollout",
            "--json",
        ]
    )

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["method"] == "rollout"
    assert payload["estimator"] == ROLLOUT_ESTIMATOR_LABEL


def _seed_eval(led_path, run_dir, run_id, *, base_seed, mbb, samples):
    """Write a per-eval payload + one ledger row for `run_id`."""
    knobs = {
        "scorer": "myopic",
        "opponent": "blueprint",
        "hands": len(samples),
        "runouts": 12,
        "include_off_tree": False,
        "base_seed": base_seed,
    }
    results = {
        "exploitability_mbb": mbb,
        "std_error_mbb": 1.0,
        "num_hands": len(samples),
        "base_seed": base_seed,
        "pair_samples_mbb": samples,
    }
    payload_path = eval_ledger.write_payload(run_dir, {"results": results}, knobs)
    provenance = eval_ledger.RunProvenance(
        run_id=run_id,
        git_commit="cafebabe" * 5,
        git_dirty=False,
        config_name="quick_test",
        card_abstraction_hash="hash",
        action_config_hash="beefcafe",
        representation_version=1,
    )
    record = eval_ledger.build_record(
        provenance=provenance,
        method="lbr",
        estimator=LBR_ESTIMATOR_LABEL,
        infosets=10,
        knobs=knobs,
        results=results,
        result_path=payload_path,
        timestamp="2026-07-17T00:00:00",
    )
    eval_ledger.append_record(record, led_path)


def test_cmd_ledger_lists_rows(tmp_path):
    led = tmp_path / "ledger.jsonl"
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    _seed_eval(led, run_dir, "run-a", base_seed=7, mbb=100.0, samples=[1.0, 2.0, 3.0])

    payload = headless._cmd_ledger(argparse.Namespace(ledger=str(led), run=None, limit=25))
    assert payload["op"] == "ledger"
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["run_id"] == "run-a"


def test_cmd_compare_valid_pairs(tmp_path):
    led = tmp_path / "ledger.jsonl"
    (tmp_path / "run-a").mkdir()
    (tmp_path / "run-b").mkdir()
    _seed_eval(led, tmp_path / "run-a", "run-a", base_seed=7, mbb=100.0, samples=[10.0, 20.0, 30.0])
    _seed_eval(led, tmp_path / "run-b", "run-b", base_seed=7, mbb=50.0, samples=[5.0, 10.0, 15.0])

    payload = headless._cmd_compare(
        argparse.Namespace(a="run-a", b="run-b", ledger=str(led), force=False, a_at=None, b_at=None)
    )
    assert payload["op"] == "compare"
    assert payload["forced"] is False
    assert payload["tier_warnings"] == []
    assert "p_value" in payload["comparison"]


def test_cmd_compare_refuses_seed_mismatch(tmp_path):
    led = tmp_path / "ledger.jsonl"
    (tmp_path / "run-a").mkdir()
    (tmp_path / "run-b").mkdir()
    _seed_eval(led, tmp_path / "run-a", "run-a", base_seed=7, mbb=100.0, samples=[1.0, 2.0, 3.0])
    _seed_eval(led, tmp_path / "run-b", "run-b", base_seed=9, mbb=50.0, samples=[1.0, 2.0, 3.0])

    with pytest.raises(SystemExit, match="Refusing to compare"):
        headless._cmd_compare(
            argparse.Namespace(
                a="run-a", b="run-b", ledger=str(led), force=False, a_at=None, b_at=None
            )
        )


def test_cmd_compare_force_overrides_mismatch(tmp_path):
    led = tmp_path / "ledger.jsonl"
    (tmp_path / "run-a").mkdir()
    (tmp_path / "run-b").mkdir()
    _seed_eval(led, tmp_path / "run-a", "run-a", base_seed=7, mbb=100.0, samples=[1.0, 2.0, 3.0])
    _seed_eval(led, tmp_path / "run-b", "run-b", base_seed=9, mbb=50.0, samples=[4.0, 5.0, 6.0])

    payload = headless._cmd_compare(
        argparse.Namespace(a="run-a", b="run-b", ledger=str(led), force=True, a_at=None, b_at=None)
    )
    assert payload["forced"] is True
    assert payload["tier_warnings"]  # non-empty: the override was recorded


def test_cmd_compare_missing_run_raises(tmp_path):
    led = tmp_path / "ledger.jsonl"
    (tmp_path / "run-a").mkdir()
    _seed_eval(led, tmp_path / "run-a", "run-a", base_seed=7, mbb=1.0, samples=[1.0, 2.0])

    with pytest.raises(SystemExit, match="No ledger entry"):
        headless._cmd_compare(
            argparse.Namespace(
                a="run-a", b="ghost", ledger=str(led), force=False, a_at=None, b_at=None
            )
        )
