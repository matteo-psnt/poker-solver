"""Tests for the headless (non-interactive) CLI transport."""

import argparse
import json
from datetime import UTC
from types import SimpleNamespace

import numpy as np
import pytest

from src.interfaces.cli import headless
from src.interfaces.commands import _base
from src.interfaces.commands import ledger as ledger_cmd
from src.interfaces.commands import train_static as train_static_cmd
from src.interfaces.errors import CommandError
from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.services import (
    LBR_ESTIMATOR_LABEL,
    StaticTrainingOutput,
)
from src.pipeline.services import scoring as services_scoring
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
    assert _base.resolve_run_dir(str(run), str(tmp_path / "other")) == run


def test_resolve_run_dir_resolves_id_under_runs_dir(tmp_path):
    """A bare run id should resolve under runs_dir."""
    (tmp_path / "run-b").mkdir()
    assert _base.resolve_run_dir("run-b", str(tmp_path)) == tmp_path / "run-b"


def test_resolve_run_dir_missing_raises_command_error(tmp_path):
    """An unknown run is a readable refusal, not a process exit."""
    with pytest.raises(CommandError, match="Run not found"):
        _base.resolve_run_dir("nope", str(tmp_path))


def test_no_command_writes_a_self_overwriting_result_file():
    """The run dir must not accumulate ``<op>_result.json``.

    It namespaced by op but not by invocation, so a repeated op overwrote
    itself -- a thirty-task run kept one summary. The durable records are the
    run's event log and evals/ + the ledger.
    """
    from src.interfaces.commands import _base

    assert not hasattr(_base, "write_result")


def test_main_train_json_stdout_is_clean(monkeypatch, tmp_path, capsys):
    """With --json, log noise must go to stderr and stdout must be parseable JSON."""
    out = StaticTrainingOutput(
        run_id="run-xyz",
        runs_dir=str(tmp_path),
        config_name="quick_test",
        iterations=2000,
        num_rows=1000,
        touched_rows=900,
        coverage=0.9,
        mean_visits_per_touched=2.5,
        runtime_seconds=5.0,
        iterations_per_second=400.0,
        dropped_updates=0,
        status="completed",
    )

    def _fake_train(config_name, **kwargs):
        print("noisy training log line")  # must NOT land on stdout under --json
        return out

    monkeypatch.setattr(train_static_cmd.services, "train_static", _fake_train)

    rc = headless.main(["train-static", "--config", "quick_test", "--json"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "noisy training log line" in captured.err
    assert "noisy training log line" not in captured.out
    payload = json.loads(captured.out)  # would raise if stdout were polluted
    assert payload["op"] == "train-static"
    assert payload["run_id"] == "run-xyz"
    assert not (tmp_path / "run-xyz").exists()  # the transport creates no files of its own


def test_main_evaluate_defaults_to_lbr(monkeypatch, tmp_path, capsys):
    """Evaluate defaults to LBR and carries the LBR estimator label."""
    run_dir = tmp_path / "run-xyz"
    run_dir.mkdir()

    fake_out = SimpleNamespace(
        infosets=42,
        checkpoint_iteration=1000,
        tree_fingerprint="treefp0000000000",
        results={"exploitability_mbb": 1.0, "std_error_mbb": 0.1},
    )
    # Patched on the owning submodule: `evaluate_and_record` dispatches through its
    # own namespace, which the re-export in the package __init__ does not stand in for.
    monkeypatch.setattr(services_scoring, "evaluate_run_lbr", lambda *a, **kw: fake_out)

    rc = headless.main(["evaluate", "--run", "run-xyz", "--runs-dir", str(tmp_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["method"] == "lbr"
    assert payload["estimator"] == LBR_ESTIMATOR_LABEL
    assert payload["infosets"] == 42


def _rows(monkeypatch, *documents):
    monkeypatch.setattr(ledger_cmd.connect, "engine_from_environment", lambda: object())
    rows = [eval_ledger.ledger_row(d) for d in documents]
    monkeypatch.setattr(ledger_cmd, "eval_index_rows", lambda _e: rows)


def _document(run_id, *, mbb=100.0, timestamp=None):
    return {
        "run_id": run_id,
        "method": "lbr",
        "timestamp": timestamp,
        "knobs": {"scorer": "myopic", "opponent": "blueprint", "hands": 3, "base_seed": 7},
        "results": {"exploitability_mbb": mbb, "std_error_mbb": 1.0, "num_hands": 3},
    }


def _ledger_ns(**over):
    base = {"run": None, "limit": 25, "experiment": None, "method": None, "since": None}
    return argparse.Namespace(**(base | over))


def test_cmd_ledger_lists_rows(monkeypatch):
    _rows(monkeypatch, _document("run-a"))
    payload = ledger_cmd.run(_ledger_ns())
    assert payload.op == "ledger"
    assert len(payload.rows) == 1
    assert payload.rows[0].run_id == "run-a"


def test_since_filter_compares_instants_not_strings(monkeypatch):
    """The ledger holds naive-local legacy rows beside UTC-aware ones; a
    lexicographic cutoff skews them by the writer's UTC offset."""
    from datetime import datetime, timedelta

    now = datetime.now().astimezone()
    old_naive = (now - timedelta(hours=2)).replace(tzinfo=None).isoformat()
    new_utc = (now + timedelta(hours=2)).astimezone(UTC).isoformat()
    _rows(monkeypatch, _document("old", timestamp=old_naive), _document("new", timestamp=new_utc))
    payload = ledger_cmd.run(_ledger_ns(since=now.isoformat()))
    assert [r.run_id for r in payload.rows] == ["new"]
