"""Tests for the headless (non-interactive) CLI transport."""

import argparse
import json
from datetime import UTC
from types import SimpleNamespace

import numpy as np
import pytest

from src.interfaces.cli import headless
from src.interfaces.commands import _base
from src.interfaces.commands import compare as compare_cmd
from src.interfaces.commands import ledger as ledger_cmd
from src.interfaces.commands import train_static as train_static_cmd
from src.interfaces.errors import CommandError
from src.pipeline.evaluation import ledger as eval_ledger
from src.pipeline.services import (
    LBR_ESTIMATOR_LABEL,
    StaticTrainingOutput,
)
from src.pipeline.services import evaluation as services_evaluation
from src.shared.jsonio import json_default
from tests.test_helpers import seed_ledger


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
    itself -- a thirty-leg run kept one summary. The durable records are the
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
        results={"exploitability_mbb": 1.0, "std_error_mbb": 0.1},
    )
    # Patched on the owning submodule: `evaluate_and_record` dispatches through its
    # own namespace, which the re-export in the package __init__ does not stand in for.
    monkeypatch.setattr(services_evaluation, "evaluate_run_lbr", lambda *a, **kw: fake_out)

    rc = headless.main(["evaluate", "--run", "run-xyz", "--runs-dir", str(tmp_path), "--json"])

    payload = json.loads(capsys.readouterr().out)
    assert rc == 0
    assert payload["method"] == "lbr"
    assert payload["estimator"] == LBR_ESTIMATOR_LABEL
    assert payload["infosets"] == 42


def _seed_eval(led_path, run_dir, run_id, *, base_seed, mbb, samples, method="lbr", timestamp=None):
    """Write a per-eval DOCUMENT (and a ledger row, for tests that read one).

    The document is what matters now: with no local runs directory the index is
    rebuilt from the published documents on every read, so a test that seeded
    only a ledger row was seeding something nothing reads.
    """
    knobs = {
        "scorer": "myopic",
        "opponent": "blueprint",
        "hands": len(samples or []),
        "runouts": 12,
        "include_off_tree": False,
        "base_seed": base_seed,
    }
    results = {
        "exploitability_mbb": mbb,
        "std_error_mbb": 1.0,
        "num_hands": len(samples or []),
        "base_seed": base_seed,
    }
    # `samples=None` seeds an eval with NO per-hand vector, which is what the
    # paired comparison refuses on -- a real shape, not a malformed record.
    if samples is not None:
        results["pair_samples_mbb"] = samples
    slug = eval_ledger.eval_slug(knobs)
    provenance = eval_ledger.RunProvenance(
        run_id=run_id,
        git_commit="cafebabe" * 5,
        git_dirty=False,
        config_name="quick_test",
        card_abstraction_hash="hash",
        action_config_hash="beefcafe",
    )
    record = eval_ledger.build_record(
        provenance=provenance,
        method=method,
        estimator=LBR_ESTIMATOR_LABEL,
        infosets=10,
        knobs=knobs,
        results=results,
        result_path=run_dir / "evals" / f"{slug}.json",
        timestamp=timestamp or "2026-07-17T00:00:00",
    )
    eval_ledger.write_eval(run_dir, record, slug)
    seed_ledger(led_path, eval_ledger.ledger_row(record))


def test_cmd_ledger_lists_rows(tmp_path, published):
    led = tmp_path / "ledger.jsonl"
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    _seed_eval(led, run_dir, "run-a", base_seed=7, mbb=100.0, samples=[1.0, 2.0, 3.0])

    payload = ledger_cmd.run(
        argparse.Namespace(
            ledger=str(led),
            run=None,
            limit=25,
            experiment=None,
            method=None,
            since=None,
            rebuild=False,
            migrate=False,
            runs_dir=str(tmp_path),
        )
    )
    assert payload["op"] == "ledger"
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["run_id"] == "run-a"


def test_cmd_compare_valid_pairs(tmp_path, published):
    led = tmp_path / "ledger.jsonl"
    (tmp_path / "run-a").mkdir()
    (tmp_path / "run-b").mkdir()
    _seed_eval(led, tmp_path / "run-a", "run-a", base_seed=7, mbb=100.0, samples=[10.0, 20.0, 30.0])
    _seed_eval(led, tmp_path / "run-b", "run-b", base_seed=7, mbb=50.0, samples=[5.0, 10.0, 15.0])

    payload = compare_cmd.run(
        argparse.Namespace(
            a="run-a",
            b="run-b",
            ledger=str(led),
            force=False,
            a_at=None,
            b_at=None,
            runs_dir=str(tmp_path),
        )
    )
    assert payload["op"] == "compare"
    assert payload["forced"] is False
    assert payload["tier_warnings"] == []
    assert "p_value" in payload["comparison"]


def test_cmd_compare_refuses_seed_mismatch(tmp_path, published):
    led = tmp_path / "ledger.jsonl"
    (tmp_path / "run-a").mkdir()
    (tmp_path / "run-b").mkdir()
    _seed_eval(led, tmp_path / "run-a", "run-a", base_seed=7, mbb=100.0, samples=[1.0, 2.0, 3.0])
    _seed_eval(led, tmp_path / "run-b", "run-b", base_seed=9, mbb=50.0, samples=[1.0, 2.0, 3.0])

    with pytest.raises(CommandError, match="Refusing to compare"):
        compare_cmd.run(
            argparse.Namespace(
                a="run-a",
                b="run-b",
                ledger=str(led),
                force=False,
                a_at=None,
                b_at=None,
                runs_dir=str(tmp_path),
            )
        )


def test_cmd_compare_force_overrides_mismatch(tmp_path, published):
    led = tmp_path / "ledger.jsonl"
    (tmp_path / "run-a").mkdir()
    (tmp_path / "run-b").mkdir()
    _seed_eval(led, tmp_path / "run-a", "run-a", base_seed=7, mbb=100.0, samples=[1.0, 2.0, 3.0])
    _seed_eval(led, tmp_path / "run-b", "run-b", base_seed=9, mbb=50.0, samples=[4.0, 5.0, 6.0])

    payload = compare_cmd.run(
        argparse.Namespace(
            a="run-a",
            b="run-b",
            ledger=str(led),
            force=True,
            a_at=None,
            b_at=None,
            runs_dir=str(tmp_path),
        )
    )
    assert payload["forced"] is True
    assert payload["tier_warnings"]  # non-empty: the override was recorded


def test_cmd_compare_missing_run_raises(tmp_path, published):
    led = tmp_path / "ledger.jsonl"
    (tmp_path / "run-a").mkdir()
    _seed_eval(led, tmp_path / "run-a", "run-a", base_seed=7, mbb=1.0, samples=[1.0, 2.0])

    with pytest.raises(CommandError, match="No ledger entry"):
        compare_cmd.run(
            argparse.Namespace(
                a="run-a",
                b="ghost",
                ledger=str(led),
                force=False,
                a_at=None,
                b_at=None,
                runs_dir=str(tmp_path),
            )
        )


def test_compare_refuses_samples_free_evals_even_under_force(tmp_path, published):
    """--force overrides a judgement, but cannot conjure samples that were never
    recorded. exact_br is deterministic and stores none; the forced path used to
    die on a bare KeyError."""
    led = tmp_path / "ledger.jsonl"
    for name in ("run-a", "run-b"):
        run_dir = tmp_path / name
        run_dir.mkdir()
        _seed_eval(
            led,
            run_dir,
            name,
            base_seed=3,
            mbb=1.0,
            samples=None,
            method="exact_br",
            timestamp="2026-01-01T00:00:00+00:00",
        )

    with pytest.raises(CommandError, match="no per-hand samples"):
        compare_cmd.run(
            argparse.Namespace(
                a="run-a",
                b="run-b",
                ledger=str(led),
                force=True,
                a_at=None,
                b_at=None,
                runs_dir=str(tmp_path),
            )
        )


def _ledger_ns(led, tmp_path, **over):
    base = {
        "ledger": str(led),
        "run": None,
        "limit": 25,
        "experiment": None,
        "method": None,
        "since": None,
        "rebuild": False,
        "migrate": False,
        "runs_dir": str(tmp_path),
    }
    base.update(over)
    return argparse.Namespace(**base)


def test_since_filter_compares_instants_not_strings(tmp_path, published):
    """The ledger holds naive-local legacy rows beside UTC-aware ones; a
    lexicographic cutoff skews them by the writer's UTC offset."""
    from datetime import datetime, timedelta

    led = tmp_path / "ledger.jsonl"
    now = datetime.now().astimezone()
    old_naive = (now - timedelta(hours=2)).replace(tzinfo=None).isoformat()
    new_utc = (now + timedelta(hours=2)).astimezone(UTC).isoformat()
    for run_id, ts in (("old", old_naive), ("new", new_utc)):
        (tmp_path / run_id).mkdir()
        _seed_eval(
            led, tmp_path / run_id, run_id, base_seed=1, mbb=1.0, samples=[1.0], timestamp=ts
        )

    payload = ledger_cmd.run(_ledger_ns(led, tmp_path, since=now.isoformat()))
    assert [r["run_id"] for r in payload["rows"]] == ["new"]


def test_missing_samples_error_names_the_right_record(tmp_path, published):
    """When only B lacks samples, the message must report B's method, not A's."""
    led = tmp_path / "ledger.jsonl"
    (tmp_path / "run-a").mkdir()
    _seed_eval(led, tmp_path / "run-a", "run-a", base_seed=3, mbb=1.0, samples=[1.0, 2.0])

    run_b = tmp_path / "run-b"
    run_b.mkdir()
    _seed_eval(
        led,
        run_b,
        "run-b",
        base_seed=3,
        mbb=2.0,
        samples=None,
        method="exact_br",
        timestamp="2026-01-02T00:00:00+00:00",
    )

    with pytest.raises(CommandError, match="exact_br"):
        compare_cmd.run(
            argparse.Namespace(
                a="run-a",
                b="run-b",
                ledger=str(led),
                force=True,
                a_at=None,
                b_at=None,
                runs_dir=str(tmp_path),
            )
        )
