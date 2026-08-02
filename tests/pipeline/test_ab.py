"""Tests for the paired knob A/B harness.

The harness exists to make two preconditions impossible to skip, so those are
what these tests pin: training is always single-worker at the given seed, and a
determinism failure raises rather than reporting numbers it has just shown to be
meaningless.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.pipeline.services import ab as services_ab
from src.pipeline.services.ab import (
    AB_METHOD,
    AB_NUM_WORKERS,
    Arm,
    DeterminismError,
    format_ab_table,
    run_ab,
)


def _stub(monkeypatch, scores: dict[object, float]):
    """Stub train+evaluate; return the list of train calls for inspection.

    ``scores`` maps an arm's *first* override value (or "control") to the
    exploitability the evaluator should report, so a test can make arms differ
    without running a solver.
    """
    calls: list[dict] = []
    counter = {"n": 0}

    def _fake_train(config_name, *, num_workers, num_iterations, seed, config_overrides, runs_dir):
        counter["n"] += 1
        key = "control" if not config_overrides else next(iter(config_overrides.values()))
        calls.append(
            {
                "config_name": config_name,
                "num_workers": num_workers,
                "num_iterations": num_iterations,
                "seed": seed,
                "overrides": config_overrides,
                "key": key,
            }
        )
        return SimpleNamespace(
            run_id=f"run-{counter['n']}",
            runs_dir="data/runs",
            iterations=num_iterations,
            num_rows=10_000,
            touched_rows=1000 + counter["n"],
            coverage=(1000 + counter["n"]) / 10_000,
            runtime_seconds=1.0,
            iterations_per_second=1.0,
            dropped_updates=0,
            status="completed",
        )

    def _fake_eval(run_dir, *, method):
        assert method == AB_METHOD, "the harness must always score with the zero-variance gate"
        index = int(Path(run_dir).name.split("-")[1]) - 1
        key = calls[index]["key"]
        return {"results": {"exploitability_mbb": scores[key]}}

    monkeypatch.setattr(services_ab, "train_static", _fake_train)
    monkeypatch.setattr(services_ab, "evaluate_and_record", _fake_eval)
    return calls


class TestPreconditions:
    def test_every_arm_trains_single_worker_at_the_same_seed(self, monkeypatch):
        """The harness's central guarantee. Multi-worker training is Hogwild, so
        a differing worker count or seed would make arms incomparable."""
        calls = _stub(monkeypatch, {"control": 100.0, 110.0: 120.0, 592.0: 110.0})

        run_ab(
            "quick_test",
            [
                Arm("heavy", {"solver__pruning_threshold": 110.0}),
                Arm("light", {"solver__pruning_threshold": 592.0}),
            ],
            iterations=5000,
            seed=42,
        )

        assert len(calls) == 3  # control + 2 arms
        assert all(c["num_workers"] == AB_NUM_WORKERS == 1 for c in calls)
        assert all(c["seed"] == 42 for c in calls)
        assert all(c["num_iterations"] == 5000 for c in calls)

    def test_control_runs_first_and_carries_no_overrides(self, monkeypatch):
        calls = _stub(monkeypatch, {"control": 100.0, 110.0: 120.0})

        result = run_ab(
            "quick_test",
            [Arm("heavy", {"solver__pruning_threshold": 110.0})],
            iterations=5000,
            seed=42,
        )

        assert calls[0]["overrides"] is None
        assert result.control.name == "control"
        assert result.control.is_control
        assert result.arms[1].name == "heavy"

    def test_arm_without_overrides_is_rejected(self, monkeypatch):
        _stub(monkeypatch, {"control": 100.0})
        with pytest.raises(ValueError, match="is the control"):
            run_ab("quick_test", [Arm("oops", {})], iterations=10, seed=1)

    def test_duplicate_arm_names_are_rejected(self, monkeypatch):
        _stub(monkeypatch, {"control": 100.0})
        with pytest.raises(ValueError, match="unique"):
            run_ab(
                "quick_test",
                [Arm("same", {"a__b": 1}), Arm("same", {"a__b": 2})],
                iterations=10,
                seed=1,
            )

    def test_no_arms_is_rejected(self, monkeypatch):
        _stub(monkeypatch, {"control": 100.0})
        with pytest.raises(ValueError, match="at least one arm"):
            run_ab("quick_test", [], iterations=10, seed=1)


class TestDeterminismGate:
    def test_matching_replica_marks_the_result_verified(self, monkeypatch):
        _stub(monkeypatch, {"control": 100.0, 110.0: 120.0})

        result = run_ab(
            "quick_test",
            [Arm("heavy", {"solver__pruning_threshold": 110.0})],
            iterations=10,
            seed=1,
            verify_determinism=True,
        )

        assert result.determinism_verified is True
        # control, replica, arm — the replica is not reported as an arm
        assert [a.name for a in result.arms] == ["control", "heavy"]

    def test_mismatched_replica_raises_instead_of_reporting(self, monkeypatch):
        """A determinism failure voids every number, so it must not be a warning."""
        calls: list[str] = []

        def _fake_train(
            config_name, *, num_workers, num_iterations, seed, config_overrides, runs_dir
        ):
            calls.append("train")
            return SimpleNamespace(
                run_id=f"run-{len(calls)}",
                runs_dir="data/runs",
                iterations=num_iterations,
                num_rows=10,
                touched_rows=1,
                coverage=0.1,
                runtime_seconds=1.0,
                iterations_per_second=1.0,
                dropped_updates=0,
                status="completed",
            )

        # Two identical control runs disagree — the failure this gate exists for.
        scores = iter([100.0, 100.5, 120.0])
        monkeypatch.setattr(services_ab, "train_static", _fake_train)
        monkeypatch.setattr(
            services_ab,
            "evaluate_and_record",
            lambda run_dir, *, method: {"results": {"exploitability_mbb": next(scores)}},
        )

        with pytest.raises(DeterminismError, match="void"):
            run_ab(
                "quick_test",
                [Arm("heavy", {"solver__pruning_threshold": 110.0})],
                iterations=10,
                seed=1,
                verify_determinism=True,
            )

    def test_unverified_result_says_so(self, monkeypatch):
        _stub(monkeypatch, {"control": 100.0, 110.0: 120.0})
        result = run_ab(
            "quick_test",
            [Arm("heavy", {"solver__pruning_threshold": 110.0})],
            iterations=10,
            seed=1,
        )
        assert result.determinism_verified is False
        assert "ASSUMED" in format_ab_table(result)


class TestTable:
    def test_deltas_are_relative_to_the_control(self, monkeypatch):
        _stub(monkeypatch, {"control": 100.0, 110.0: 150.0, 592.0: 90.0})

        result = run_ab(
            "quick_test",
            [
                Arm("worse", {"solver__pruning_threshold": 110.0}),
                Arm("better", {"solver__pruning_threshold": 592.0}),
            ],
            iterations=10,
            seed=1,
        )
        table = format_ab_table(result)

        assert "+50.0%" in table
        assert "-10.0%" in table
        assert "board-restricted" in table.lower()
