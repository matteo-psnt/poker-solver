"""Which runs a forget may take -- the decisions, not the deleting.

`--apply` executes the plan as printed, so every assertion is about what the
plan CONTAINS. A run that reaches it wrongly is gone, record and rungs alike.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.interfaces.commands import forget_runs

NEW, OLD, UNKNOWN = "new", "old", "lost"
GB = forget_runs.GB


def _row(run_id, *, commit=NEW, config="production", status="completed"):
    return SimpleNamespace(run_id=run_id, git_commit=commit, config_name=config, status=status)


def _new_game(commit):
    return {NEW: True, OLD: False}.get(commit)


def _plan(rows, held, **kwargs):
    return forget_runs.plan_runs(rows, held, new_game=_new_game, **kwargs)


RUNG = {"STATIC_CHECKPOINT.json": 10, "static-100.ckpt.zst": 3 * GB}


class TestReasons:
    def test_a_current_run_with_a_rung_is_kept(self):
        plan = _plan([_row("run-a")], {"run-a": RUNG})
        assert plan.plan == []

    def test_an_old_game_run_goes(self):
        plan = _plan([_row("run-a", commit=OLD)], {"run-a": RUNG})
        assert [e["run"] for e in plan.plan] == ["run-a"]
        assert plan.plan[0]["reasons"] == ["old-game"]
        assert plan.plan[0]["gib"] == 3.0

    def test_a_smoke_run_goes_whatever_its_game(self):
        plan = _plan([_row("run-a", config="quick_test")], {"run-a": RUNG})
        assert plan.plan[0]["reasons"] == ["smoke"]

    def test_a_run_with_only_a_manifest_is_empty(self):
        """A manifest is a claim, not a rung: nothing can load this run."""
        plan = _plan([_row("run-a")], {"run-a": {"STATIC_CHECKPOINT.json": 10}})
        assert plan.plan[0]["reasons"] == ["empty"]

    def test_a_run_the_container_never_heard_of_is_empty(self):
        plan = _plan([_row("run-a")], {})
        assert plan.plan[0]["reasons"] == ["empty"]

    def test_objects_without_a_record_are_orphans(self):
        plan = _plan([], {"run-x": RUNG})
        assert plan.plan[0]["reasons"] == ["orphan"]
        assert plan.plan[0]["status"] == "(no record)"


class TestProtection:
    def test_a_live_run_is_never_planned_whatever_its_reasons(self):
        plan = _plan([_row("run-a", commit=OLD, config="quick_test", status="running")], {})
        assert plan.plan == []
        assert plan.protected == ["run-a: running -- reconcile-runs first"]

    def test_an_unknown_commit_protects_rather_than_drops(self):
        plan = _plan([_row("run-a", commit=UNKNOWN)], {"run-a": RUNG})
        assert plan.plan == []
        assert plan.unknown_lineage == ["run-a"]

    def test_naming_an_unknown_lineage_run_forgets_it(self):
        plan = _plan([_row("run-a", commit=UNKNOWN)], {"run-a": RUNG}, named=["run-a"])
        assert plan.plan[0]["reasons"] == ["named"]
        assert plan.unknown_lineage == []


class TestTheExecutableList:
    def test_the_plan_names_every_object_so_apply_deletes_what_was_printed(self):
        plan = _plan([_row("run-a", commit=OLD)], {"run-a": RUNG})
        assert plan.plan[0]["objects"] == ["STATIC_CHECKPOINT.json", "static-100.ckpt.zst"]

    def test_only_narrows_to_the_reasons_asked_for(self):
        rows = [_row("run-old", commit=OLD), _row("run-smoke", config="quick_test")]
        held = {"run-old": RUNG, "run-smoke": RUNG, "run-orphan": RUNG}
        plan = _plan(rows, held, only=["smoke"])
        assert [e["run"] for e in plan.plan] == ["run-smoke"]

    def test_the_tally_is_by_first_reason_and_sums_gib(self):
        rows = [_row("run-a", commit=OLD), _row("run-b", commit=OLD, config="quick_test")]
        plan = _plan(rows, {"run-a": RUNG, "run-b": RUNG})
        assert plan.by_reason == {"old-game": {"runs": 2, "gib": 6.0}}
        assert plan.freed_gib == 6.0
