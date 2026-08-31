"""Which rungs a prune may drop — the decisions, not the deleting.

Every assertion here is about what the plan CONTAINS, because `--apply` executes
that list rather than recomputing it. A rung that reaches the plan wrongly is
deleted wrongly, and a checkpoint is not recoverable.
"""

from __future__ import annotations

import json

import pytest

from src.interfaces.commands import prune_checkpoints
from src.interfaces.errors import CommandError
from src.shared.cloudtask.node import archive


def _run(published, name, *, rungs, status="completed", scored=()):
    """A published run holding `rungs`, as the share presents one."""
    run_dir = published / name
    (run_dir / "evals").mkdir(parents=True)
    for rung in rungs:
        (run_dir / f"{archive.MARKER_PREFIX}static-{rung}.zarr").write_text("")
    events = [{"event": "created", "run_id": name}]
    if status is not None:
        events.append({"event": "status", "status": status})
    (run_dir / "run.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    for index, rung in enumerate(scored):
        (run_dir / "evals" / f"e{index}.json").write_text(
            json.dumps({"run_id": name, "checkpoint_iteration": rung})
        )
    return run_dir


def _plan(published, **kwargs):
    """`price=False`: sizing is a listing per run against the real share, and
    these tests are about which rungs are chosen, not how big they are."""
    return prune_checkpoints.COMMAND.invoke(price=False, **kwargs)


class TestWhatItDrops:
    def test_keeps_the_newest_rungs_and_drops_the_rest(self, published):
        _run(published, "run-a", rungs=[100, 200, 300, 400, 500])
        plan = _plan(published, keep=2)
        entry = next(e for e in plan.plan if e["run"] == "run-a")
        assert entry["drop"] == [100, 200, 300]
        assert entry["keeping"] == [400, 500]

    def test_a_scored_rung_is_never_dropped(self, published):
        """An eval names the checkpoint it measured. Dropping it makes a
        published number unreproducible, which is worse than the disk it frees."""
        _run(published, "run-a", rungs=[100, 200, 300, 400, 500], scored=[100, 200])
        plan = _plan(published, keep=2)
        entry = next(e for e in plan.plan if e["run"] == "run-a")
        assert entry["drop"] == [300]
        assert entry["scored_kept"] == [100, 200]

    def test_the_latest_rung_survives_keep_of_one(self, published):
        _run(published, "run-a", rungs=[100, 200, 300])
        plan = _plan(published, keep=1)
        entry = next(e for e in plan.plan if e["run"] == "run-a")
        assert 300 in entry["keeping"]
        assert 300 not in entry["drop"]

    def test_a_run_with_nothing_to_drop_is_absent_from_the_plan(self, published):
        _run(published, "run-a", rungs=[100, 200])
        assert _plan(published, keep=3).plan == []


class TestWhatItProtects:
    def test_a_running_run_is_protected(self, published):
        """It is still publishing rungs, and its newest may be mid-copy."""
        _run(published, "run-a", rungs=[100, 200, 300, 400], status="running")
        plan = _plan(published, keep=1)
        assert plan.plan == []
        assert any("run-a" in line for line in plan.protected)

    def test_a_run_with_no_status_at_all_is_protected(self, published):
        """Absence of evidence is not terminal. The default has to be the safe
        direction, because the unsafe one deletes a live ladder."""
        _run(published, "run-a", rungs=[100, 200, 300, 400], status=None)
        plan = _plan(published, keep=1)
        assert plan.plan == []

    def test_an_attempt_that_died_does_not_make_the_run_terminal(self, published):
        """`status` appears on ATTEMPT records too, and a bare scan for the last
        one returns that. Two runs still training were once folded back as
        `died` exactly this way -- here it would delete their ladders."""
        run_dir = _run(published, "run-a", rungs=[100, 200, 300, 400], status="running")
        with (run_dir / "run.jsonl").open("a") as handle:
            handle.write(json.dumps({"event": "attempt_ended", "status": "died"}) + "\n")
        assert _plan(published, keep=1).plan == []

    def test_a_run_holding_no_rungs_is_not_reported(self, published):
        (published / "run-a" / "evals").mkdir(parents=True)
        (published / "run-a" / "run.jsonl").write_text(
            json.dumps({"event": "status", "status": "completed"}) + "\n"
        )
        plan = _plan(published, keep=1)
        assert plan.plan == []
        assert plan.protected == []


class TestTheSafetyCatch:
    def test_dry_run_is_the_default(self, published):
        _run(published, "run-a", rungs=[100, 200, 300, 400])
        assert _plan(published, keep=1).applied is False

    def test_keep_below_one_is_refused(self, published):
        _run(published, "run-a", rungs=[100, 200])
        with pytest.raises(CommandError, match="at least 1"):
            _plan(published, keep=0)
