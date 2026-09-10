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
from src.pipeline.training.run_tracker import RunMetadata
from src.shared import records
from src.shared.cloudtask.node import archive
from src.shared.config import Config


def _run(published, name, *, rungs, status="completed", scored=(), suffix=".zarr"):
    """A published run holding `rungs`, as a materialised tree presents one.

    `suffix` because a marker carries whichever spelling the rung was published
    under, and prune has to read both.
    """
    run_dir = published / name
    (run_dir / "evals").mkdir(parents=True)
    for rung in rungs:
        (run_dir / f"{archive.MARKER_PREFIX}static-{rung}{suffix}").write_text("")
    # A REAL `created`, not a hand-rolled one. The fold refuses a created event
    # without a config -- so a sparse fixture reads as an UNREADABLE run, which
    # protects, and every drop assertion below would pass for the wrong reason.
    created = RunMetadata.new(
        name, "quick_test", Config.default(), action_config_hash="abc123"
    ).creation_facts()
    events = [{"event": "created", **created}]
    if status is not None:
        events.append({"event": "status", "status": status})
    (run_dir / "run.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    for index, rung in enumerate(scored):
        (run_dir / "evals" / f"e{index}.json").write_text(
            json.dumps({"run_id": name, "checkpoint_iteration": rung})
        )
    return run_dir


def _plan(published, *, price=False, **kwargs):
    """`price=False` by default: sizing is a HEAD per run against the real
    container, and most of these tests are about which rungs are chosen, not
    how big they are."""
    return prune_checkpoints.COMMAND.invoke(price=price, **kwargs)


@pytest.fixture(autouse=True)
def _record(monkeypatch):
    """A record that names nothing scored; a test that needs rows overrides.
    The store's coordinates are stubbed too: `run` reads them unconditionally,
    and a unit test must not depend on this laptop's Terraform state."""
    from src.interfaces.cloud import config as cloud_config

    monkeypatch.setattr(cloud_config.CloudConfig, "load", staticmethod(lambda: _Config()))
    monkeypatch.setattr(prune_checkpoints.connect, "engine_from_environment", lambda: object())
    monkeypatch.setattr(prune_checkpoints.connect, "record_source_from_environment", lambda: None)
    monkeypatch.setattr(prune_checkpoints.queries, "scored_rungs", lambda _: {})


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

    def test_a_rung_scored_only_in_the_record_is_kept(self, published, monkeypatch):
        """Scores stopped being written beside the run. The protection above
        reads `evals/*.json`, so without this a run scored into the database
        alone -- every run since -- has the rungs its numbers name deleted."""
        monkeypatch.setattr(prune_checkpoints.connect, "engine_from_environment", lambda: object())
        # The status still comes off `run.jsonl`, as it does for every other case
        # here; patching the engine alone would hand `_is_terminal` a fake one.
        monkeypatch.setattr(
            prune_checkpoints.connect, "record_source_from_environment", lambda: None
        )
        monkeypatch.setattr(
            prune_checkpoints.queries, "scored_rungs", lambda _: {"run-a": {100, 200}}
        )
        _run(published, "run-a", rungs=[100, 200, 300, 400, 500])
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


class _Config:
    storage_account = "a"
    share_key = "k"


class TestALegacyRunIsNotProtectedForever:
    """MEASURED: 9 completed runs holding 44 rungs, protected as "still running".

    A run written before the event log carries a `.run.json` and no log at all.
    This folded the events itself, found none, and `tail_value` handed back its
    `running` default -- so their ladders could never be pruned. Going through
    `RunMetadata.load`, which knows all three layouts, is what `reconcile-runs`
    already did; two answers to "is this run finished" was one too many.
    """

    def test_a_snapshot_only_run_reads_as_terminal(self, published):
        run_dir = published / "run-legacy"
        run_dir.mkdir(parents=True)
        (run_dir / f"{archive.MARKER_PREFIX}static-100.zarr").write_text("")
        (run_dir / f"{archive.MARKER_PREFIX}static-200.zarr").write_text("")
        metadata = RunMetadata.new(
            "run-legacy", "quick_test", Config.default(), action_config_hash="abc123"
        )
        metadata.status = "completed"
        (run_dir / ".run.json").write_text(json.dumps(metadata.to_dict()))

        plan = _plan(published, keep=1)
        entry = next((e for e in plan.plan if e["run"] == "run-legacy"), None)
        assert entry is not None, "a legacy run must be prunable, not protected forever"
        assert entry["drop"] == [100]

    def test_a_snapshot_only_run_still_training_is_protected(self, published):
        """The safe direction has to survive the fix: a run whose snapshot says
        it is running keeps its whole ladder."""
        run_dir = published / "run-live"
        run_dir.mkdir(parents=True)
        (run_dir / f"{archive.MARKER_PREFIX}static-100.zarr").write_text("")
        (run_dir / f"{archive.MARKER_PREFIX}static-200.zarr").write_text("")
        metadata = RunMetadata.new(
            "run-live", "quick_test", Config.default(), action_config_hash="abc123"
        )
        (run_dir / ".run.json").write_text(json.dumps(metadata.to_dict()))

        plan = _plan(published, keep=1)
        assert not [e for e in plan.plan if e["run"] == "run-live"]
        assert any("run-live" in p for p in plan.protected)


class TestPruningReachesTheContainer:
    """MEASURED on the real archive: prune matched `.zarr` markers only and
    deleted from the share only. After the format change that made it a no-op
    for every new run, and for the runs it did act on it left the container
    object behind forever -- the container being the one store that still grows
    and the only one nothing pruned.
    """

    def test_a_run_published_under_the_new_name_is_still_prunable(self, published):
        """The marker carries whichever spelling the rung was published under.
        Fixed on `.zarr`, this found no rungs at all and pruned nothing."""
        _run(published, "run-new", rungs=[100, 200, 300, 400], suffix=records.SNAPSHOT_SUFFIX)

        plan = _plan(published, keep=2, apply=False, runs=["run-new"])

        assert plan.rungs_dropped == 2
        assert plan.plan[0]["drop"] == [100, 200]
        assert plan.plan[0]["snapshots"] == [
            f"static-100{records.SNAPSHOT_SUFFIX}",
            f"static-200{records.SNAPSHOT_SUFFIX}",
        ]

    def test_the_container_object_is_deleted(self, published, monkeypatch):
        """The ONLY delete this command performs now."""
        _run(published, "run-a", rungs=[100, 200, 300, 400])
        deleted: list[str] = []
        from src.interfaces.cloud.store import blob

        monkeypatch.setattr(
            blob, "delete_rung", lambda _c, run, name: deleted.append(f"{run}/{name}") or True
        )

        plan = _plan(published, keep=2, apply=True, runs=["run-a"])

        assert deleted == ["run-a/static-100.ckpt.zst", "run-a/static-200.ckpt.zst"]
        assert plan.objects_deleted == 2

    def test_a_rung_is_priced_from_the_container(self, published, monkeypatch):
        """Priced by SAMPLING one rung per run: rungs of a run are the same
        table at different iterations, so per-run is where the accuracy is."""
        _run(published, "run-a", rungs=[100, 200, 300, 400])
        from src.interfaces.cloud.store import blob

        monkeypatch.setattr(blob, "rung_size", lambda *_a: 3 * 1024**3)

        plan = _plan(published, keep=2, price=True, apply=False, runs=["run-a"])

        assert plan.freed_gib == 6.0, "two rungs at 3 GiB each, priced from the container"
