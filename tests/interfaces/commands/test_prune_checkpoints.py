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


class TestTheOrphanSweep:
    """`--apply` deletes the marker BEFORE the bytes, so an interrupted sweep
    leaves an unclaimed directory rather than a rung that lies about existing.
    That trade is only sound if the leftovers are actually collected."""

    @staticmethod
    def _share(monkeypatch, files):
        from src.interfaces.cloud import config as cloud_config
        from src.interfaces.cloud.store import share

        deleted: list[str] = []

        def list_entries(_service, _name, path, *, etags=False):
            seen: dict[str, bool] = {}
            for candidate in files:
                if candidate.startswith(f"{path}/"):
                    rest = candidate[len(path) + 1 :]
                    seen[rest.split("/")[0]] = "/" in rest
            return [
                share.ShareEntry(name=n, is_directory=d, size=1) for n, d in sorted(seen.items())
            ]

        def walk(_service, _name, path, *, skip_dir=None):
            return [(p, "v1") for p in list(files) if p.startswith(f"{path}/")]

        def delete_file(_service, _name, path):
            deleted.append(path)
            return files.discard(path) is None

        def delete_directory(_service, _name, path):
            return not any(p.startswith(f"{path}/") for p in files)

        monkeypatch.setattr(share, "list_entries", list_entries)
        monkeypatch.setattr(share, "walk_files", walk)
        monkeypatch.setattr(share, "delete_file", delete_file)
        monkeypatch.setattr(share, "delete_directory", delete_directory)
        monkeypatch.setattr(share, "share_client", lambda _c: object())
        monkeypatch.setattr(cloud_config.CloudConfig, "load", staticmethod(lambda: _Config()))
        return deleted

    def test_an_unclaimed_snapshot_is_swept_and_a_claimed_one_is_not(self, published, monkeypatch):
        _run(published, "run-a", rungs=[100, 200, 300, 400])
        base = "archive/run-a"
        # The markers live on the SHARE, which is where `claimed` is read from --
        # not in the local record tree the plan was decided against.
        files = {
            f"{base}/{archive.MARKER_PREFIX}static-300.zarr",
            f"{base}/{archive.MARKER_PREFIX}static-400.zarr",
            # 300 and 400 are KEPT: claimed by a marker, so never touched.
            f"{base}/static-300.zarr/c/0",
            f"{base}/static-400.zarr/c/0",
            # 999 is the leftover shape: bytes with no marker naming them.
            f"{base}/static-999.zarr/c/0",
        }
        deleted = self._share(monkeypatch, set(files))

        prune_checkpoints.COMMAND.invoke(keep=2, price=False, apply=True, runs=["run-a"])

        assert f"{base}/static-999.zarr/c/0" in deleted, "the orphan's bytes were left behind"
        assert f"{base}/static-300.zarr/c/0" not in deleted
        assert f"{base}/static-400.zarr/c/0" not in deleted


class _Config:
    share_name = "s"
