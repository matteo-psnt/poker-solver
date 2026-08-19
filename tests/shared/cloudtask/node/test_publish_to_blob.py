"""Rungs reach the container, and a failure to reach it is LOUD.

The pass is separate from `publish_run` on purpose: the share short-circuits on
its completion marker, so a Blob upload riding inside that loop would never
send the rungs that landed before the container existed.
"""

from __future__ import annotations

import pytest

from src.shared.cloudtask.node import archive

SAS = "https://acct.blob.core.windows.net/checkpoints?sig=x"


def _run(tmp_path, *rungs, marked=()):
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    for rung in rungs:
        (run_dir / rung).mkdir()
        (run_dir / rung / ".zarray").write_text("{}")
        if rung in marked:
            (run_dir / archive.marker_for(rung)).write_text("")
    (run_dir / "STATIC_CHECKPOINT.json").write_text("{}")
    return run_dir


class TestWhatItUploads:
    def test_every_rung_the_container_lacks(self, tmp_path, monkeypatch):
        sent = []
        monkeypatch.setattr(archive.blobstore, "exists", lambda *_a: False)
        monkeypatch.setattr(
            archive.blobstore, "put_rung", lambda _s, run, snap, _p: sent.append((run, snap)) or 1
        )
        run_dir = _run(tmp_path, "static-100.zarr", "static-200.zarr")
        assert archive.publish_rungs_to_blob(run_dir, "run-a", SAS) == 2
        assert sent == [("run-a", "static-100.zarr"), ("run-a", "static-200.zarr")]

    def test_a_rung_already_there_is_skipped(self, tmp_path, monkeypatch):
        """Existence is a HEAD on the object. One rung is one atomically
        committed blob, so there is no half-written state to guard."""
        monkeypatch.setattr(archive.blobstore, "exists", lambda _s, _r, snap: "100" in snap)
        sent = []
        monkeypatch.setattr(
            archive.blobstore, "put_rung", lambda _s, run, snap, _p: sent.append(snap) or 1
        )
        run_dir = _run(tmp_path, "static-100.zarr", "static-200.zarr")
        assert archive.publish_rungs_to_blob(run_dir, "run-a", SAS) == 1
        assert sent == ["static-200.zarr"]

    def test_the_shares_marker_does_not_short_circuit_it(self, tmp_path, monkeypatch):
        """THE REASON THIS IS A SEPARATE PASS. Every rung already on the share
        carries a marker, and `publish_run` skips those -- so a Blob upload
        inside that loop would never send the history."""
        monkeypatch.setattr(archive.blobstore, "exists", lambda *_a: False)
        sent = []
        monkeypatch.setattr(
            archive.blobstore, "put_rung", lambda _s, _r, snap, _p: sent.append(snap) or 1
        )
        run_dir = _run(tmp_path, "static-100.zarr", marked=("static-100.zarr",))
        assert archive.publish_rungs_to_blob(run_dir, "run-a", SAS) == 1
        assert sent == ["static-100.zarr"]

    def test_it_uploads_only_snapshots(self, tmp_path, monkeypatch):
        monkeypatch.setattr(archive.blobstore, "exists", lambda *_a: False)
        sent = []
        monkeypatch.setattr(
            archive.blobstore, "put_rung", lambda _s, _r, snap, _p: sent.append(snap) or 1
        )
        run_dir = _run(tmp_path, "static-100.zarr")
        (run_dir / "evals").mkdir()
        assert archive.publish_rungs_to_blob(run_dir, "run-a", SAS) == 1
        assert sent == ["static-100.zarr"]


class TestWhatItDoesWhenItCannot:
    def test_no_sas_uploads_nothing(self, tmp_path, monkeypatch):
        """The rollout and the rollback: a task dispatched before the container
        existed publishes to the share exactly as it always did."""
        monkeypatch.setattr(
            archive.blobstore, "exists", lambda *_a: pytest.fail("asked without a SAS")
        )
        assert archive.publish_rungs_to_blob(_run(tmp_path, "static-100.zarr"), "run-a", "") == 0

    def test_a_failure_is_logged_and_does_not_kill_the_task(self, tmp_path, monkeypatch):
        """Same contract as `publish_run`: a task still making progress must not
        die because a copy of its output could not be written."""

        def _explode(*_a):
            raise RuntimeError("container is gone")

        monkeypatch.setattr(archive.blobstore, "exists", _explode)
        logged: list[str] = []
        run_dir = _run(tmp_path, "static-100.zarr", "static-200.zarr")
        assert archive.publish_rungs_to_blob(run_dir, "run-a", SAS, logged.append) == 0
        assert len(logged) == 2, "every rung that did not land says so"
        assert all("NOT in the container" in line for line in logged)

    def test_one_bad_rung_does_not_stop_the_others(self, tmp_path, monkeypatch):
        monkeypatch.setattr(archive.blobstore, "exists", lambda *_a: False)

        def _put(_s, _r, snap, _p):
            if "100" in snap:
                raise RuntimeError("transient")
            return 1

        monkeypatch.setattr(archive.blobstore, "put_rung", _put)
        run_dir = _run(tmp_path, "static-100.zarr", "static-200.zarr")
        assert archive.publish_rungs_to_blob(run_dir, "run-a", SAS) == 1
