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
    """A run holding FILE snapshots -- one `.ckpt.zst` per rung, which is what
    the trainer writes since the format changed."""
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    for rung in rungs:
        (run_dir / rung).write_bytes(b"\x28\xb5\x2f\xfd" + b"\x00" * 64)
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
        run_dir = _run(tmp_path, "static-100.ckpt.zst", "static-200.ckpt.zst")
        assert archive.publish_rungs_to_blob(run_dir, "run-a", SAS) == 2
        assert sent == [("run-a", "static-100.ckpt.zst"), ("run-a", "static-200.ckpt.zst")]

    def test_a_rung_already_there_is_skipped(self, tmp_path, monkeypatch):
        """Existence is a HEAD on the object. One rung is one atomically
        committed blob, so there is no half-written state to guard."""
        monkeypatch.setattr(archive.blobstore, "exists", lambda _s, _r, snap: "100" in snap)
        sent = []
        monkeypatch.setattr(
            archive.blobstore, "put_rung", lambda _s, run, snap, _p: sent.append(snap) or 1
        )
        run_dir = _run(tmp_path, "static-100.ckpt.zst", "static-200.ckpt.zst")
        assert archive.publish_rungs_to_blob(run_dir, "run-a", SAS) == 1
        assert sent == ["static-200.ckpt.zst"]

    def test_the_shares_marker_does_not_short_circuit_it(self, tmp_path, monkeypatch):
        """THE REASON THIS IS A SEPARATE PASS. Every rung already on the share
        carries a marker, and `publish_run` skips those -- so a Blob upload
        inside that loop would never send the history."""
        monkeypatch.setattr(archive.blobstore, "exists", lambda *_a: False)
        sent = []
        monkeypatch.setattr(
            archive.blobstore, "put_rung", lambda _s, _r, snap, _p: sent.append(snap) or 1
        )
        run_dir = _run(tmp_path, "static-100.ckpt.zst", marked=("static-100.ckpt.zst",))
        assert archive.publish_rungs_to_blob(run_dir, "run-a", SAS) == 1
        assert sent == ["static-100.ckpt.zst"]

    def test_it_uploads_only_snapshots(self, tmp_path, monkeypatch):
        monkeypatch.setattr(archive.blobstore, "exists", lambda *_a: False)
        sent = []
        monkeypatch.setattr(
            archive.blobstore, "put_rung", lambda _s, _r, snap, _p: sent.append(snap) or 1
        )
        run_dir = _run(tmp_path, "static-100.ckpt.zst")
        (run_dir / "evals").mkdir()
        assert archive.publish_rungs_to_blob(run_dir, "run-a", SAS) == 1
        assert sent == ["static-100.ckpt.zst"]


class TestWhatItDoesWhenItCannot:
    def test_no_sas_uploads_nothing(self, tmp_path, monkeypatch):
        """The rollout and the rollback: a task dispatched before the container
        existed publishes to the share exactly as it always did."""
        monkeypatch.setattr(
            archive.blobstore, "exists", lambda *_a: pytest.fail("asked without a SAS")
        )
        assert (
            archive.publish_rungs_to_blob(_run(tmp_path, "static-100.ckpt.zst"), "run-a", "") == 0
        )

    def test_a_failure_is_logged_and_does_not_kill_the_task(self, tmp_path, monkeypatch):
        """Same contract as `publish_run`: a task still making progress must not
        die because a copy of its output could not be written."""

        def _explode(*_a):
            raise RuntimeError("container is gone")

        monkeypatch.setattr(archive.blobstore, "exists", _explode)
        logged: list[str] = []
        run_dir = _run(tmp_path, "static-100.ckpt.zst", "static-200.ckpt.zst")
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
        run_dir = _run(tmp_path, "static-100.ckpt.zst", "static-200.ckpt.zst")
        assert archive.publish_rungs_to_blob(run_dir, "run-a", SAS) == 1


class TestTheFlip:
    """With a SAS, the snapshot bytes stop going to the share.

    What still goes there is the metadata -- manifests, `.run.json`, loose
    result files -- which is kilobytes against the 831 GiB the snapshots were.
    """

    def test_snapshots_do_not_reach_the_share(self, tmp_path):
        """Asserted as an OUTCOME rather than by patching the copier: the same
        helper copies the manifest, which still belongs on the share."""
        run_dir = _run(tmp_path, "static-100.ckpt.zst")
        destination = tmp_path / "share" / "run-a"
        assert archive.publish_run(run_dir, destination, sas=SAS) is True
        assert not (destination / "static-100.ckpt.zst").exists(), "the bytes went to the share"
        assert (destination / "STATIC_CHECKPOINT.json").exists(), "the manifest still belongs here"

    def test_the_marker_is_still_written(self, tmp_path):
        """It is what says the rung is complete SOMEWHERE. `migrate-checkpoints`
        refuses an unmarked rung, `prune-checkpoints` reads markers to know what
        the share holds, and `_rungs_landed` accepts one in place of the bytes.
        """
        run_dir = _run(tmp_path, "static-100.ckpt.zst")
        destination = tmp_path / "share" / "run-a"
        archive.publish_run(run_dir, destination, sas=SAS)
        assert (destination / archive.marker_for("static-100.ckpt.zst")).exists()

    def test_the_manifest_is_published_on_a_marker_alone(self, tmp_path):
        """Requiring the DIRECTORY would freeze manifest publishing the moment
        snapshots stopped landing here: the manifest would name rungs the share
        does not hold and the run would never advertise a checkpoint again."""
        run_dir = _run(tmp_path, "static-100.ckpt.zst")
        (run_dir / "STATIC_CHECKPOINT.json").write_text(
            '{"iteration": 100, "zarr": "static-100.ckpt.zst", "retained": []}'
        )
        destination = tmp_path / "share" / "run-a"
        assert archive.publish_run(run_dir, destination, sas=SAS) is True
        assert (destination / "STATIC_CHECKPOINT.json").exists()

    def test_without_a_sas_the_share_still_gets_the_bytes(self, tmp_path):
        """The rollback. A task dispatched before the container existed
        publishes exactly as it always did."""
        run_dir = _run(tmp_path, "static-100.ckpt.zst")
        destination = tmp_path / "share" / "run-a"
        assert archive.publish_run(run_dir, destination) is True
        assert (destination / "static-100.ckpt.zst").is_file()


class TestALegacyRungIsNeverStranded:
    """A DIRECTORY snapshot is a rung from before the format changed. The
    container only takes files -- converting a directory needs `zarr`, which
    this module may not import -- so the share is still its home.

    Skipping the share copy on the strength of a SAS marked such a rung
    complete and wrote it nowhere: no bytes on the share, none in the
    container, and a marker claiming it was published.
    """

    def _legacy(self, tmp_path):
        run_dir = tmp_path / "run-a"
        (run_dir / "static-100.zarr" / "c").mkdir(parents=True)
        (run_dir / "static-100.zarr" / ".zarray").write_text("{}")
        (run_dir / "static-100.zarr" / "c" / "0").write_bytes(b"\x01" * 32)
        (run_dir / "STATIC_CHECKPOINT.json").write_text("{}")
        return run_dir

    def test_it_reaches_the_share_even_with_a_sas(self, tmp_path):
        destination = tmp_path / "share" / "run-a"
        archive.publish_run(self._legacy(tmp_path), destination, sas=SAS)
        assert (destination / "static-100.zarr" / ".zarray").exists(), "the rung went nowhere"

    def test_the_marker_follows_the_bytes(self, tmp_path):
        destination = tmp_path / "share" / "run-a"
        archive.publish_run(self._legacy(tmp_path), destination, sas=SAS)
        assert (destination / archive.marker_for("static-100.zarr")).exists()

    def test_the_blob_pass_leaves_directories_alone(self, tmp_path, monkeypatch):
        """It cannot convert one, so it must not claim to have moved one."""
        monkeypatch.setattr(archive.blobstore, "exists", lambda *_a: False)
        monkeypatch.setattr(
            archive.blobstore, "put_rung", lambda *_a: pytest.fail("uploaded a directory")
        )
        assert archive.publish_rungs_to_blob(self._legacy(tmp_path), "run-a", SAS) == 0
