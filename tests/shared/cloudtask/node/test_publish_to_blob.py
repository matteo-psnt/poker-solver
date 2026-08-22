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
            archive.blobstore,
            "put_object",
            lambda _s, name, _p: sent.append(tuple(name.split("/", 1))) or 1,
        )
        run_dir = _run(tmp_path, "static-100.ckpt.zst", "static-200.ckpt.zst")
        assert archive.publish_rungs_to_blob(run_dir, "run-a", SAS) == 2
        assert sent == [("run-a", "static-100.ckpt.zst"), ("run-a", "static-200.ckpt.zst")]

    def test_a_rung_already_there_is_skipped(self, tmp_path, monkeypatch):
        """Existence is a HEAD on the object. One rung is one atomically
        committed blob, so there is no half-written state to guard."""
        monkeypatch.setattr(archive.blobstore, "exists", lambda _s, name: "100" in name)
        sent = []
        monkeypatch.setattr(
            archive.blobstore,
            "put_object",
            lambda _s, name, _p: sent.append(name.split("/", 1)[1]) or 1,
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
            archive.blobstore,
            "put_object",
            lambda _s, name, _p: sent.append(name.split("/", 1)[1]) or 1,
        )
        run_dir = _run(tmp_path, "static-100.ckpt.zst", marked=("static-100.ckpt.zst",))
        assert archive.publish_rungs_to_blob(run_dir, "run-a", SAS) == 1
        assert sent == ["static-100.ckpt.zst"]

    def test_it_uploads_only_snapshots(self, tmp_path, monkeypatch):
        monkeypatch.setattr(archive.blobstore, "exists", lambda *_a: False)
        sent = []
        monkeypatch.setattr(
            archive.blobstore,
            "put_object",
            lambda _s, name, _p: sent.append(name.split("/", 1)[1]) or 1,
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

        def _put(_s, name, _p):
            if "100" in name:
                raise RuntimeError("transient")
            return 1

        monkeypatch.setattr(archive.blobstore, "put_object", _put)
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

    def test_the_blob_pass_leaves_directories_alone(self, tmp_path, monkeypatch):
        """It cannot convert one, so it must not claim to have moved one."""
        monkeypatch.setattr(archive.blobstore, "exists", lambda *_a: False)
        monkeypatch.setattr(
            archive.blobstore, "put_object", lambda *_a: pytest.fail("uploaded a directory")
        )
        assert archive.publish_rungs_to_blob(self._legacy(tmp_path), "run-a", SAS) == 0


class TestTheManifestGoesWithTheRungs:
    """A pointer stored apart from what it points at drifts. The manifest names
    which rung is current, so it belongs beside them -- and the share's copy is
    only what answers while the share still holds one.
    """

    def test_it_is_published_into_the_container(self, tmp_path, monkeypatch):
        sent: dict[str, bytes] = {}
        monkeypatch.setattr(archive.blobstore, "exists", lambda *_a: False)
        monkeypatch.setattr(archive.blobstore, "put_object", lambda _s, _n, _p: 1)
        monkeypatch.setattr(
            archive.blobstore, "put_bytes", lambda _s, name, body: sent.setdefault(name, body) and 1
        )
        run_dir = _run(tmp_path, "static-100.ckpt.zst")

        archive.publish_run(run_dir, tmp_path / "archive" / run_dir.name, sas=SAS)

        assert f"{run_dir.name}/STATIC_CHECKPOINT.json" in sent

    def test_the_rungs_land_before_it(self, tmp_path, monkeypatch):
        """Published first, the manifest advertises a rung nothing can fetch --
        which is why the share copy is written last too."""
        order: list[str] = []
        monkeypatch.setattr(archive.blobstore, "exists", lambda *_a: False)
        monkeypatch.setattr(
            archive.blobstore, "put_object", lambda _s, name, _p: order.append(name) or 1
        )
        monkeypatch.setattr(
            archive.blobstore, "put_bytes", lambda _s, name, _b: order.append(name) or 1
        )
        run_dir = _run(tmp_path, "static-100.ckpt.zst")

        archive.publish_run(run_dir, tmp_path / "archive" / run_dir.name, sas=SAS)

        assert order[-1].endswith("STATIC_CHECKPOINT.json"), order

    def test_a_container_manifest_is_read_before_the_share(self, tmp_path, monkeypatch):
        share = tmp_path / "archive" / "run-a"
        share.mkdir(parents=True)
        (share / "STATIC_CHECKPOINT.json").write_text('{"zarr": "from-the-share"}')
        monkeypatch.setattr(
            archive.blobstore, "read_object", lambda _s, _n: b'{"zarr": "from-the-container"}'
        )

        body = archive.published_manifest(share, SAS)

        assert "from-the-container" in body

    def test_the_share_answers_when_the_container_has_none(self, tmp_path, monkeypatch):
        share = tmp_path / "archive" / "run-a"
        share.mkdir(parents=True)
        (share / "STATIC_CHECKPOINT.json").write_text('{"zarr": "from-the-share"}')
        monkeypatch.setattr(archive.blobstore, "read_object", lambda _s, _n: None)

        assert "from-the-share" in archive.published_manifest(share, SAS)
