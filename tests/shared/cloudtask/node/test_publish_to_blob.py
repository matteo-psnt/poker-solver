"""A run reaches the container, in an order a reader can trust.

ONE PASS, and the order inside it is the contract: rungs, then loose metadata,
then the manifests that name them. It was two passes -- a share copy that
published the manifest, and a separate Blob pass that uploaded the rungs, called
in that order -- so every tick PUT a manifest naming a rung the container did
not yet hold. `TestTheManifestGoesLast` asserted that ordering throughout and
passed anyway: with a SAS the old `publish_run` uploaded no rung at all, so the
only call it ever saw was the manifest's and `order[-1]` was trivially true.
"""

from __future__ import annotations

import json

import pytest

from src.shared.cloudtask.node import archive

SAS = "https://acct.blob.core.windows.net/checkpoints?sig=x"


def _run(tmp_path, *rungs, manifest="{}", loose=()):
    """A run holding FILE snapshots -- one `.ckpt.zst` per rung, which is what
    the trainer writes since the format changed."""
    run_dir = tmp_path / "run-a"
    run_dir.mkdir()
    for rung in rungs:
        (run_dir / rung).write_bytes(b"\x28\xb5\x2f\xfd" + b"\x00" * 64)
    for name in loose:
        (run_dir / name).write_text("{}\n")
    (run_dir / "STATIC_CHECKPOINT.json").write_text(manifest)
    return run_dir


def _capture(monkeypatch, *, exists=False):
    """A fake container that REMEMBERS what it stored, returning the PUT order.

    Modelling the store rather than stubbing each call is what makes the
    ordering assertions mean anything: `exists` answers True for an object this
    pass has already uploaded, so a manifest withheld in a test is a manifest
    the real store would have withheld too.

    `exists` seeds what the container held BEFORE the pass -- a bool, or a
    predicate on the object name.
    """
    order: list[str] = []
    held = exists if callable(exists) else (lambda _s, _n: exists)
    monkeypatch.setattr(archive.blobstore, "exists", lambda s, name: name in order or held(s, name))
    monkeypatch.setattr(
        archive.blobstore, "put_object", lambda _s, name, _p: order.append(name) or 1
    )
    monkeypatch.setattr(
        archive.blobstore, "put_bytes", lambda _s, name, _b: order.append(name) or 1
    )
    return order


class TestWhatItUploads:
    def test_every_rung_the_container_lacks(self, tmp_path, monkeypatch):
        order = _capture(monkeypatch)
        assert archive.publish_run(
            _run(tmp_path, "static-100.ckpt.zst", "static-200.ckpt.zst"), "run-a", SAS
        )
        assert order[:2] == ["run-a/static-100.ckpt.zst", "run-a/static-200.ckpt.zst"]

    def test_a_rung_already_there_is_skipped(self, tmp_path, monkeypatch):
        """Existence is a HEAD on the object. One rung is one atomically
        committed blob, so there is no half-written state to guard -- and the
        skip is why a resumed task does not re-send the whole ladder."""
        order = _capture(monkeypatch, exists=lambda _s, name: "100" in name)
        assert archive.publish_run(
            _run(tmp_path, "static-100.ckpt.zst", "static-200.ckpt.zst"), "run-a", SAS
        )
        assert "run-a/static-100.ckpt.zst" not in order
        assert "run-a/static-200.ckpt.zst" in order

    def test_loose_metadata_goes_too_and_is_always_rewritten(self, tmp_path, monkeypatch):
        """`.run.json` and the progress curve change as a run advances, so they
        are PUT unconditionally -- an existence skip would freeze them at
        whatever the first tick uploaded."""
        order = _capture(monkeypatch, exists=True)
        run_dir = _run(tmp_path, "static-100.ckpt.zst", loose=(".run.json", "progress.jsonl"))
        assert archive.publish_run(run_dir, "run-a", SAS)
        assert "run-a/.run.json" in order
        assert "run-a/progress.jsonl" in order
        assert "run-a/static-100.ckpt.zst" not in order, "an immutable rung still skips"

    def test_it_uploads_no_directory(self, tmp_path, monkeypatch):
        """A DIRECTORY snapshot is a rung from before the format changed, and
        converting one needs `zarr`, which this module may not import. Claiming
        to have moved one is how a rung ends up nowhere at all."""
        order = _capture(monkeypatch)
        run_dir = _run(tmp_path, "static-100.ckpt.zst")
        (run_dir / "static-50.zarr" / "c").mkdir(parents=True)
        (run_dir / "static-50.zarr" / ".zarray").write_text("{}")
        (run_dir / "evals").mkdir()
        assert archive.publish_run(run_dir, "run-a", SAS)
        assert not any("zarr" in name or "evals" in name for name in order), order


class TestAZeroByteRecordIsNeverPublished:
    """Measured 08-23: two reference runs' records were zeroed under retrying
    evaluate tasks, and a restored copy was re-zeroed within minutes by tasks
    holding poisoned fetches. An empty file is the residue of a truncating
    write, never content, so publishing it spreads the zeroing to every later
    fetch of the run."""

    def test_an_empty_loose_file_is_skipped_and_said_so(self, tmp_path, monkeypatch):
        order = _capture(monkeypatch)
        logged: list[str] = []
        run_dir = _run(tmp_path, "static-100.ckpt.zst", loose=(".run.json",))
        (run_dir / ".run.json").write_text("")

        assert archive.publish_run(run_dir, "run-a", SAS, logged.append)

        assert "run-a/.run.json" not in order, "a 0-byte record reached the store"
        assert any("never 0 bytes" in line for line in logged), logged

    def test_it_does_not_overwrite_the_good_copy_already_there(self, tmp_path, monkeypatch):
        """SKIPPING IS SUCCESS. The store keeps what it has -- the alternative
        is that one truncating publish destroys the last good record."""
        order = _capture(monkeypatch, exists=lambda _s, name: name.endswith(".run.json"))
        run_dir = _run(tmp_path, loose=(".run.json",))
        (run_dir / ".run.json").write_text("")

        assert archive.publish_run(run_dir, "run-a", SAS)

        assert "run-a/.run.json" not in order


class TestTheManifestGoesLast:
    """A pointer stored apart from what it points at drifts, so the manifest is
    published only once every rung it names is fetchable."""

    def test_the_rungs_land_before_it(self, tmp_path, monkeypatch):
        order = _capture(monkeypatch)
        run_dir = _run(
            tmp_path,
            "static-100.ckpt.zst",
            manifest=json.dumps({"zarr": "static-100.zarr"}),
            loose=(".run.json",),
        )
        assert archive.publish_run(run_dir, "run-a", SAS)
        assert order[-1] == "run-a/STATIC_CHECKPOINT.json", order
        assert order.index("run-a/static-100.ckpt.zst") < len(order) - 1

    def test_a_manifest_naming_a_rung_the_container_lacks_is_withheld(self, tmp_path, monkeypatch):
        """The PRUNED case. A rung is dropped without rewriting the ladder that
        advertises it, so the pointer must not move to name what no fetch can
        get. The manifest spelling is the directory one; the object is not."""
        order = _capture(monkeypatch, exists=lambda _s, name: "static-100" not in name)
        logged: list[str] = []
        run_dir = _run(tmp_path, manifest=json.dumps({"zarr": "static-100.zarr"}))
        assert archive.publish_run(run_dir, "run-a", SAS, logged.append) is False
        assert "run-a/STATIC_CHECKPOINT.json" not in order
        assert any("not in the container" in line for line in logged), logged

    def test_it_is_checked_by_the_object_name(self, tmp_path, monkeypatch):
        """`static-N.zarr` in the manifest is `static-N.ckpt.zst` in the store.
        Compared raw, every rung reads as missing and no run ever advertises a
        checkpoint again."""
        asked: list[str] = []
        _capture(monkeypatch, exists=lambda _s, name: asked.append(name) or True)
        run_dir = _run(tmp_path, manifest=json.dumps({"zarr": "static-100.zarr"}))
        assert archive.publish_run(run_dir, "run-a", SAS)
        assert "run-a/static-100.ckpt.zst" in asked, asked


class TestWhatItDoesWhenItCannot:
    def test_no_sas_publishes_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            archive.blobstore, "exists", lambda *_a: pytest.fail("asked without a SAS")
        )
        assert archive.publish_run(_run(tmp_path, "static-100.ckpt.zst"), "run-a", "") is False

    def test_a_failure_is_loud_and_does_not_kill_the_task(self, tmp_path, monkeypatch):
        """A task still making progress must not die because a copy of its
        output could not be written -- but silence is the failure shape this
        project keeps paying for, so every rung that did not land says so."""

        def _explode(*_a):
            raise RuntimeError("container is gone")

        monkeypatch.setattr(archive.blobstore, "exists", _explode)
        logged: list[str] = []
        run_dir = _run(tmp_path, "static-100.ckpt.zst", "static-200.ckpt.zst")
        assert archive.publish_run(run_dir, "run-a", SAS, logged.append) is False
        assert sum("NOT in the container" in line for line in logged) == 2

    def test_a_failed_rung_withholds_the_manifest(self, tmp_path, monkeypatch):
        """Otherwise "a killed task loses one rung" becomes "a killed task loses
        everything": the pointer moves past a rung that never landed."""
        order = _capture(monkeypatch)
        monkeypatch.setattr(
            archive.blobstore,
            "put_object",
            lambda _s, name, _p: (
                order.append(name) or (_ for _ in ()).throw(RuntimeError("transient"))
            ),
        )
        run_dir = _run(tmp_path, "static-100.ckpt.zst")
        assert archive.publish_run(run_dir, "run-a", SAS) is False
        assert "run-a/STATIC_CHECKPOINT.json" not in order

    def test_one_bad_rung_does_not_stop_the_others(self, tmp_path, monkeypatch):
        """Reported per rung, so a transient failure on one costs that one."""
        sent: list[str] = []
        monkeypatch.setattr(archive.blobstore, "exists", lambda *_a: False)

        def _put(_s, name, _p):
            if "100" in name:
                raise RuntimeError("transient")
            sent.append(name)
            return 1

        monkeypatch.setattr(archive.blobstore, "put_object", _put)
        run_dir = _run(tmp_path, "static-100.ckpt.zst", "static-200.ckpt.zst")
        assert archive.publish_run(run_dir, "run-a", SAS) is False
        assert sent == ["run-a/static-200.ckpt.zst"]
