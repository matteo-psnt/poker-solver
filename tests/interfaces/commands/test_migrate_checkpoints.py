"""The sweep stages a rung locally before tarring it, and can prove it.

WHY THIS EXISTS. The staging helper was added in one commit and its call site
was never replaced -- `38 insertions(+)`, zero deletions. The function sat there
as dead code, the loop went on tarring straight off the share, and three sweeps
timed out having moved nothing. The commit was clean, 2,776 tests passed and the
whole gate was green, because nothing asserted WHERE the bytes were read from.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.engine.solver.storage import snapshot_format
from src.interfaces.commands import migrate_checkpoints
from src.shared import records
from src.shared.cloudtask.node import archive

ARRAYS = {"regrets": np.arange(64, dtype=np.float32), "visited": np.ones(16, dtype=np.uint8)}


@pytest.fixture
def share(tmp_path, monkeypatch):
    """A share holding one marked ZARR rung -- what the migration converts."""
    import zarr

    root = tmp_path / "share" / "archive" / "run-a"
    root.mkdir(parents=True)
    group = zarr.open(zarr.DirectoryStore(str(root / "static-100.zarr")), mode="w")
    for name, array in ARRAYS.items():
        group.create_dataset(name, data=array, dtype=array.dtype)
    group.attrs["iteration"] = 100
    group.attrs["fingerprint"] = "cafe"
    (root / archive.marker_for("static-100.zarr")).write_text("")
    # The manifest is the CLAIM, and the sweep migrates what is claimed. A run
    # whose manifest names nothing has snapshots no reader can resolve, and
    # moving them would fill the container with rungs nothing can ask for.
    (root / records.STATIC_CHECKPOINT).write_text(
        json.dumps({"zarr": "static-100.zarr", "iteration": 100, "retained": []})
    )
    monkeypatch.setenv("POKER_SOLVER_CHECKPOINT_SAS", "https://a.blob.core.windows.net/c?sig=x")
    monkeypatch.setenv("RUN_WORK_DIR", str(tmp_path / "work"))
    return tmp_path / "share"


def _run(share, monkeypatch, **over):
    seen: list[Path] = []
    monkeypatch.setattr(
        migrate_checkpoints.blobstore if hasattr(migrate_checkpoints, "blobstore") else archive,
        "marker_for",
        archive.marker_for,
        raising=False,
    )
    import src.shared.cloudtask.node.blobstore as blobstore

    monkeypatch.setattr(blobstore, "exists", lambda *_a: False)
    monkeypatch.setattr(
        blobstore, "put_rung", lambda _s, _r, _n, path: (seen.append(Path(path)), 4096)[1]
    )
    args = argparse.Namespace(
        share=str(share),
        runs=None,
        limit=0,
        verify=False,
        drop_share=False,
        apply=False,
        **over,
    )
    return migrate_checkpoints.run(args), seen


class TestItConvertsRatherThanCopies:
    """The share holds zarr and the container holds the format that replaced
    it, so migrating is a re-encode -- not a copy, and not a tar of a copy."""

    def test_the_object_is_named_for_the_new_format(self, share, monkeypatch):
        _payload, seen = _run(share, monkeypatch)
        assert seen, "nothing was uploaded at all"
        assert seen[0].name == "static-100.ckpt.zst"

    def test_what_is_uploaded_reads_back_as_the_same_arrays(self, share, monkeypatch, tmp_path):
        """A conversion that changes a value is a run trained on a different
        number, and nothing downstream could see it."""
        kept: dict[str, Path] = {}
        import src.shared.cloudtask.node.blobstore as blobstore

        monkeypatch.setattr(blobstore, "exists", lambda *_a: False)

        def _put(_s, _r, _n, path):
            kept["at"] = Path(tmp_path / "kept.ckpt.zst")
            shutil.copyfile(path, kept["at"])
            return Path(path).stat().st_size

        monkeypatch.setattr(blobstore, "put_rung", _put)
        migrate_checkpoints.run(
            argparse.Namespace(
                share=str(share),
                runs=None,
                limit=0,
                verify=False,
                drop_share=False,
                apply=False,
            )
        )
        arrays, attrs = snapshot_format.read_snapshot(kept["at"])
        for name, original in ARRAYS.items():
            assert np.array_equal(arrays[name], original), name
        assert attrs["fingerprint"] == "cafe", "the tree identity must survive the conversion"

    def test_the_path_handed_to_put_rung_is_not_on_the_share(self, share, monkeypatch):
        """Reading a rung straight off the share walks ~5,500 chunk files
        serially over SMB -- more than eight minutes for one, and three sweeps
        died proving it."""
        _payload, seen = _run(share, monkeypatch)
        assert share not in seen[0].parents, f"read straight off the share: {seen[0]}"

    def test_it_is_under_the_node_work_directory(self, share, monkeypatch, tmp_path):
        _payload, seen = _run(share, monkeypatch)
        assert (tmp_path / "work") in seen[0].parents

    def test_staging_is_cleaned_up(self, share, monkeypatch, tmp_path):
        """A node runs many rungs and its disk is 256 GB; leaving each staged
        copy behind fills it well before the ladder is done."""
        _payload, _seen = _run(share, monkeypatch)
        staged = tmp_path / "work" / "migrate-staging"
        assert not list(staged.glob("static-*")), "a staged rung was left behind"


class TestWhatItRefuses:
    def test_an_unmarked_rung_is_not_uploaded(self, share, monkeypatch):
        """Copying one in would launder a possibly-partial snapshot into a store
        where existence MEANS complete."""
        (share / "archive" / "run-a" / archive.marker_for("static-100.zarr")).unlink()
        payload, seen = _run(share, monkeypatch)
        assert seen == []
        assert payload.unmarked == ["run-a/static-100.zarr"]

    def test_no_sas_refuses_outright(self, share, monkeypatch):
        from src.interfaces.errors import CommandError

        monkeypatch.delenv("POKER_SOLVER_CHECKPOINT_SAS")
        with pytest.raises(CommandError):
            _run(share, monkeypatch)


def _header(body: bytes) -> bytes:
    """The wire shape `read_head` returns: an 8-byte length, then the JSON."""
    return len(body).to_bytes(snapshot_format.HEADER_LENGTH_BYTES, "little") + body


def _args(share, **over):
    base = {
        "share": str(share),
        "runs": None,
        "limit": 0,
        "verify": False,
        "drop_share": False,
        "apply": False,
    }
    return argparse.Namespace(**(base | over))


@pytest.fixture
def stranded(tmp_path, monkeypatch):
    """A run in the state six 300M runs are actually in: markers and a
    manifest on the share, and not one byte of snapshot beside them."""
    root = tmp_path / "share" / "archive" / "run-b"
    root.mkdir(parents=True)
    (root / archive.marker_for("static-100.zarr")).write_text("")
    (root / records.STATIC_CHECKPOINT).write_text(
        json.dumps({"zarr": "static-100.zarr", "iteration": 100, "retained": []})
    )
    monkeypatch.setenv("POKER_SOLVER_CHECKPOINT_SAS", "https://a.blob.core.windows.net/c?sig=x")
    monkeypatch.setenv("RUN_WORK_DIR", str(tmp_path / "work"))
    return tmp_path / "share"


class TestTheGateSeesWhatIsNotOnTheShare:
    """`--verify` listed share DIRECTORIES, so a run holding none contributed
    zero rows. Six runs were in exactly that state -- 316 marked rungs whose
    bytes live only as tars -- and the gate reported nothing missing while it
    was one confirmation away from deleting the only other copy.
    """

    def test_a_rung_with_no_bytes_at_all_is_a_phantom_and_blocks_nothing(
        self, stranded, monkeypatch, capsys
    ):
        """MEASURED: 1,030 of these against 3 genuinely gone. `prune` drops a
        snapshot without rewriting the manifest that advertises it, so a
        settled run's ladder names rungs it has not held for weeks. Treating
        the manifest as the authority on what EXISTS reported all of them as
        loss and refused a deletion that was safe."""
        import src.shared.cloudtask.node.blobstore as blobstore

        monkeypatch.setattr(blobstore, "exists", lambda *_a: False)
        payload = migrate_checkpoints.run(_args(stranded, verify=True))
        migrate_checkpoints.render(payload)

        assert payload.phantom == ["run-b/static-100.zarr"]
        out = capsys.readouterr().out
        assert "SHARE: every rung it holds is in the container" in out

    def test_a_rung_the_share_holds_and_the_container_lacks_refuses_the_share(
        self, share, monkeypatch, capsys
    ):
        import src.shared.cloudtask.node.blobstore as blobstore

        monkeypatch.setattr(blobstore, "exists", lambda *_a: False)
        migrate_checkpoints.render(migrate_checkpoints.run(_args(share, verify=True)))

        assert "SHARE: DO NOT DELETE" in capsys.readouterr().out

    def test_a_share_directory_no_manifest_names_is_not_migrated(self, share, monkeypatch):
        """`_prune` deletes what the manifest does not name, so an orphan
        directory is residue. Uploading it fills the container with rungs
        nothing can resolve -- and it is the one thing safe to delete."""
        (share / "archive" / "run-a" / "static-999.zarr").mkdir()
        _payload, seen = _run(share, monkeypatch)

        assert [p.name for p in seen] == ["static-100.ckpt.zst"]

    def test_the_orphan_is_reported_rather_than_ignored(self, share, monkeypatch):
        import src.shared.cloudtask.node.blobstore as blobstore

        (share / "archive" / "run-a" / "static-999.zarr").mkdir()
        monkeypatch.setattr(blobstore, "exists", lambda *_a: True)
        monkeypatch.setattr(blobstore, "read_head", lambda *_a: _header(b'{"arrays": []}'))
        payload = migrate_checkpoints.run(_args(share, verify=True))

        assert payload.unclaimed == ["run-a/static-999.zarr"]
        assert payload.rungs_already_there == 1, "the claimed rung verified clean"


class TestPresentIsNotTheSameAsReadable:
    def test_an_object_that_will_not_open_is_not_counted_as_migrated(self, share, monkeypatch):
        """An upload that died mid-stream still answers 200 to a HEAD. This is
        the check that gates deleting the other copy, so it opens the header."""
        import src.shared.cloudtask.node.blobstore as blobstore

        monkeypatch.setattr(blobstore, "exists", lambda *_a: True)
        monkeypatch.setattr(blobstore, "read_head", lambda *_a: b"\x00" * 64)
        payload = migrate_checkpoints.run(_args(share, verify=True))

        assert payload.unreadable, "a truncated object passed as present"
        assert payload.rungs_already_there == 0


class TestDroppingTheShareCopy:
    """The other end of the migration. Every drop is gated on the container
    being able to supply what is about to be deleted, because after this the
    container is the only copy."""

    def _blob(self, monkeypatch, *, present=True, header=True):
        import src.shared.cloudtask.node.blobstore as blobstore

        monkeypatch.setattr(blobstore, "exists", lambda *_a: present)
        monkeypatch.setattr(
            blobstore,
            "read_head",
            lambda *_a: _header(b'{"arrays": [{"name": "regrets"}]}') if header else b"\x00" * 64,
        )

    def test_a_rung_the_container_holds_is_dropped(self, share, monkeypatch):
        self._blob(monkeypatch)
        payload = migrate_checkpoints.run(_args(share, drop_share=True, apply=True))

        assert payload.rungs_dropped == 1
        assert not (share / "archive" / "run-a" / "static-100.zarr").exists()

    def test_the_marker_is_kept(self, share, monkeypatch):
        """Zero bytes, and it is the share's index of what was published:
        `prune` reads markers for the ladder and `verify_published_rungs`
        builds from them. Deleting it makes a rung the container HOLDS
        unscoreable."""
        self._blob(monkeypatch)
        migrate_checkpoints.run(_args(share, drop_share=True, apply=True))

        assert (share / "archive" / "run-a" / archive.marker_for("static-100.zarr")).exists()

    def test_a_rung_the_container_lacks_is_kept(self, share, monkeypatch):
        self._blob(monkeypatch, present=False)
        payload = migrate_checkpoints.run(_args(share, drop_share=True, apply=True))

        assert payload.rungs_dropped == 0
        assert payload.kept == ["run-a/static-100.zarr: not in the container"]
        assert (share / "archive" / "run-a" / "static-100.zarr").is_dir(), "the only copy"

    def test_a_rung_whose_object_will_not_open_is_kept(self, share, monkeypatch):
        """Present is not readable. A truncated upload answers a HEAD, and
        this is the last moment the other copy still exists."""
        self._blob(monkeypatch, header=False)
        payload = migrate_checkpoints.run(_args(share, drop_share=True, apply=True))

        assert payload.rungs_dropped == 0
        assert (share / "archive" / "run-a" / "static-100.zarr").is_dir()

    def test_without_apply_it_deletes_nothing(self, share, monkeypatch):
        self._blob(monkeypatch)
        payload = migrate_checkpoints.run(_args(share, drop_share=True))

        assert payload.rungs_dropped == 1, "it still reports what it would drop"
        assert (share / "archive" / "run-a" / "static-100.zarr").is_dir()

    def test_an_unclaimed_directory_is_dropped_too(self, share, monkeypatch):
        """Dropping walks the SHARE, not the manifest: a directory no manifest
        names is duplicate weight as well, and the manifest cannot see it."""
        (share / "archive" / "run-a" / "static-999.zarr").mkdir()
        self._blob(monkeypatch)
        payload = migrate_checkpoints.run(_args(share, drop_share=True, apply=True))

        assert payload.rungs_dropped == 2
        assert not (share / "archive" / "run-a" / "static-999.zarr").exists()


class TestASpaceJoinedRunsValueStillFilters:
    """MEASURED: a shell that did not word-split `--runs $ids` handed the node
    ONE space-joined value. It matched no run, so 29 dispatched tasks each
    considered 0 runs, deleted nothing, and exited 0 -- a whole fan-out that
    reported success and did not happen.
    """

    def test_one_joined_value_selects_every_run_it_names(self, share, monkeypatch):
        (share / "archive" / "run-b").mkdir()
        import src.shared.cloudtask.node.blobstore as blobstore

        monkeypatch.setattr(blobstore, "exists", lambda *_a: False)
        payload = migrate_checkpoints.run(_args(share, verify=True, runs=["run-a run-b"]))

        assert payload.runs_considered == 2, "both ids in the joined value must be seen"

    def test_a_normal_list_is_unaffected(self, share, monkeypatch):
        import src.shared.cloudtask.node.blobstore as blobstore

        monkeypatch.setattr(blobstore, "exists", lambda *_a: False)
        payload = migrate_checkpoints.run(_args(share, verify=True, runs=["run-a"]))

        assert payload.runs_considered == 1


class TestDeletingARungIsParallel:
    def test_every_file_under_the_snapshot_goes(self, share, monkeypatch):
        """`shutil.rmtree` is serial and SMB is latency-bound: 169 deletes/sec
        measured, which is 443 rungs in a four-hour task. Whatever the pool,
        the tree must be gone."""
        nested = share / "archive" / "run-a" / "static-100.zarr" / "deep" / "nested"
        nested.mkdir(parents=True, exist_ok=True)
        for i in range(25):
            (nested / str(i)).write_text("chunk")
        import src.shared.cloudtask.node.blobstore as blobstore

        monkeypatch.setattr(blobstore, "exists", lambda *_a: True)
        monkeypatch.setattr(blobstore, "read_head", lambda *_a: _header(b'{"arrays": [{"n": 1}]}'))

        migrate_checkpoints.run(_args(share, drop_share=True, apply=True))

        assert not (share / "archive" / "run-a" / "static-100.zarr").exists()
