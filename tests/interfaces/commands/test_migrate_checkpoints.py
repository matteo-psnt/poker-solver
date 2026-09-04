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
        share=str(share), runs=None, limit=0, verify=False, recover_tars=False, **over
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
                share=str(share), runs=None, limit=0, verify=False, recover_tars=False
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
    base = {"share": str(share), "runs": None, "limit": 0, "verify": False, "recover_tars": False}
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

    def test_a_rung_with_no_share_bytes_but_a_tar_is_recoverable(self, stranded, monkeypatch):
        import src.shared.cloudtask.node.blobstore as blobstore

        monkeypatch.setattr(blobstore, "exists", lambda _s, _r, name: name.endswith(".tar"))
        payload = migrate_checkpoints.run(_args(stranded, verify=True))

        assert payload.recoverable == ["run-b/static-100.zarr"]
        assert payload.missing == [], "it is not on the share; a sweep cannot move it"

    def test_a_rung_with_no_bytes_at_all_is_reported_lost(self, stranded, monkeypatch):
        import src.shared.cloudtask.node.blobstore as blobstore

        monkeypatch.setattr(blobstore, "exists", lambda *_a: False)
        payload = migrate_checkpoints.run(_args(stranded, verify=True))

        assert payload.lost == ["run-b/static-100.zarr"]

    def test_an_outstanding_rung_refuses_the_deletion(self, stranded, monkeypatch, capsys):
        import src.shared.cloudtask.node.blobstore as blobstore

        monkeypatch.setattr(blobstore, "exists", lambda *_a: False)
        migrate_checkpoints.render(migrate_checkpoints.run(_args(stranded, verify=True)))

        assert "DO NOT DELETE the share" in capsys.readouterr().out

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


class TestRecoveringARungThatOnlyExistsAsATar:
    def test_the_arrays_survive_the_tar_and_come_back_identical(
        self, stranded, monkeypatch, tmp_path
    ):
        """313 rungs across six production runs have no other copy, so this
        crossing is the only thing standing between them and being lost."""
        import tarfile

        import zarr

        import src.shared.cloudtask.node.blobstore as blobstore

        source = tmp_path / "src" / "static-100.zarr"
        group = zarr.open(zarr.DirectoryStore(str(source)), mode="w")
        for name, array in ARRAYS.items():
            group.create_dataset(name, data=array, dtype=array.dtype)
        group.attrs["fingerprint"] = "cafe"
        tar_path = tmp_path / "rung.tar"
        with tarfile.open(tar_path, "w") as bundle:
            bundle.add(source, arcname="static-100.zarr")

        def _get(_s, _r, name, destination):
            Path(destination).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(tar_path, Path(destination) / name)
            return True

        kept: dict[str, Path] = {}

        def _put(_s, _r, _n, path):
            kept["at"] = tmp_path / "kept.ckpt.zst"
            shutil.copyfile(path, kept["at"])
            return Path(path).stat().st_size

        monkeypatch.setattr(blobstore, "exists", lambda _s, _r, name: name.endswith(".tar"))
        monkeypatch.setattr(blobstore, "get_rung", _get)
        monkeypatch.setattr(blobstore, "put_rung", _put)

        payload = migrate_checkpoints.run(_args(stranded, recover_tars=True))

        assert payload.rungs_uploaded == 1
        arrays, attrs = snapshot_format.read_snapshot(kept["at"])
        for name, expected in ARRAYS.items():
            assert np.array_equal(arrays[name], expected), name
        assert attrs["fingerprint"] == "cafe", "provenance must survive the crossing"
