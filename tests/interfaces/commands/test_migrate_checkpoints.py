"""The sweep stages a rung locally before tarring it, and can prove it.

WHY THIS EXISTS. The staging helper was added in one commit and its call site
was never replaced -- `38 insertions(+)`, zero deletions. The function sat there
as dead code, the loop went on tarring straight off the share, and three sweeps
timed out having moved nothing. The commit was clean, 2,776 tests passed and the
whole gate was green, because nothing asserted WHERE the bytes were read from.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pytest

from src.engine.solver.storage import snapshot_format
from src.interfaces.commands import migrate_checkpoints
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
    args = argparse.Namespace(share=str(share), runs=None, limit=0, verify=False, **over)
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
            argparse.Namespace(share=str(share), runs=None, limit=0, verify=False)
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
