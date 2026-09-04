"""The sweep stages a rung locally before tarring it, and can prove it.

WHY THIS EXISTS. The staging helper was added in one commit and its call site
was never replaced -- `38 insertions(+)`, zero deletions. The function sat there
as dead code, the loop went on tarring straight off the share, and three sweeps
timed out having moved nothing. The commit was clean, 2,776 tests passed and the
whole gate was green, because nothing asserted WHERE the bytes were read from.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src.interfaces.commands import migrate_checkpoints
from src.shared.cloudtask.node import archive


@pytest.fixture
def share(tmp_path, monkeypatch):
    """A share holding one marked rung, and a node-local work directory."""
    root = tmp_path / "share" / "archive" / "run-a"
    rung = root / "static-100.zarr"
    (rung / "c").mkdir(parents=True)
    (rung / ".zarray").write_text("{}")
    (rung / "c" / "0").write_bytes(b"\x01" * 64)
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


class TestItUploadsFromStagingNotTheShare:
    def test_the_path_handed_to_put_rung_is_not_on_the_share(self, share, monkeypatch):
        """THE REGRESSION. Tarring straight off the share walks ~5,500 chunk
        files serially over SMB -- more than eight minutes for one rung, and
        three sweeps died proving it."""
        _payload, seen = _run(share, monkeypatch)
        assert seen, "nothing was uploaded at all"
        uploaded = seen[0]
        assert share not in uploaded.parents, f"tarred straight off the share: {uploaded}"

    def test_it_is_under_the_node_work_directory(self, share, monkeypatch, tmp_path):
        _payload, seen = _run(share, monkeypatch)
        assert (tmp_path / "work") in seen[0].parents

    def test_the_staged_copy_is_the_same_tree(self, share, monkeypatch):
        """Staging must reproduce the rung, or the object is wrong in a way
        nothing downstream can see until a load fails."""
        captured: dict[str, list[str]] = {}
        import src.shared.cloudtask.node.blobstore as blobstore

        monkeypatch.setattr(blobstore, "exists", lambda *_a: False)

        def _put(_s, _r, _n, path):
            captured["names"] = sorted(p.name for p in Path(path).rglob("*") if p.is_file())
            return 4096

        monkeypatch.setattr(blobstore, "put_rung", _put)
        migrate_checkpoints.run(
            argparse.Namespace(share=str(share), runs=None, limit=0, verify=False)
        )
        assert captured["names"] == [".zarray", "0"]

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
