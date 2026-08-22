"""The legs/ sync fetches what moved, not what the share holds.

legs/ had grown to 2,252 files by 08-22, and every refresh re-downloaded all of
them -- 33s from the laptop -- to show the ten most recent. A record never
changes once written, so the version key is the file's etag and the cost of a
refresh is the listing plus the few files that are new or rewritten.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.interfaces.cloud.store import share
from src.interfaces.commands import tasks

if TYPE_CHECKING:
    from pathlib import Path

    import pytest


class FakeShare:
    """A flat legs/ directory with versions, counting what is downloaded."""

    def __init__(self, files: dict[str, tuple[str, str]]):
        self.files = files  # name -> (etag, body)
        self.downloaded: list[str] = []

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _list(_service, _share, path, *, etags=False):
            assert path == "legs"
            assert etags, "a sync that does not ask for etags cannot tell what moved"
            return [
                share.ShareEntry(name=name, is_directory=False, size=len(body), etag=etag)
                for name, (etag, body) in self.files.items()
            ]

        def _download(_service, _share, path, destination: Path):
            name = path.removeprefix("legs/")
            self.downloaded.append(name)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(self.files[name][1])

        monkeypatch.setattr(share, "list_entries", _list)
        monkeypatch.setattr(share, "download_file", _download)


def _legs(root: Path) -> dict[str, str]:
    return {path.name: path.read_text() for path in (root / "legs").iterdir()}


class TestIncrementalSync:
    def test_a_first_materialisation_fetches_everything(self, tmp_path, monkeypatch):
        fake = FakeShare({"a.start.json": ("e1", "A"), "b.start.json": ("e2", "B")})
        fake.install(monkeypatch)
        fetched = tasks.download_tasks(None, "share", tmp_path / "one")
        assert fetched == 2
        assert _legs(tmp_path / "one") == {"a.start.json": "A", "b.start.json": "B"}

    def test_a_refresh_fetches_only_what_changed(self, tmp_path, monkeypatch):
        fake = FakeShare({"a.start.json": ("e1", "A"), "b.progress.json": ("e2", "B")})
        fake.install(monkeypatch)
        tasks.download_tasks(None, "share", tmp_path / "one")
        fake.downloaded.clear()

        # A progress file rewritten, a record added, a record unchanged.
        fake.files["b.progress.json"] = ("e3", "B2")
        fake.files["c.exit.json"] = ("e4", "C")
        fetched = tasks.download_tasks(None, "share", tmp_path / "two", tmp_path / "one")

        assert fetched == 2
        assert sorted(fake.downloaded) == ["b.progress.json", "c.exit.json"]
        assert _legs(tmp_path / "two") == {
            "a.start.json": "A",
            "b.progress.json": "B2",
            "c.exit.json": "C",
        }

    def test_a_file_gone_from_the_share_is_gone_from_the_refresh(self, tmp_path, monkeypatch):
        """Compaction folds records into a bundle and deletes them; the sync
        must not resurrect them from the previous tree."""
        fake = FakeShare({"a.start.json": ("e1", "A"), "old.exit.json": ("e2", "O")})
        fake.install(monkeypatch)
        tasks.download_tasks(None, "share", tmp_path / "one")
        del fake.files["old.exit.json"]
        fake.files["legs-2026.bundle.json"] = ("e9", "{}")

        tasks.download_tasks(None, "share", tmp_path / "two", tmp_path / "one")

        assert set(_legs(tmp_path / "two")) == {"a.start.json", "legs-2026.bundle.json"}

    def test_the_version_manifest_stays_out_of_the_legs_directory(self, tmp_path, monkeypatch):
        """`read_documents` globs `*.json` under legs/; a manifest inside it
        would be read as a record."""
        FakeShare({"a.start.json": ("e1", "A")}).install(monkeypatch)
        tasks.download_tasks(None, "share", tmp_path / "one")
        assert (tmp_path / "one" / tasks._ETAGS_NAME).is_file()
        assert set(_legs(tmp_path / "one")) == {"a.start.json"}

    def test_a_previous_tree_without_a_manifest_is_fetched_in_full(self, tmp_path, monkeypatch):
        fake = FakeShare({"a.start.json": ("e1", "A")})
        fake.install(monkeypatch)
        stale = tmp_path / "stale"
        (stale / "legs").mkdir(parents=True)
        (stale / "legs" / "a.start.json").write_text("older than the manifest")
        tasks.download_tasks(None, "share", tmp_path / "two", stale)
        assert fake.downloaded == ["a.start.json"]
        assert _legs(tmp_path / "two") == {"a.start.json": "A"}
