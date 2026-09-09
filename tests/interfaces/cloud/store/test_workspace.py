"""Answering a question against the published record, without keeping a copy."""

from __future__ import annotations

import json
import threading
from typing import TYPE_CHECKING

import pytest

from src.interfaces.cloud.store import share, workspace
from src.interfaces.errors import CommandError

if TYPE_CHECKING:
    from pathlib import Path


def _mark(root: Path, _previous: Path | None) -> None:
    """A build that writes one marker file.

    A named function rather than a lambda because `write_text` returns a
    character count and `build` is declared `-> None`; a lambda body is an
    expression, so it cannot help returning it.
    """
    (root / "marker").write_text("x")


class _FakeShare:
    """A share as a dict of path -> bytes.

    Stands in for ShareServiceClient at the two seams workspace uses --
    ``list_entries``/``walk_files`` to discover and ``download_file`` to pull --
    so the materialiser is tested without an Azure account.
    """

    def __init__(self, files: dict[str, str]):
        self.files = files
        self.written: dict[str, str] = {}
        self.etags: dict[str, str] = {}
        self.downloads: list[str] = []


@pytest.fixture
def fake(monkeypatch):
    store = _FakeShare(
        {
            "archive/run-a/run.jsonl": json.dumps({"event": "created", "run_id": "run-a"}) + "\n",
            "archive/run-a/STATIC_CHECKPOINT.json": json.dumps({"iteration": 1000}),
            "archive/run-a/evals/slug1.json": json.dumps({"run_id": "run-a"}),
            "archive/run-a/static-1000.zarr/0.0": "BULK",
            "archive/run-a/.complete-static-1000.zarr": "",
            "archive/run-b/run.jsonl": json.dumps({"event": "created", "run_id": "run-b"}) + "\n",
        }
    )

    def walk_files(service, share_name, path, *, skip_dir=None):
        prefix = f"{path}/"
        found = []
        for p in service.files:
            if not p.startswith(prefix):
                continue
            parts = p[len(prefix) :].split("/")
            if skip_dir is not None and any(skip_dir(part) for part in parts[:-1]):
                continue
            found.append((p, service.etags.get(p, "v1")))
        return found

    def list_entries(service, share_name, path, *, etags=False):
        names = set()
        prefix = f"{path}/"
        for p in service.files:
            if p.startswith(prefix):
                rest = p[len(prefix) :]
                names.add((rest.split("/")[0], "/" in rest))
        return [share.ShareEntry(name=n, is_directory=d, size=0) for n, d in sorted(names)]

    def download_file(service, share_name, path, destination):
        service.downloads.append(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(service.files[path])

    def read_text(service, share_name, path):
        return service.files.get(path)

    def write_text(service, share_name, path, body):
        service.written[path] = body
        service.files[path] = body

    monkeypatch.setattr(share, "walk_files", walk_files)
    monkeypatch.setattr(share, "list_entries", list_entries)
    monkeypatch.setattr(share, "download_file", download_file)
    monkeypatch.setattr(share, "read_text", read_text)
    monkeypatch.setattr(share, "write_text", write_text)
    return store


RECORD = {
    "run-a": {
        "rungs": {"static-1000.ckpt.zst"},
        "manifest": b'{"zarr": "static-1000.ckpt.zst", "iteration": 1000, "retained": []}',
    },
    "run-b": {"rungs": set(), "manifest": b'{"zarr": "", "iteration": 0, "retained": []}'},
}


class TestPullMetadata:
    """The tree a reader walks, built from what the container holds."""

    def test_it_writes_each_run_s_manifest(self, tmp_path):
        assert workspace.pull_metadata(tmp_path, RECORD) == 2
        assert (tmp_path / "run-a" / "STATIC_CHECKPOINT.json").is_file()
        assert (tmp_path / "run-b" / "STATIC_CHECKPOINT.json").is_file()

    def test_markers_come_from_the_rungs_the_container_holds(self, tmp_path):
        """A marker's whole content is that it exists, and what it says is
        whether a rung the manifest advertises can actually be fetched. One
        listing answers that for every run, and it answers from the bytes
        rather than from a second claim beside them."""
        workspace.pull_metadata(tmp_path, RECORD)

        assert (tmp_path / "run-a" / ".complete-static-1000.ckpt.zst").is_file()
        assert not list((tmp_path / "run-b").glob(".complete-*"))

    def test_one_run_pulls_only_that_run(self, tmp_path):
        workspace.pull_metadata(tmp_path, RECORD, run="run-a")
        assert (tmp_path / "run-a").is_dir()
        assert not (tmp_path / "run-b").exists()

    def test_an_unpublished_run_says_what_is_published(self, tmp_path):
        with pytest.raises(CommandError, match="run-a"):
            workspace.pull_metadata(tmp_path, RECORD, run="run-nope")

    def test_an_ambiguous_fragment_is_refused(self, tmp_path):
        with pytest.raises(CommandError):
            workspace.pull_metadata(tmp_path, RECORD, run="run-")

    def test_the_local_tree_mirrors_the_published_one(self, tmp_path):
        """The readers are ordinary local-path code; the layout must match."""
        workspace.pull_metadata(tmp_path, RECORD)
        loaded = json.loads((tmp_path / "run-a" / "STATIC_CHECKPOINT.json").read_text())
        assert loaded["zarr"] == "static-1000.ckpt.zst"

    def test_a_run_with_no_manifest_still_gets_its_directory(self, tmp_path):
        """A task that died before its first checkpoint has rungs and no
        manifest, and a reader must see the run rather than nothing."""
        record = {"run-c": {"rungs": {"static-5.ckpt.zst"}, "manifest": None}}
        assert workspace.pull_metadata(tmp_path, record) == 0
        assert (tmp_path / "run-c" / ".complete-static-5.ckpt.zst").is_file()


class TestSourceSeam:
    """There is only one source now: the published record."""

    def test_share_derives_the_index_rather_than_reading_a_shared_file(self, tmp_path):
        """A second writable file on a share with no atomic append is the
        contention the per-run records exist to remove."""
        from src.interfaces.commands import _base

        derived = _base.ledger_for(tmp_path)
        assert derived.parent == tmp_path, "derived inside the materialised tree"
        assert derived.is_file(), "rebuild_ledger ran"


class TestSharedTrees:
    """One materialised record, shared -- and deleted only when nobody holds it.

    The measured defect: five endpoints answering questions about the same
    record each pulled their own copy. `/api/runs` and `/api/evals` pulled the
    whole thing at 12.4s each, and a run's three detail panels pulled that run
    three times over. It is ~120 round trips for 0.23 MB, so paying it once is
    nearly the whole fix.
    """

    def test_a_second_reader_inside_the_ttl_does_not_rebuild(self):
        builds = []

        def build(root, _previous):
            builds.append(root)
            (root / "marker").write_text("x")

        trees = workspace.SharedTrees(ttl=60.0)
        with trees.acquire("record", build) as first, trees.acquire("record", build) as second:
            assert first == second
        assert len(builds) == 1
        trees.close()

    def test_concurrent_misses_build_once(self):
        """Not "a duplicated read": eight panels mounting together is eight
        simultaneous sweeps of the share, which is how throttling is met."""
        started, release = threading.Event(), threading.Event()
        builds = []

        def build(root, _previous):
            builds.append(root)
            started.set()
            release.wait(timeout=2)

        trees = workspace.SharedTrees(ttl=60.0)
        seen: list = []

        def read():
            with trees.acquire("record", build) as root:
                seen.append(root)

        threads = [threading.Thread(target=read) for _ in range(6)]
        threads[0].start()
        assert started.wait(timeout=2)
        for thread in threads[1:]:
            thread.start()
        release.set()
        for thread in threads:
            thread.join(timeout=2)

        assert len(builds) == 1
        assert len(set(seen)) == 1
        trees.close()

    def test_a_reader_during_a_rebuild_is_served_stale_rather_than_blocked(self):
        """The measured defect this exists for.

        Discovery alone is ~5.4s against the share, so a reader that WAITS for
        an in-flight rebuild pays a multi-second sweep for freshness it did not
        ask for. The builder already holds the expired tree, so handing it out
        costs one refcount and no round trips.
        """
        started, release = threading.Event(), threading.Event()

        def slow_build(root, _previous):
            (root / "marker").write_text("x")
            started.set()
            release.wait(timeout=2)

        trees = workspace.SharedTrees(ttl=0.0, stale_grace=60.0)
        with trees.acquire("record", _mark) as first:
            pass

        builder = threading.Thread(target=lambda: _read(trees, slow_build))
        builder.start()
        assert started.wait(timeout=2), "the rebuild never began"

        # The rebuild is in flight and will not finish until `release`. A reader
        # arriving now must come back with the OLD tree, not hang on the new one.
        with trees.acquire("record", _never_called) as during:
            assert during == first, "served a different tree than the expired one"
        release.set()
        builder.join(timeout=2)
        trees.close()

    def test_a_stale_tree_past_the_grace_blocks_instead_of_answering(self):
        """A build that keeps failing must be REPORTED, not answered from an
        ever-older tree. Same bound as the payload cache one layer up."""
        started, release = threading.Event(), threading.Event()

        def slow_build(root, _previous):
            (root / "marker").write_text("x")
            started.set()
            release.wait(timeout=2)

        # ttl and grace both zero: the previous tree is already past its welcome.
        trees = workspace.SharedTrees(ttl=0.0, stale_grace=0.0)
        with trees.acquire("record", _mark):
            pass

        builder = threading.Thread(target=lambda: _read(trees, slow_build))
        builder.start()
        assert started.wait(timeout=2)

        waited = threading.Event()

        def late_reader():
            with trees.acquire("record", _mark):
                waited.set()

        thread = threading.Thread(target=late_reader)
        thread.start()
        assert not waited.wait(timeout=0.3), "answered from a tree past its grace"
        release.set()
        thread.join(timeout=2)
        builder.join(timeout=2)
        assert waited.is_set()
        trees.close()

    def test_expiry_does_not_delete_a_tree_still_being_read(self):
        """The hazard refcounting exists for: expiry alone pulls the directory
        out from under a reader mid-answer."""
        trees = workspace.SharedTrees(ttl=0.0)  # every lookup is a miss
        with trees.acquire("record", _mark) as held:
            with trees.acquire("record", _mark) as fresh:
                assert fresh != held
            assert (held / "marker").is_file(), "the first reader's tree was deleted under it"
        assert not held.exists(), "a released tree was never cleaned up"
        trees.close()

    def test_close_removes_what_nobody_holds(self):
        trees = workspace.SharedTrees(ttl=60.0)
        with trees.acquire("record", _mark) as root:
            pass
        assert root.is_dir()
        trees.close()
        assert not root.exists()

    def test_a_failed_build_leaves_nothing_and_frees_the_key(self):
        trees = workspace.SharedTrees(ttl=60.0)

        def explode(root, _previous):
            raise RuntimeError("Azure said no")

        with (
            pytest.raises(RuntimeError, match="Azure said no"),
            trees.acquire("record", explode),
        ):
            pass
        with trees.acquire("record", _mark) as root:
            assert (root / "marker").is_file()
        trees.close()

    def test_a_refresh_is_handed_the_expired_tree_and_may_read_it(self):
        """The incremental sync's contract: the previous tree is still there,
        intact, for the whole of the build that replaces it."""
        trees = workspace.SharedTrees(ttl=0.0)  # every lookup is a miss
        handed: list = []

        def build(root, previous):
            handed.append(previous)
            if previous is not None:
                assert (previous / "marker").read_text() == "x", "the old tree was gone"
            (root / "marker").write_text("x")

        with trees.acquire("record", build) as first:
            pass
        with trees.acquire("record", build) as second:
            pass
        assert handed == [None, first]
        assert second != first
        assert not first.exists(), "the expired tree outlived its replacement"
        trees.close()

    def test_a_failed_refresh_keeps_the_expired_tree_for_the_next_attempt(self):
        trees = workspace.SharedTrees(ttl=0.0)
        with trees.acquire("record", _mark) as first:
            pass

        def explode(_root, _previous):
            raise RuntimeError("Azure said no")

        with (
            pytest.raises(RuntimeError, match="Azure said no"),
            trees.acquire("record", explode),
        ):
            pass
        handed: list = []
        with trees.acquire("record", lambda _root, previous: handed.append(previous)):
            pass
        assert handed == [first], "the retry was not offered the tree it could have synced from"
        trees.close()

    def test_two_caches_nest_rather_than_refusing(self):
        """`create_app` promises two applications can live in one process, and
        each brings its own lifespan. Refusing the second would surface as a
        RuntimeError raised from inside one."""
        with workspace.shared_record_cache(ttl=60.0) as outer:
            with workspace.shared_record_cache(ttl=60.0) as inner:
                assert inner is not outer
                assert workspace.active_cache() is inner
            assert workspace.active_cache() is outer
        assert workspace.active_cache() is None

    def test_sharing_is_off_unless_asked_for(self):
        """The command line must keep answering against the record as it is NOW
        -- a run published thirty seconds ago must not be invisible to
        `promote`."""
        assert workspace.active_cache() is None
        with workspace.shared_record_cache(ttl=60.0) as cache:
            assert workspace.active_cache() is cache
        assert workspace.active_cache() is None


def _read(trees, build):
    with trees.acquire("record", build):
        pass


def _never_called(root, _previous):
    raise AssertionError("a reader served from the stale tree must not build")
