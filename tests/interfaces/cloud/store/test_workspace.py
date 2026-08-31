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


class TestPullMetadata:
    def test_pulls_the_json_record(self, fake, tmp_path):
        workspace.pull_metadata(fake, "s", tmp_path)
        assert (tmp_path / "run-a" / "run.jsonl").is_file()
        assert (tmp_path / "run-a" / "evals" / "slug1.json").is_file()
        assert (tmp_path / "run-b" / "run.jsonl").is_file()

    def test_never_pulls_checkpoint_data(self, fake, tmp_path):
        """~540 MB of zarr chunks that no reading command opens."""
        workspace.pull_metadata(fake, "s", tmp_path)
        assert not (tmp_path / "run-a" / "static-1000.zarr").exists()

    def test_completion_markers_are_recreated_rather_than_downloaded(self, fake, tmp_path):
        """A marker's whole content is that it exists, so its name in the
        listing is the entire fact -- and it is what says whether a rung the
        manifest advertises can actually be scored."""
        workspace.pull_metadata(fake, "s", tmp_path)
        marker = tmp_path / "run-a" / ".complete-static-1000.zarr"
        assert marker.is_file()

    def test_one_run_pulls_only_that_run(self, fake, tmp_path):
        workspace.pull_metadata(fake, "s", tmp_path, run="run-a")
        assert (tmp_path / "run-a").is_dir()
        assert not (tmp_path / "run-b").exists()

    def test_an_unpublished_run_says_what_is_published(self, fake, tmp_path):
        with pytest.raises(CommandError, match="run-a"):
            workspace.pull_metadata(fake, "s", tmp_path, run="run-nope")

    def test_the_local_tree_mirrors_the_published_one(self, fake, tmp_path):
        """The readers are ordinary local-path code; the layout must match."""
        workspace.pull_metadata(fake, "s", tmp_path)
        loaded = json.loads((tmp_path / "run-a" / "run.jsonl").read_text())
        assert loaded["run_id"] == "run-a"


class TestSourceSeam:
    """There is only one source now: the published record."""

    def test_share_derives_the_index_rather_than_reading_a_shared_file(self, tmp_path):
        """A second writable file on a share with no atomic append is the
        contention the per-run records exist to remove."""
        from src.interfaces.commands import _base

        derived = _base.ledger_for(tmp_path)
        assert derived.parent == tmp_path, "derived inside the materialised tree"
        assert derived.is_file(), "rebuild_ledger ran"


class TestTheWalkIsPruned:
    """Filtering checkpoint data out AFTER the walk still paid for the walk.

    A run's .zarr snapshots hold thousands of chunk files and listing them is a
    round trip per directory -- measured at 167s to pull 146 small JSON files.
    """

    def test_it_never_lists_inside_a_snapshot_directory(self, fake, tmp_path, monkeypatch):
        listed: list[str] = []
        original = share.list_entries

        def spy(service, share_name, path):
            listed.append(path)
            return original(service, share_name, path)

        monkeypatch.setattr(share, "list_entries", spy)
        workspace.pull_metadata(fake, "s", tmp_path)

        assert not [p for p in listed if ".zarr" in p], f"descended into a snapshot: {listed}"


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


class TestIncrementalRefresh:
    """A published record never changes once written, so a refresh should fetch
    only what moved. The console rebuilds this tree every 45s; before this it
    re-downloaded all 4,251 immutable documents each time -- 6.4s of a 24.4s
    rebuild, and `_checkout` blocks every other request while it runs."""

    def test_a_refresh_against_an_unchanged_share_downloads_nothing(self, fake, tmp_path):
        first = tmp_path / "first"
        workspace.pull_metadata(fake, "s", first)
        assert fake.downloads, "the first pull must actually fetch"

        fake.downloads.clear()
        second = tmp_path / "second"
        fetched = workspace.pull_metadata(fake, "s", second, previous=first)

        assert fetched == 0
        assert fake.downloads == []
        # And the tree is COMPLETE, not merely cheap.
        assert (second / "run-a" / "evals" / "slug1.json").is_file()
        assert (second / "run-b" / "run.jsonl").is_file()

    def test_only_the_changed_document_is_refetched(self, fake, tmp_path):
        first = tmp_path / "first"
        workspace.pull_metadata(fake, "s", first)

        fake.etags["archive/run-a/evals/slug1.json"] = "v2"
        fake.downloads.clear()
        second = tmp_path / "second"
        fetched = workspace.pull_metadata(fake, "s", second, previous=first)

        assert fetched == 1
        assert fake.downloads == ["archive/run-a/evals/slug1.json"]

    def test_a_new_document_is_picked_up(self, fake, tmp_path):
        first = tmp_path / "first"
        workspace.pull_metadata(fake, "s", first)

        fake.files["archive/run-b/evals/slug9.json"] = json.dumps({"run_id": "run-b"})
        second = tmp_path / "second"
        workspace.pull_metadata(fake, "s", second, previous=first)

        assert (second / "run-b" / "evals" / "slug9.json").is_file()

    def test_a_previous_tree_with_no_manifest_is_ignored_not_trusted(self, fake, tmp_path):
        """A tree from before this existed has no etags. Reusing its files on
        name alone would serve whatever it happened to hold."""
        stale = tmp_path / "stale"
        (stale / "run-a" / "evals").mkdir(parents=True)
        (stale / "run-a" / "evals" / "slug1.json").write_text("STALE")

        fresh = tmp_path / "fresh"
        workspace.pull_metadata(fake, "s", fresh, previous=stale)

        assert (fresh / "run-a" / "evals" / "slug1.json").read_text() != "STALE"
