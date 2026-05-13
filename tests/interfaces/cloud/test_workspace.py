"""Answering a question against the published record, without keeping a copy."""

from __future__ import annotations

import json
import threading

import pytest

from src.interfaces.cloud import share, workspace
from src.interfaces.errors import CommandError


class _FakeShare:
    """A share as a dict of path -> bytes.

    Stands in for ShareServiceClient at the two seams workspace uses --
    ``list_entries``/``walk_files`` to discover and ``download_file`` to pull --
    so the materialiser is tested without an Azure account.
    """

    def __init__(self, files: dict[str, str]):
        self.files = files
        self.written: dict[str, str] = {}


@pytest.fixture
def fake(monkeypatch):
    store = _FakeShare(
        {
            "archive/run-a/run.jsonl": json.dumps({"event": "created", "run_id": "run-a"}) + "\n",
            "archive/run-a/STATIC_CHECKPOINT.json": json.dumps({"iteration": 1000}),
            "archive/run-a/evals/slug1.json": json.dumps({"run_id": "run-a"}),
            "archive/run-a/static-1000.zarr/0.0": "BULK",
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
            found.append(p)
        return found

    def list_entries(service, share_name, path):
        names = set()
        prefix = f"{path}/"
        for p in service.files:
            if p.startswith(prefix):
                rest = p[len(prefix) :]
                names.add((rest.split("/")[0], "/" in rest))
        return [share.ShareEntry(name=n, is_directory=d, size=0) for n, d in sorted(names)]

    def download_file(service, share_name, path, destination):
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
        assert not list(tmp_path.rglob("*.zarr*"))

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


class TestBaseline:
    def test_round_trips_through_the_share(self, fake):
        workspace.write_baseline(fake, "s", json.dumps({"run_id": "run-a"}))
        body = workspace.read_baseline(fake, "s")
        assert body is not None
        assert json.loads(body)["run_id"] == "run-a"

    def test_absent_baseline_reads_as_none(self, fake):
        assert workspace.read_baseline(fake, "s") is None

    def test_it_lands_at_the_share_root_beside_the_archive(self, fake):
        workspace.write_baseline(fake, "s", "{}")
        assert workspace.BASELINE_NAME in fake.written
        assert "/" not in workspace.BASELINE_NAME


class TestSourceSeam:
    """There is only one source now: the published record."""

    def test_share_derives_the_index_rather_than_reading_a_shared_file(self, tmp_path):
        """A second writable file on a share with no atomic append is the
        contention the per-run records exist to remove."""
        import argparse

        from src.interfaces.commands import _base

        args = argparse.Namespace(source="share", runs_dir="unused", ledger="unused.jsonl")
        derived = _base.ledger_for(args, tmp_path)
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

        def build(root):
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

        def build(root):
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
        with trees.acquire("record", lambda root: (root / "marker").write_text("x")) as held:
            with trees.acquire("record", lambda root: (root / "marker").write_text("x")) as fresh:
                assert fresh != held
            assert (held / "marker").is_file(), "the first reader's tree was deleted under it"
        assert not held.exists(), "a released tree was never cleaned up"
        trees.close()

    def test_close_removes_what_nobody_holds(self):
        trees = workspace.SharedTrees(ttl=60.0)
        with trees.acquire("record", lambda root: (root / "marker").write_text("x")) as root:
            pass
        assert root.is_dir()
        trees.close()
        assert not root.exists()

    def test_a_failed_build_leaves_nothing_and_frees_the_key(self):
        trees = workspace.SharedTrees(ttl=60.0)

        def explode(root):
            raise RuntimeError("Azure said no")

        with pytest.raises(RuntimeError), trees.acquire("record", explode):
            pass
        with trees.acquire("record", lambda root: (root / "marker").write_text("x")) as root:
            assert (root / "marker").is_file()
        trees.close()

    def test_sharing_is_off_unless_asked_for(self):
        """The command line must keep answering against the record as it is NOW
        -- a run published thirty seconds ago must not be invisible to
        `promote`."""
        assert workspace.active_cache() is None
        with workspace.shared_record_cache(ttl=60.0) as cache:
            assert workspace.active_cache() is cache
        assert workspace.active_cache() is None
