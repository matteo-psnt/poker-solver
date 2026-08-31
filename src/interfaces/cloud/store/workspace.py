"""Answering a question about the record without keeping a copy of it.

Reading materialises the published JSON into a temporary directory, answers the
question there, and throws it away -- so a question is answerable on any
machine, rather than only on the one that last synced. Two boxes cannot hold
different answers and a fresh checkout is not blind.

Unless a :func:`shared_record_cache` is in force, in which case one tree answers
for every reader that arrives inside its lifetime. That is a server's concern
and nothing else's; see the note above it.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from src.interfaces import run_names
from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.store import share
from src.interfaces.errors import CommandError
from src.shared.cloudtask.node import archive

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from azure.storage.fileshare import ShareServiceClient


# `<op>_result.json` -- train, evaluate, resume, train-static. The writer was
# deleted with the clobbering result file it produced; nothing reads them, and
# they are still on the share as history. `fetch` still syncs them, because that
# is a copy of the record; this is a question being asked of it.
DEAD_SUFFIX = "_result.json"

# Sized by latency, not bandwidth: 105 metadata files took 7.5s at 16 threads,
# 4.6s at 32, 3.5s at 64. Threads blocked on a socket cost almost nothing.
_PARALLEL_DOWNLOADS = 64

# The etag manifest an incremental refresh compares against, written at the ROOT
# of a materialised tree. Every reader of that tree filters to directories, so a
# file beside the run directories is invisible to them.
_ETAGS_NAME = "records.etags"


def _is_snapshot_dir(name: str) -> bool:
    """A checkpoint directory, never worth descending into to ask a question.

    Delegates to :func:`share.is_snapshot_dir` so the walk and the per-path
    filter below cannot disagree about what a snapshot is -- when they did, a
    directory was descended into and then discarded file by file.
    """
    return share.is_snapshot_dir(name)


def pull_metadata(
    service: ShareServiceClient,
    share_name: str,
    destination: Path,
    *,
    run: str | None = None,
    previous: Path | None = None,
) -> int:
    """Materialise the published JSON record into ``destination``.

    Returns how many files were FETCHED, not how many the tree holds: with
    ``previous`` -- a tree this function built earlier -- an unchanged etag is
    hard-linked across instead of downloaded.
    """
    published = [
        entry.name
        for entry in share.list_entries(service, share_name, share.ARCHIVE_DIR)
        if entry.is_directory
    ]
    if run is not None:
        # Resolved HERE, against the share's own listing, because this decides
        # what gets downloaded -- a fragment rejected at this point never
        # reaches `resolve_run_dir`, and the reader would refuse a run that
        # exists. Same rule on both sides: `src.interfaces.run_names`.
        matches = run_names.matching(run, published)
        if len(matches) > 1:
            raise CommandError(run_names.ambiguous_message(run, matches))
        if not matches:
            raise CommandError(
                f"'{run}' is not published. Published runs: {', '.join(published) or '(none)'}"
            )
        published = matches

    # The WALK, not only the downloads. Each run is an independent traversal of
    # directory listings, and a listing is a round trip like any other: 18 runs
    # walked one after another was 12.4s of the ~20s a `--source share` read
    # took, before a single file had been fetched.
    def _walk(name: str) -> tuple[list[tuple[str, Path, str | None]], list[Path]]:
        found: list[tuple[str, Path, str | None]] = []
        markers: list[Path] = []
        for remote, etag in share.walk_files(
            service,
            share_name,
            f"{share.ARCHIVE_DIR}/{name}",
            skip_dir=_is_snapshot_dir,
        ):
            relative = remote[len(f"{share.ARCHIVE_DIR}/") :]
            leaf = Path(relative).name
            # RECREATED, never downloaded: a completion marker's whole content
            # is that it exists, so its name in this listing is the entire fact
            # and fetching it would be a round trip per rung. Without them the
            # local tree cannot tell a rung the share holds from one a manifest
            # merely names -- which is the gap `runinfo` reports.
            if leaf.startswith(archive.MARKER_PREFIX):
                markers.append(destination / relative)
                continue
            if share.is_snapshot_path(relative) or not share.is_metadata(leaf):
                continue
            if leaf.endswith(DEAD_SUFFIX):
                continue
            found.append((remote, destination / relative, etag))
        return found, markers

    with ThreadPoolExecutor(max_workers=min(_PARALLEL_DOWNLOADS, len(published) or 1)) as pool:
        walked = list(pool.map(_walk, published))
    wanted = [entry for batch, _ in walked for entry in batch]
    for _, markers in walked:
        for marker in markers:
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.touch()

    # INCREMENTAL against `previous`, on the same argument `download_tasks` uses
    # for legs/: a published record never changes once written, so an unchanged
    # etag means the bytes are already on disk and a hard link is the whole
    # refresh. Before this, a 45s console TTL re-fetched all 4,251 immutable
    # documents every time -- 6.4s of the 24.4s rebuild, paid every 45 seconds.
    known = _etags(previous)
    fetch: list[tuple[str, Path]] = []
    for remote, local, etag in wanted:
        relative = remote[len(f"{share.ARCHIVE_DIR}/") :]
        held = previous / relative if previous is not None else None
        if held is not None and etag is not None and known.get(relative) == etag and held.is_file():
            local.parent.mkdir(parents=True, exist_ok=True)
            _link(held, local)
        else:
            fetch.append((remote, local))

    # One round trip per file, and a run's eval documents now carry their full
    # sample vectors -- so this is latency-bound on a link where latency is the
    # whole cost. The downloads are independent and `download_file` builds its
    # own file client, so they overlap.
    if fetch:
        with ThreadPoolExecutor(max_workers=_PARALLEL_DOWNLOADS) as pool:
            futures = [
                pool.submit(share.download_file, service, share_name, remote, local)
                for remote, local in fetch
            ]
            for future in futures:
                future.result()

    (destination / _ETAGS_NAME).write_text(
        "".join(
            f"{etag or ''}\t{remote[len(f'{share.ARCHIVE_DIR}/') :]}\n"
            for remote, _, etag in wanted
        )
    )
    return len(fetch)


def resolve_published_run(run: str) -> str:
    """The full published run id for ``run``, which may be a fragment.

    Readers resolve a fragment locally (``resolve_run_dir``); DISPATCH has to
    resolve too, because the id is sent to a node and the node has no fragment
    matcher. Unresolved, `score --run 15261` cost a snapshot upload, a node
    allocation and three retries before failing "no such run on the share".
    Resolved against the share's own listing, so both sides answer from one
    source -- the rule already stated in :func:`pull_metadata`.
    """
    config = CloudConfig.load()
    service = share.share_client(config)
    published = [
        entry.name
        for entry in share.list_entries(service, config.share_name, share.ARCHIVE_DIR)
        if entry.is_directory
    ]
    matches = run_names.matching(run, published)
    if len(matches) > 1:
        raise CommandError(run_names.ambiguous_message(run, matches))
    if not matches:
        raise CommandError(
            f"'{run}' is not published. Published runs: {', '.join(published) or '(none)'}"
        )
    return matches[0]


def verify_published_rungs(run_id: str, rungs: Sequence[str]) -> None:
    """Refuse rungs the SHARE does not actually hold, before anything is dispatched.

    Checked against the share's own listing rather than the run's manifest,
    because the two disagree: pruning removes a snapshot without rewriting the
    manifest that advertises it, so `runinfo` offers rungs that
    `fetch_for_evaluation` then cannot find. Unverified, each such rung cost a
    snapshot upload, a node allocation and a `uv sync` before dying on "the
    manifest names static-N.zarr but it is not on the share" -- ~26 tasks in the
    2026-08-23/24 window.

    An empty rung means "the latest checkpoint", which the ladder cannot name in
    advance and the node resolves itself, so it is not checked here.
    """
    wanted = [rung for rung in rungs if rung]
    if not wanted:
        return
    config = CloudConfig.load()
    service = share.share_client(config)
    entries = share.list_entries(service, config.share_name, f"{share.ARCHIVE_DIR}/{run_id}")
    names = {entry.name for entry in entries}
    available = sorted(
        name.removeprefix("static-").removesuffix(".zarr")
        for name in names
        if name.startswith("static-") and archive.marker_for(name) in names
    )
    missing = [
        rung
        for rung in wanted
        if f"static-{rung}.zarr" not in names
        or archive.marker_for(f"static-{rung}.zarr") not in names
    ]
    if missing:
        raise CommandError(
            f"{run_id} has no published, complete checkpoint for: {', '.join(missing)}.\n"
            f"  On the share: {', '.join(available) or '(none)'}\n"
            "A rung the manifest advertises can still have been pruned -- this checks "
            "the share itself, so the mismatch surfaces here instead of on a node."
        )


@dataclass
class _Tree:
    """One materialised subtree, and who is still reading it."""

    path: Path
    born: float
    holders: int = 0
    retired: bool = False


@dataclass
class SharedTrees:
    """Materialised subtrees, shared between readers for ``ttl`` seconds.

    Exists for one caller: a server, where several endpoints answer questions
    about the SAME record within a second of each other. Measured before this,
    per browser refresh: `/api/runs` and `/api/evals` pulled the whole record
    (12.4s each) and `/api/runs/{id}`'s three panels pulled one run three times
    over -- the same few hundred kilobytes, five times, because a context
    manager that deletes its tree on exit cannot share it with the next caller.

    Two properties do the work:

    single-flight
        Concurrent misses on a key WAIT for the first build rather than each
        starting one. A page mount fires eight queries at once; without this it
        is eight simultaneous sweeps of the share, which is both slow and the
        most likely way to meet Azure Files throttling.
    refcounting
        A tree is deleted when it expires AND nobody holds it. Expiry alone
        would pull the directory out from under a reader mid-answer; never
        deleting would leak one tree per refresh for the life of the server.
    """

    ttl: float
    _lock: threading.Condition = field(default_factory=threading.Condition, repr=False)
    _trees: dict[str, _Tree] = field(default_factory=dict, repr=False)
    _building: set[str] = field(default_factory=set, repr=False)

    @contextmanager
    def acquire(self, key: str, build: Callable[[Path, Path | None], None]) -> Iterator[Path]:
        """The tree for ``key``, built by ``build`` if there is no fresh one.

        ``build`` is handed the EXPIRED tree's path when there is one -- held
        for the duration, so a refresh can carry unchanged files across instead
        of fetching them again. It must treat that tree as read-only.
        """
        tree = self._checkout(key, build)
        try:
            yield tree.path
        finally:
            with self._lock:
                tree.holders -= 1
                self._drop_if_unused(tree)

    def _checkout(self, key: str, build: Callable[[Path, Path | None], None]) -> _Tree:
        with self._lock:
            while True:
                previous = self._trees.get(key)
                if previous is not None and time.monotonic() - previous.born < self.ttl:
                    previous.holders += 1
                    return previous
                if key in self._building:
                    # Someone else is already paying for this. Waiting costs the
                    # remainder of ONE sweep; racing costs a whole extra one.
                    self._lock.wait()
                    continue
                if previous is not None:
                    previous.holders += 1
                self._building.add(key)
                break

        path = Path(tempfile.mkdtemp(prefix="poker-share-"))
        try:
            build(path, previous.path if previous is not None else None)
        except BaseException:
            # The waiters must be released even on failure, or a single bad
            # credential parks every other request until the server is killed.
            # The expired tree stays: the next reader retries the refresh from
            # it rather than from nothing.
            shutil.rmtree(path, ignore_errors=True)
            with self._lock:
                self._building.discard(key)
                self._release(previous)
                self._lock.notify_all()
            raise

        with self._lock:
            fresh = _Tree(path=path, born=time.monotonic(), holders=1)
            self._trees[key] = fresh
            if previous is not None:
                previous.retired = True
            self._release(previous)
            self._building.discard(key)
            self._lock.notify_all()
            return fresh

    def _release(self, tree: _Tree | None) -> None:
        """Let go of the hold a build took on its predecessor. Caller holds the lock."""
        if tree is not None:
            tree.holders -= 1
            self._drop_if_unused(tree)

    def _retire(self, key: str) -> None:
        """Give up the cached tree for ``key``. Caller holds the lock."""
        tree = self._trees.pop(key, None)
        if tree is not None:
            tree.retired = True
            self._drop_if_unused(tree)

    def _drop_if_unused(self, tree: _Tree) -> None:
        """Caller holds the lock."""
        if tree.retired and tree.holders == 0:
            shutil.rmtree(tree.path, ignore_errors=True)

    def close(self) -> None:
        """Retire everything. A tree still being read is deleted on release."""
        with self._lock:
            for key in list(self._trees):
                self._retire(key)


# Opt-in rather than a module default: a server answering eight panels should pay
# once, while the one-shot CLI would LOSE the guarantee its readers are built on
# -- that every answer is against the record as it is now.
_ACTIVE: SharedTrees | None = None
_ACTIVE_LOCK = threading.Lock()

RECORD_KEY = "record"


@contextmanager
def shared_record_cache(ttl: float) -> Iterator[SharedTrees]:
    """For the duration, materialising the record is memoised across readers.

    NESTS rather than refusing. Two applications in one process is something
    `create_app` explicitly promises -- a test and its subject, most of all --
    and each brings its own lifespan; refusing the second would raise a
    RuntimeError from inside a lifespan, which is a confusing place to meet one.
    The inner cache shadows the outer for its duration and takes its own trees
    with it, so neither can serve the other's answers.
    """
    global _ACTIVE
    cache = SharedTrees(ttl=ttl)
    with _ACTIVE_LOCK:
        previous, _ACTIVE = _ACTIVE, cache
    try:
        yield cache
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE = previous
        cache.close()


def active_cache() -> SharedTrees | None:
    """The cache in force, if any. For callers materialising their own subtree."""
    return _ACTIVE


def _etags(tree: Path | None) -> dict[str, str]:
    """The versions a tree was built from. Empty for a tree with no manifest."""
    if tree is None or not (tree / _ETAGS_NAME).is_file():
        return {}
    found: dict[str, str] = {}
    for line in (tree / _ETAGS_NAME).read_text().splitlines():
        etag, sep, name = line.partition("\t")
        if sep and etag:
            found[name] = etag
    return found


def _link(source: Path, destination: Path) -> None:
    """A hard link where the filesystem allows one, a copy where it does not."""
    try:
        destination.hardlink_to(source)
    except OSError:
        shutil.copyfile(source, destination)


def _materialise(root: Path, *, run: str | None, previous: Path | None = None) -> None:
    """Pull the published record into ``root``, reusing ``previous`` where it can."""
    config = CloudConfig.load()
    service = share.share_client(config)
    pull_metadata(service, config.share_name, root, run=run, previous=previous)


def _require_published(root: Path, run: str) -> None:
    """Refuse an unpublished run against a WHOLE-record tree.

    A scoped pull rejects the fragment itself, naming what is published. Served
    from the shared tree there is no scoped pull to do it, and the reader's own
    "Run not found" names two local paths instead -- so the check is repeated
    here, against the same listing, to keep one message for one failure.
    """
    published = sorted(entry.name for entry in root.iterdir() if entry.is_dir())
    matches = run_names.matching(run, published)
    if len(matches) > 1:
        raise CommandError(run_names.ambiguous_message(run, matches))
    if not matches:
        raise CommandError(
            f"'{run}' is not published. Published runs: {', '.join(published) or '(none)'}"
        )


@contextmanager
def share_records(*, run: str | None = None) -> Iterator[Path]:
    """A local runs directory holding the published record, for the duration.

    Yields a path the ordinary local readers can use, then removes it. Nothing
    is left behind: this is a question being answered, not a sync.

    Under :func:`shared_record_cache` the WHOLE record is materialised even for
    a scoped read, and every reader is served from it. A scoped pull is cheaper
    once (3.7s against 12.4s) and more expensive three times, which is what a
    run's detail page does -- and the whole tree answers every other panel for
    free.
    """
    cache = _ACTIVE
    if cache is None:
        with tempfile.TemporaryDirectory(prefix="poker-share-") as tmp:
            root = Path(tmp)
            _materialise(root, run=run)
            yield root
        return

    def _refresh(root: Path, previous: Path | None) -> None:
        """The count `pull_metadata` returns is not part of the cache protocol."""
        _materialise(root, run=None, previous=previous)

    with cache.acquire(RECORD_KEY, _refresh) as root:
        if run is not None:
            _require_published(root, run)
        yield root
