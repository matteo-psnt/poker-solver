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
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.interfaces import run_names
from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.store import blob
from src.interfaces.errors import CommandError
from src.shared import records
from src.shared.cloudtask.node import archive

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping, Sequence


# Sized by latency, not bandwidth: 105 metadata files took 7.5s at 16 threads,
# 4.6s at 32, 3.5s at 64. Threads blocked on a socket cost almost nothing.
_PARALLEL_DOWNLOADS = 64

# The etag manifest an incremental refresh compares against, written at the ROOT
# of a materialised tree. Every reader of that tree filters to directories, so a
# file beside the run directories is invisible to them.
_ETAGS_NAME = "records.etags"


def pull_metadata(
    destination: Path,
    record: Mapping[str, Mapping[str, Any]],
    *,
    run: str | None = None,
) -> int:
    """Materialise the published record into ``destination``. Returns manifests.

    `record` is what the container holds, from `blob.published_record` -- passed
    in rather than read here, so this stays a function about building a tree and
    the one round trip has a single owner.
    """
    published = sorted(record)
    if run is not None:
        # Resolved HERE, against what is actually published, because this
        # decides what gets written -- a fragment rejected at this point never
        # reaches `resolve_run_dir`, and the reader would refuse a run that
        # exists. Same rule on both sides: `src.interfaces.run_names`.
        matches = run_names.matching(run, published)
        if len(matches) > 1:
            raise CommandError(run_names.ambiguous_message(run, matches))
        if not matches:
            raise CommandError(run_names.unknown_message(run, published))
        published = matches

    # THE CONTAINER IS THE RECORD. Every run's manifest and every rung it
    # holds come from one listing plus a small download each -- no share, no
    # etag cache, no hard links. That machinery existed for 4,251 immutable
    # eval documents re-fetched on a 45s console TTL; those live in Postgres
    # now and what is left is 345 manifests of a few KB.
    written = 0
    for name in published:
        entry = record.get(name) or {}
        run_dir = destination / name
        run_dir.mkdir(parents=True, exist_ok=True)
        body = entry.get("manifest")
        if body is not None:
            (run_dir / records.STATIC_CHECKPOINT).write_bytes(body)
            written += 1
        # A marker's whole content is that it exists, and what it says is which
        # rungs a fetch could actually get. The container's listing is that.
        for rung in entry.get("rungs") or ():
            (run_dir / archive.marker_for(rung)).touch()
    return written


def resolve_published_run(run: str) -> str:
    """The full published run id for ``run``, which may be a fragment.

    Readers resolve a fragment locally (``resolve_run_dir``); DISPATCH has to
    resolve too, because the id is sent to a node and the node has no fragment
    matcher. Unresolved, `score --run 15261` cost a snapshot upload, a node
    allocation and three retries before failing "no such run".

    Resolved against the CONTAINER, because that is the store a node will fetch
    from. Asking the share instead is what made every run invisible to dispatch
    the moment the snapshots stopped landing there: the listing went empty and
    `score --run` refused a run whose rungs were all present, a gate reading the
    store that no longer answers.

    A DELIMITER walk, so resolving a name costs 332 prefixes rather than all
    1,695 objects -- 0.27s against 2.40s.
    """
    config = CloudConfig.load()
    published = blob.published_run_ids(config)
    matches = run_names.matching(run, published)
    if len(matches) > 1:
        raise CommandError(run_names.ambiguous_message(run, matches))
    if not matches:
        raise CommandError(run_names.unknown_message(run, published))
    return matches[0]


def verify_published_rungs(run_id: str, rungs: Sequence[str]) -> None:
    """Refuse rungs the CONTAINER does not hold, before anything is dispatched.

    THE MANIFEST SAYS PUBLISHED, THE CONTAINER SAYS WHAT IS THERE. Pruning
    removes a snapshot without rewriting the manifest that advertises it, so
    `runinfo` offers rungs that a node then cannot fetch -- unverified, each
    cost a snapshot upload, a node allocation and a `uv sync` before dying,
    ~26 tasks in the 2026-08-23/24 window.

    Presence IS completeness here: one rung is one atomically-committed object,
    so there is no half-written state a marker had to rule out.

    An empty rung means "the latest checkpoint", which the ladder cannot name in
    advance and the node resolves itself, so it is not checked here.
    """
    wanted = [rung for rung in rungs if rung]
    if not wanted:
        return
    config = CloudConfig.load()
    # BY PREFIX. This asked for every run's rungs and kept one run's: 0.23s
    # against 2.40s, paid on every `score --at` before anything is dispatched.
    held = blob.rungs_for(config, run_id)
    published = {
        name.removeprefix("static-").removesuffix(records.SNAPSHOT_SUFFIX): name for name in held
    }
    available = sorted(published)
    missing = [rung for rung in wanted if rung not in published]
    if missing:
        raise CommandError(
            f"{run_id} has no published checkpoint for: {', '.join(missing)}.\n"
            f"  In the container: {', '.join(available) or '(none)'}\n"
            "A rung the manifest advertises can still have been pruned -- this checks "
            "the container itself, so the mismatch surfaces here instead of on a node."
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
    stale-while-revalidate
        A reader arriving during a rebuild is handed the EXPIRED tree rather
        than blocked on the new one, for ``stale_grace`` past the TTL. Discovery
        alone is ~5.4s against the share (measured: 1.3s to list 300 runs, 4.0s
        to walk them), so blocking made roughly one page load in six pay a
        multi-second wait for data it did not need to be that fresh.
    """

    ttl: float
    # Mirrors `TtlCache`'s `serve_stale_for` one layer up, and for the same
    # reason: a refresh that keeps failing must reach the caller as a failure at
    # a bounded age rather than ageing silently behind a badge.
    stale_grace: float = 0.0
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
                    # Someone else is already paying for this. Racing them costs
                    # a whole extra sweep, so never build here -- but do not WAIT
                    # for them either while a readable tree is in hand: the
                    # builder already holds `previous`, so serving it costs one
                    # more refcount and no round trips. Bounded, because a build
                    # that keeps failing must eventually be reported rather than
                    # answered from an ever-older tree.
                    if previous is not None and self._within_grace(previous):
                        previous.holders += 1
                        return previous
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

    def _within_grace(self, tree: _Tree) -> bool:
        """Whether an EXPIRED tree is still young enough to answer from."""
        return time.monotonic() - tree.born < self.ttl + self.stale_grace

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
def shared_record_cache(ttl: float, stale_grace: float = 0.0) -> Iterator[SharedTrees]:
    """For the duration, materialising the record is memoised across readers.

    NESTS rather than refusing. Two applications in one process is something
    `create_app` explicitly promises -- a test and its subject, most of all --
    and each brings its own lifespan; refusing the second would raise a
    RuntimeError from inside a lifespan, which is a confusing place to meet one.
    The inner cache shadows the outer for its duration and takes its own trees
    with it, so neither can serve the other's answers.
    """
    global _ACTIVE
    cache = SharedTrees(ttl=ttl, stale_grace=stale_grace)
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


def _materialise(root: Path, *, run: str | None, previous: Path | None = None) -> None:  # noqa: ARG001 -- `previous` is the console's cache handle, kept while it still passes one
    """Pull the published record into ``root``."""
    pull_metadata(root, blob.published_record(CloudConfig.load()), run=run)


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
        raise CommandError(run_names.unknown_message(run, published))


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
