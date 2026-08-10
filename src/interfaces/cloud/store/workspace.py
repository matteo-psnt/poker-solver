"""Answering a question about the record without keeping a copy of it.

Every reading command -- ``runinfo``, ``progress``, ``curve``, ``report``,
``compare``, ``ledger`` -- used to require a local ``data/runs`` populated by
``fetch``. The data was in the cloud; the questions were only answerable on the
machine that had last synced. Two boxes could hold different answers, and a
fresh checkout could hold none.

Reading materialises the published JSON into a temporary directory, answers the
question there, and throws it away -- unless a :func:`shared_record_cache` is in
force, in which case one tree answers for every reader that arrives inside its
lifetime. That is a server's concern and nothing else's; see the note above it.
Materialising rather than
reading in place is deliberate: the record is a few hundred KB of small JSON, so
pulling it costs one round trip per file, while reading it in place would make
every command an SMB client and every reader aware of two filesystems. The
readers stay ordinary local-path code; only where the path comes from changes.

Checkpoints are never materialised. They are ~540 MB of zarr chunks that no
reading command opens, and the rule that the manifest defines what is complete
stays in ``fetch``, which is the command that genuinely wants them.
"""

from __future__ import annotations

import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from azure.storage.fileshare import ShareServiceClient

from src.interfaces import run_names
from src.interfaces.cloud.config import CloudConfig
from src.interfaces.cloud.store import share
from src.interfaces.errors import CommandError

BASELINE_NAME = "baseline.json"

# `<op>_result.json` -- train, evaluate, resume, train-static. The writer was
# deleted with the clobbering result file it produced; nothing reads them, and
# they are still on the share as history. `fetch` still syncs them, because that
# is a copy of the record; this is a question being asked of it.
DEAD_SUFFIX = "_result.json"

"""How wide to fan out
--------------------
Every one of these is a round trip carrying a few kilobytes, so the pool is
sized by latency and not by bandwidth or CPU: 105 metadata files measured 7.5s
at 16 threads, 4.6s at 32 and 3.5s at 64. Threads blocked on a socket cost
almost nothing, and the share is not the bottleneck -- the round trip is.
"""
_PARALLEL_DOWNLOADS = 64


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
) -> int:
    """Download every published JSON record into ``destination``. Returns the count."""
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
    def _walk(name: str) -> list[tuple[str, Path]]:
        found: list[tuple[str, Path]] = []
        for remote in share.walk_files(
            service,
            share_name,
            f"{share.ARCHIVE_DIR}/{name}",
            skip_dir=_is_snapshot_dir,
        ):
            relative = remote[len(f"{share.ARCHIVE_DIR}/") :]
            leaf = Path(relative).name
            if share.is_snapshot_path(relative) or not share.is_metadata(leaf):
                continue
            if leaf.endswith(DEAD_SUFFIX):
                continue
            found.append((remote, destination / relative))
        return found

    with ThreadPoolExecutor(max_workers=min(_PARALLEL_DOWNLOADS, len(published) or 1)) as pool:
        wanted = [entry for batch in pool.map(_walk, published) for entry in batch]

    # One round trip per file, and a run's eval documents now carry their full
    # sample vectors -- so this is latency-bound on a link where latency is the
    # whole cost. The downloads are independent and `download_file` builds its
    # own file client, so they overlap.
    with ThreadPoolExecutor(max_workers=_PARALLEL_DOWNLOADS) as pool:
        futures = [
            pool.submit(share.download_file, service, share_name, remote, local)
            for remote, local in wanted
        ]
        for future in futures:
            future.result()
    return len(wanted)


def read_baseline(service: ShareServiceClient, share_name: str) -> str | None:
    """The published baseline document, or None when none has been promoted."""
    return share.read_text(service, share_name, BASELINE_NAME)


def write_baseline(service: ShareServiceClient, share_name: str, body: str) -> None:
    """Publish the baseline pointer.

    Small, single-writer, and the conclusion of every experiment: which run the
    next one forks from. It was the only artifact that never left the machine
    that wrote it, so a reinstall lost it and two boxes could silently disagree.
    """
    share.write_text(service, share_name, BASELINE_NAME, body)


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
    def acquire(self, key: str, build: Callable[[Path], None]) -> Iterator[Path]:
        """The tree for ``key``, built by ``build`` if there is no fresh one."""
        tree = self._checkout(key, build)
        try:
            yield tree.path
        finally:
            with self._lock:
                tree.holders -= 1
                self._drop_if_unused(tree)

    def _checkout(self, key: str, build: Callable[[Path], None]) -> _Tree:
        with self._lock:
            while True:
                tree = self._trees.get(key)
                if tree is not None and time.monotonic() - tree.born < self.ttl:
                    tree.holders += 1
                    return tree
                if key in self._building:
                    # Someone else is already paying for this. Waiting costs the
                    # remainder of ONE sweep; racing costs a whole extra one.
                    self._lock.wait()
                    continue
                self._retire(key)
                self._building.add(key)
                break

        path = Path(tempfile.mkdtemp(prefix="poker-share-"))
        try:
            build(path)
        except BaseException:
            # The waiters must be released even on failure, or a single bad
            # credential parks every other request until the server is killed.
            shutil.rmtree(path, ignore_errors=True)
            with self._lock:
                self._building.discard(key)
                self._lock.notify_all()
            raise

        with self._lock:
            fresh = _Tree(path=path, born=time.monotonic(), holders=1)
            self._trees[key] = fresh
            self._building.discard(key)
            self._lock.notify_all()
            return fresh

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


"""Who gets to share, and who does not
-----------------------------------
Sharing is opt-in through a context manager rather than a module-level default,
because the two callers want opposite things. A server answers the same question
for eight panels a second apart and should pay once. The command line is
one-shot, gains nothing -- and would LOSE the guarantee the readers are built
on: that every answer is against the published record as it is now. A run
published thirty seconds ago must not be invisible to `promote`.
"""
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


def _materialise(root: Path, *, run: str | None) -> None:
    """Pull the published record into ``root``."""
    config = CloudConfig.load()
    service = share.share_client(config)
    pull_metadata(service, config.share_name, root, run=run)
    baseline = read_baseline(service, config.share_name)
    if baseline is not None:
        (root / BASELINE_NAME).write_text(baseline)


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

    with cache.acquire(RECORD_KEY, lambda root: _materialise(root, run=None)) as root:
        if run is not None:
            _require_published(root, run)
        yield root
