"""Preallocated flat-array infoset storage over a static betting tree.

Every infoset the solver can touch is known once :class:`BettingTree` is built,
so storage is a pair of flat arrays sized exactly to the tree and indexed by
arithmetic -- no key table, no id allocation, no owner map, no resize path. The
row is a pure function of the tree and identical in every process, so every
infoset is writable by every worker from the start, and the traversal's drop
counter is unreachable through this backend. A nonzero value is worth asserting
on.

Layout (ragged, zero padding, bucket-major within each street):

    regrets/strategy_sum   flat, length tree.num_slots
    reach/utility          flat, length tree.num_rows

    infoset (node n, bucket b) owns slots
        [slot_base[n] + b*slot_stride[n],  ... + num_actions[n])

Ragged rather than a dense ``(num_infosets, max_actions)`` rectangle, which at
the production tree's mean of ~2.6 actions against ``max_actions=10`` would cost
roughly 4x the memory.

Workers Hogwild-write shared memory -- lock-free and racy by design, unchanged
in its convergence argument. The arrays are mapped once at a fixed size and
every process computes identical indices, so there is nothing to exchange at
runtime.
"""

from __future__ import annotations

import hashlib
import logging
from multiprocessing import shared_memory
from typing import TYPE_CHECKING

import numpy as np

from src.engine.solver.infoset.model import InfoSet

if TYPE_CHECKING:
    from collections.abc import Iterator

    from src.engine.solver.betting_tree import BettingTree

logger = logging.getLogger(__name__)


# float32 halves the footprint of the two hot arrays at a measured relative error
# of 5.95e-8, comfortably inside f32 epsilon (see the m0002 downcast measurement).
REGRET_DTYPE = np.float32
STRATEGY_DTYPE = np.float32
UTILITY_DTYPE = np.float64
REACH_DTYPE = np.int64
VISITED_DTYPE = np.uint8

_ARRAYS = ("regrets", "strategy_sum", "reach_counts", "cumulative_utility", "visited")


class StaticArrayStorage:
    """Infoset storage backed by arrays sized to a static betting tree.

    Deliberately not a subclass of the legacy ``Storage`` ABC: that interface is
    keyed by ``InfoSetKey``, and reintroducing a string key here would restore
    the hashing this design exists to remove. Consumers address infosets by
    ``(node_id, bucket)``, which the tree already owns.
    """

    def __init__(
        self,
        tree: BettingTree,
        *,
        session_id: str | None = None,
        attach: bool = False,
    ):
        """Allocate (or attach to) arrays sized to ``tree``.

        session_id:
            ``None`` allocates process-local arrays. A name allocates them in
            shared memory so worker processes can Hogwild-write the same table.
        attach:
            Map an existing session's arrays instead of creating them — the
            worker-process path. The worker rebuilds the tree locally from config
            rather than receiving it, so no index information crosses a process
            boundary at any point. Requires ``session_id``.
        """
        if attach and session_id is None:
            raise ValueError("attach=True requires a session_id to attach to")

        self.tree = tree
        self.session_id = session_id
        self._shm: list[shared_memory.SharedMemory] = []
        self._owns_shm = False

        if session_id is None:
            self.regrets = np.zeros(tree.num_slots, dtype=REGRET_DTYPE)
            self.strategy_sum = np.zeros(tree.num_slots, dtype=STRATEGY_DTYPE)
            self.reach_counts = np.zeros(tree.num_rows, dtype=REACH_DTYPE)
            self.cumulative_utility = np.zeros(tree.num_rows, dtype=UTILITY_DTYPE)
            self.visited = np.zeros(tree.num_rows, dtype=VISITED_DTYPE)
        elif attach:
            self._map_shared(create=False)
        else:
            self._map_shared(create=True)

    # ---- shared memory ---------------------------------------------------

    def _spec(self) -> dict[str, tuple[int, np.dtype]]:
        return {
            "regrets": (self.tree.num_slots, np.dtype(REGRET_DTYPE)),
            "strategy_sum": (self.tree.num_slots, np.dtype(STRATEGY_DTYPE)),
            "reach_counts": (self.tree.num_rows, np.dtype(REACH_DTYPE)),
            "cumulative_utility": (self.tree.num_rows, np.dtype(UTILITY_DTYPE)),
            "visited": (self.tree.num_rows, np.dtype(VISITED_DTYPE)),
        }

    def _shm_name(self, array: str) -> str:
        """Segment name, keyed by BOTH the session and the tree fingerprint.

        Workers rebuild the tree locally and then index shared arrays with it, so
        two processes disagreeing about the tree would each write to different
        rows of the same memory — silent, total corruption with no error
        anywhere. Folding the fingerprint into the name makes that unattachable
        rather than merely unlikely: a mismatched tree computes a different name
        and simply cannot find the segments.

        Hashed to a fixed width because shared-memory names are length-capped
        (31 characters on macOS), so concatenating a session id and a fingerprint
        would overflow for ordinary session names.
        """
        digest = hashlib.sha256(
            f"{self.session_id}|{self.tree.fingerprint()}".encode()
        ).hexdigest()[:12]
        return f"sts_{array[:4]}_{digest}"

    def _create_segment(self, array: str, size: int) -> shared_memory.SharedMemory:
        """Create this session's segment, reclaiming one an earlier task abandoned.

        POSIX shared memory outlives the process that made it. A coordinator
        killed without reaching :meth:`close` -- SIGKILL, an OOM, the wall-clock
        guard -- leaves its segments in ``/dev/shm``, and because the session id
        IS the run id, the NEXT task for that run computes the same names and dies
        on ``FileExistsError`` before doing any work. That breaks retry-safety
        exactly where the design leans on it hardest: a task is meant to be
        re-runnable to an absolute target, and a Batch retry lands on the same
        node by preference.

        Reclaiming is safe because the name encodes the run: another task holding
        it is another task training the SAME run, which is already incoherent --
        two coordinators would interleave writes into one table and checkpoint
        over each other. So a name that is already taken means a dead predecessor,
        not a live peer. Tasks for different runs, or against a different tree,
        hash to different names and are untouched.
        """
        name = self._shm_name(array)
        try:
            return shared_memory.SharedMemory(name=name, create=True, size=size)
        except FileExistsError:
            logger.warning(
                "Reclaiming shared segment %s left behind by an earlier task of session %r.",
                name,
                self.session_id,
            )
            # `track=False`: this process is destroying the segment, not adopting
            # it, so the tracker should never hear about it in either direction.
            #
            # FileNotFoundError is not an error here: it means the segment vanished
            # between the failed create and this attach (the dead task's resource
            # tracker getting there first), which is the outcome we wanted.
            try:
                stale = shared_memory.SharedMemory(name=name, create=False, track=False)
                stale.close()
                stale.unlink()
            except FileNotFoundError:
                pass
            return shared_memory.SharedMemory(name=name, create=True, size=size)

    def _map_shared(self, *, create: bool) -> None:
        """Create or attach the shared segments and bind them as array views.

        Attaching is UNTRACKED, and that word carries two failures.

        On POSIX a resource tracker unlinks whatever it holds when its process
        dies, and ``SharedMemory`` registers every segment it opens -- including
        one it merely attached to. So an attacher that dies destroys arrays the
        coordinator is still training on. Measured, and the distinction is what
        makes it subtle:

            spawned child (shares the parent's tracker), SIGKILLed  -> SURVIVES
            SEPARATE interpreter, SIGTERM or SIGKILL                -> DESTROYED

        A 50M task died at 38,000,000 with all 16 workers of the next chunk
        raising FileNotFoundError on segments that existed moments earlier.

        The first fix was to attach normally and then unregister. That closed
        the hazard above and opened a quieter one, because a spawned worker
        SHARES the coordinator's tracker: the worker's unregister removed the
        COORDINATOR's entry, and the tracker holds one entry per name, not one
        per process. Two consequences, both seen in production. The coordinator's
        own ``unlink()`` then unregistered a name no longer in the cache, so the
        tracker raised ``KeyError`` and printed a traceback -- five per chunk
        boundary, in the log a human reads to find out why a task died. And the
        segments were left untracked, so a coordinator lost to SIGKILL, an OOM or
        the wall-clock guard no longer had its arrays reclaimed at all, which is
        what :meth:`_create_segment` keeps finding in ``/dev/shm``.

        ``track=False`` states the actual intent -- an attacher takes no
        ownership -- and leaves the creator's registration, the one that SHOULD
        clean up after a kill, untouched.
        """
        self._owns_shm = create
        for name, (length, dtype) in self._spec().items():
            if create:
                shm = self._create_segment(name, max(1, length * dtype.itemsize))
            else:
                try:
                    shm = shared_memory.SharedMemory(
                        name=self._shm_name(name), create=False, track=False
                    )
                except FileNotFoundError as exc:
                    raise FileNotFoundError(
                        f"No shared segment for array {name!r} in session "
                        f"{self.session_id!r} under betting tree "
                        f"{self.tree.fingerprint()}. Either the session is not "
                        "running, or this process built a DIFFERENT tree than the "
                        "coordinator — segment names are keyed by the fingerprint "
                        "precisely so a mismatch cannot silently attach."
                    ) from exc
            self._shm.append(shm)
            array = np.ndarray((length,), dtype=dtype, buffer=shm.buf)
            if create:
                array.fill(0)
            setattr(self, name, array)

    def close(self) -> None:
        """Release this process's mappings; the creator also unlinks them."""
        for shm in self._shm:
            try:
                shm.close()
                if self._owns_shm:
                    shm.unlink()
            except (FileNotFoundError, BufferError):
                pass
        self._shm = []

    # ---- infoset access --------------------------------------------------

    def infoset_at(self, node_id: int, bucket: int) -> InfoSet:
        """Resolve an infoset for the traversal, marking it as covered.

        The returned views alias shared memory — writes through
        ``infoset.regrets`` land in storage with no write-back step. Always
        writable: a static row cannot be unknown to a worker.

        Coverage is marked here rather than from ``reach_counts`` because that
        counter increments only where the traverser enumerates its own actions,
        so it silently omits opponent nodes, where the average strategy
        accumulates. Counting those as untouched understates coverage by more
        than half — and coverage is the diagnostic that makes under-training
        visible. Read-only consumers must use :meth:`view` instead, or a metrics
        sweep would mark the entire tree as covered.
        """
        infoset = self.view(node_id, bucket)
        self.visited[infoset.row] = 1
        return infoset

    def view(self, node_id: int, bucket: int) -> InfoSet:
        """Same view as :meth:`infoset_at` but without marking coverage.

        The bounds check is not defensive boilerplate. Under ``base + bucket *
        stride`` an out-of-range bucket does not fall off the end of the array —
        it lands on a perfectly valid row belonging to a *different node*, and
        two unrelated infosets then share storage with no error anywhere. The
        old key-addressed design was immune (a bad bucket just made a distinct,
        meaningless key); flat indexing is not, so the check buys back the
        safety that indexing gave up. The production bucketer raises on illegal
        combos before reaching here, so this fires only for a buggy or custom
        ``BucketingStrategy`` — which is exactly when silent corruption would be
        hardest to trace.
        """
        node = self.tree.nodes[node_id]
        if not 0 <= bucket < self.tree.buckets_per_node[node_id]:
            raise IndexError(
                f"bucket {bucket} out of range for node {node_id} "
                f"({node.street.name}, {self.tree.buckets_per_node[node_id]} buckets). "
                "An out-of-range bucket would alias another node's infoset."
            )

        start, end = self.tree.slots(node_id, bucket)
        row = self.tree.row(node_id, bucket)

        infoset = InfoSet(None, node.legal_actions, allocate_arrays=False)
        infoset.regrets = self.regrets[start:end]
        infoset.strategy_sum = self.strategy_sum[start:end]
        infoset.node_id = node_id
        infoset.bucket = bucket
        infoset.row = row
        infoset.attach_stats_views(self.reach_counts, self.cumulative_utility, row, read_only=False)
        infoset.sync_stats_to_storage(self.reach_counts[row], self.cumulative_utility[row])
        return infoset

    def view_at_row(self, row: int) -> InfoSet:
        """Read-only-safe view for a flat row index."""
        node_id, bucket = self.tree.row_to_infoset(row)
        return self.view(node_id, bucket)

    def num_infosets(self) -> int:
        """Total rows in the tree.

        A property of the CONFIG, not of progress -- same before and after a
        run. :meth:`num_touched_infosets` is the one that moves.
        """
        return self.tree.num_rows

    def num_touched_infosets(self) -> int:
        """Rows the traversal has resolved at least once — the real progress signal.

        ``num_touched / num_infosets`` is tree coverage, and
        ``reach_counts.sum() / num_touched`` is the mean visits per touched
        infoset. The second number is the one that matters: CFR needs it in the
        thousands, and a static denominator makes that checkable during a run
        rather than inferable afterwards.
        """
        return int(np.count_nonzero(self.visited))

    def coverage(self) -> float:
        """Fraction of the enumerated tree the traversal has reached."""
        return self.num_touched_infosets() / self.tree.num_rows if self.tree.num_rows else 0.0

    def mean_visits_per_touched_infoset(self) -> float:
        """Mean traverser visits across rows that have been reached at all.

        The headline convergence diagnostic. Compare against the 1e3-1e4 that
        CFR needs before a regret average carries signal.
        """
        touched = self.num_touched_infosets()
        return float(self.reach_counts.sum()) / touched if touched else 0.0

    def iter_infosets(self) -> Iterator[InfoSet]:
        for node in self.tree.nodes:
            for bucket in range(self.tree.num_buckets(node.street)):
                yield self.view(node.node_id, bucket)

    def iter_touched_infosets(self) -> Iterator[InfoSet]:
        """Only rows the traversal has reached — the useful subset for metrics."""
        for node in self.tree.nodes:
            flags = self.tree.node_row_vector(self.visited, node.node_id)
            for bucket in np.flatnonzero(flags):
                yield self.view(node.node_id, int(bucket))

    # ---- diagnostics -----------------------------------------------------

    def nbytes(self) -> int:
        return sum(int(getattr(self, name).nbytes) for name in _ARRAYS)

    def __str__(self) -> str:
        return (
            f"StaticArrayStorage(nodes={len(self.tree)}, rows={self.tree.num_rows:,}, "
            f"slots={self.tree.num_slots:,}, {self.nbytes() / 1e6:.1f} MB)"
        )
