from collections.abc import Iterator, Sequence
from pathlib import Path

import numpy as np

from src.core.game.actions import Action
from src.engine.solver.infoset import InfoSet, InfoSetKey
from src.engine.solver.storage.base import Storage
from src.engine.solver.storage.helpers import (
    get_missing_checkpoint_files,
    load_checkpoint_data,
)


class InMemoryStorage(Storage):
    """Read-only storage for loading checkpoints (charts, play, analysis, debugging).

    Not for training; use SharedArrayStorage for training workloads.

    InfoSet objects are materialized lazily. Loading holds the decompressed
    checkpoint arrays and the key<->id maps, but builds an :class:`InfoSet` (a
    Python object plus per-row array copies) only when its key is first looked
    up. An interactive session -- a played hand, a preflop chart -- touches a tiny
    fraction of a large table, so eagerly constructing all N objects was pure
    latency: ~36s of a ~77s load at 6.8M infosets. Consumers that genuinely need
    every infoset (:meth:`iter_infosets`, :attr:`infosets`) still materialize the
    whole table, just on demand rather than up front.
    """

    def __init__(self, checkpoint_dir: Path | None = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        # Lazily-materialized InfoSet cache, keyed by id. The authoritative row
        # count is ``id_to_key``; this may hold only the touched subset.
        self._infosets_by_id: dict[int, InfoSet] = {}
        self.key_to_id: dict[InfoSetKey, int] = {}
        self.id_to_key: dict[int, InfoSetKey] = {}
        self.next_id = 0
        # Source arrays for lazy materialization (None until a checkpoint loads).
        self._arrays: dict[str, np.ndarray] | None = None
        self._action_lists: list[list[Action]] | None = None

        if self.checkpoint_dir and self.checkpoint_dir.exists():
            self._load_from_checkpoint()

    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: Sequence[Action]) -> InfoSet:
        infoset_id = self.key_to_id.get(key)
        if infoset_id is not None:
            return self._materialize(infoset_id)

        raise ValueError(
            f"InMemoryStorage is read-only. Cannot create infoset for key {key}. "
            "Use SharedArrayStorage for training."
        )

    def get_infoset(self, key: InfoSetKey) -> InfoSet | None:
        infoset_id = self.key_to_id.get(key)
        if infoset_id is None:
            return None
        return self._materialize(infoset_id)

    def num_infosets(self) -> int:
        # The count is the number of stored rows, not how many have been
        # materialized so far, so it holds before any lookup.
        return len(self.id_to_key)

    def iter_infosets(self) -> Iterator[InfoSet]:
        for infoset_id in self.id_to_key:
            yield self._materialize(infoset_id)

    @property
    def infosets(self) -> dict[InfoSetKey, InfoSet]:
        return {key: self._materialize(infoset_id) for infoset_id, key in self.id_to_key.items()}

    def checkpoint(self, iteration: int):
        raise NotImplementedError(
            "InMemoryStorage is read-only. Use SharedArrayStorage for training."
        )

    def clear(self):
        self._infosets_by_id.clear()
        self.key_to_id.clear()
        self.id_to_key.clear()
        self.next_id = 0
        self._arrays = None
        self._action_lists = None

    def _materialize(self, infoset_id: int) -> InfoSet:
        """Build (and cache) the InfoSet for ``infoset_id`` from the source arrays."""
        cached = self._infosets_by_id.get(infoset_id)
        if cached is not None:
            return cached
        assert self._arrays is not None and self._action_lists is not None

        n_actions = self._arrays["action_counts"][infoset_id]
        infoset = InfoSet(self.id_to_key[infoset_id], self._action_lists[infoset_id])
        infoset.regrets = self._arrays["regrets"][infoset_id, :n_actions].copy()
        infoset.strategy_sum = self._arrays["strategies"][infoset_id, :n_actions].astype(
            np.float64, copy=True
        )
        infoset.sync_stats_to_storage(
            self._arrays["reach_counts"][infoset_id],
            self._arrays["cumulative_utility"][infoset_id],
        )
        self._infosets_by_id[infoset_id] = infoset
        return infoset

    def _load_from_checkpoint(self):
        if not self.checkpoint_dir:
            return
        missing_files = get_missing_checkpoint_files(self.checkpoint_dir)
        if missing_files:
            raise ValueError(f"Checkpoint is incomplete. Missing files: {missing_files}")

        data = load_checkpoint_data(self.checkpoint_dir, context="InMemoryStorage checkpoint load")
        # Row index is the infoset id, so the two directions come straight from order.
        self.id_to_key = dict(enumerate(data.keys))
        self.key_to_id = {key: i for i, key in self.id_to_key.items()}
        self.next_id = data.max_id
        self._arrays = data.arrays
        self._action_lists = data.action_lists

        print(f"Loaded {len(self.id_to_key)} infosets from checkpoint (materialized on demand)")

    def __str__(self) -> str:
        checkpoint_info = f", checkpoint_dir={self.checkpoint_dir}" if self.checkpoint_dir else ""
        return f"InMemoryStorage(num_infosets={self.num_infosets()}{checkpoint_info})"
