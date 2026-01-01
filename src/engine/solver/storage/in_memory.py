from pathlib import Path

import numpy as np

from src.core.game.actions import Action
from src.engine.solver.infoset import InfoSet, InfoSetKey
from src.engine.solver.storage.base import Storage
from src.engine.solver.storage.helpers import (
    build_legal_actions,
    get_missing_checkpoint_files,
    load_checkpoint_data,
)


class InMemoryStorage(Storage):
    """
    Read-only storage for loading checkpoints (charts, analysis, debugging).

    Not for training; use SharedArrayStorage for training workloads.
    """

    def __init__(self, checkpoint_dir: Path | None = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        self._infosets_by_id: dict[int, InfoSet] = {}
        self.key_to_id: dict[InfoSetKey, int] = {}
        self.id_to_key: dict[int, InfoSetKey] = {}
        self.next_id = 0

        if self.checkpoint_dir and self.checkpoint_dir.exists():
            self._load_from_checkpoint()

    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: list[Action]) -> InfoSet:
        infoset_id = self.key_to_id.get(key)
        if infoset_id is not None:
            return self._infosets_by_id[infoset_id]

        raise ValueError(
            f"InMemoryStorage is read-only. Cannot create infoset for key {key}. "
            "Use SharedArrayStorage for training."
        )

    def get_infoset(self, key: InfoSetKey) -> InfoSet | None:
        infoset_id = self.key_to_id.get(key)
        if infoset_id is None:
            return None
        return self._infosets_by_id.get(infoset_id)

    def num_infosets(self) -> int:
        return len(self._infosets_by_id)

    def iter_infosets(self):
        return self._infosets_by_id.values()

    @property
    def infosets(self) -> dict[InfoSetKey, InfoSet]:
        return {
            self.id_to_key[infoset_id]: infoset
            for infoset_id, infoset in self._infosets_by_id.items()
        }

    def checkpoint(self, iteration: int):
        raise NotImplementedError(
            "InMemoryStorage is read-only. Use SharedArrayStorage for training."
        )

    def clear(self):
        self._infosets_by_id.clear()
        self.key_to_id.clear()
        self.id_to_key.clear()
        self.next_id = 0

    def _load_from_checkpoint(self):
        if not self.checkpoint_dir:
            return
        missing_files = get_missing_checkpoint_files(self.checkpoint_dir)
        if missing_files:
            raise ValueError(f"Checkpoint is incomplete. Missing files: {missing_files}")

        data = load_checkpoint_data(self.checkpoint_dir, context="InMemoryStorage checkpoint load")
        self.key_to_id = data.owned_keys
        self.id_to_key = {v: k for k, v in self.key_to_id.items()}
        self.next_id = data.max_id

        action_sigs = data.action_signatures
        print(f"Loaded {len(action_sigs)} action signatures from checkpoint")

        all_regrets = data.arrays["regrets"]
        action_counts = data.arrays["action_counts"]
        all_strategies = data.arrays["strategies"]
        reach_counts = data.arrays["reach_counts"]
        cumulative_utilities = data.arrays["cumulative_utility"]

        for infoset_id, key in self.id_to_key.items():
            n_actions = action_counts[infoset_id]
            regrets = all_regrets[infoset_id, :n_actions].copy()
            strategies = all_strategies[infoset_id, :n_actions].astype(np.float64, copy=True)

            legal_actions = build_legal_actions(
                action_sigs, infoset_id, "InMemoryStorage checkpoint load"
            )

            infoset = InfoSet(key, legal_actions)
            infoset.regrets = regrets
            infoset.strategy_sum = strategies
            infoset.sync_stats_to_storage(
                reach_counts[infoset_id],
                cumulative_utilities[infoset_id],
            )

            self._infosets_by_id[infoset_id] = infoset

        print(f"Loaded {len(self._infosets_by_id)} infosets from checkpoint")

    def __str__(self) -> str:
        checkpoint_info = f", checkpoint_dir={self.checkpoint_dir}" if self.checkpoint_dir else ""
        return f"InMemoryStorage(num_infosets={self.num_infosets()}{checkpoint_info})"
