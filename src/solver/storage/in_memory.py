import pickle
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from src.bucketing.utils.infoset import InfoSet, InfoSetKey
from src.game.actions import Action
from src.solver.storage.base import Storage
from src.solver.storage.helpers import (
    CHECKPOINT_REQUIRED_FILES,
    _reconstruct_action,
    _validate_action_signatures,
    get_missing_checkpoint_files,
)


class InMemoryStorage(Storage):
    """
    Read-only storage for loading checkpoints (charts, analysis, debugging).

    Not for training; use SharedArrayStorage for training workloads.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None

        self._infosets_by_id: Dict[int, InfoSet] = {}
        self.key_to_id: Dict[InfoSetKey, int] = {}
        self.id_to_key: Dict[int, InfoSetKey] = {}
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

    def get_infoset(self, key: InfoSetKey) -> Optional[InfoSet]:
        infoset_id = self.key_to_id.get(key)
        if infoset_id is None:
            return None
        return self._infosets_by_id.get(infoset_id)

    def num_infosets(self) -> int:
        return len(self._infosets_by_id)

    def iter_infosets(self):
        return self._infosets_by_id.values()

    @property
    def infosets(self) -> Dict[InfoSetKey, InfoSet]:
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
            raise ValueError(
                f"Checkpoint is incomplete. Missing files: {missing_files}\n"
                f"Required: {list(CHECKPOINT_REQUIRED_FILES)}"
            )

        key_mapping_file = self.checkpoint_dir / "key_mapping.pkl"
        regrets_file = self.checkpoint_dir / "regrets.npy"
        strategies_file = self.checkpoint_dir / "strategies.npy"
        action_counts_file = self.checkpoint_dir / "action_counts.npy"
        reach_counts_file = self.checkpoint_dir / "reach_counts.npy"
        cumulative_utility_file = self.checkpoint_dir / "cumulative_utility.npy"

        with open(key_mapping_file, "rb") as f:
            mapping_data = pickle.load(f)
        try:
            self.key_to_id = mapping_data["owned_keys"]
            self.id_to_key = {v: k for k, v in self.key_to_id.items()}
            self.next_id = mapping_data["max_id"]
        except KeyError as exc:
            raise ValueError("Invalid checkpoint format: missing key mappings") from exc

        action_sigs_file = self.checkpoint_dir / "action_signatures.pkl"
        with open(action_sigs_file, "rb") as f:
            saved_action_sigs = pickle.load(f)
        print(f"Loaded {len(saved_action_sigs)} action signatures from checkpoint")

        all_regrets = np.load(regrets_file, mmap_mode="r")
        action_counts = np.load(action_counts_file, mmap_mode="r")
        all_strategies = np.load(strategies_file, mmap_mode="r")
        reach_counts = np.load(reach_counts_file, mmap_mode="r")
        cumulative_utilities = np.load(cumulative_utility_file, mmap_mode="r")

        _validate_action_signatures(
            action_counts, saved_action_sigs, "InMemoryStorage checkpoint load"
        )

        for infoset_id, key in self.id_to_key.items():
            n_actions = action_counts[infoset_id]
            regrets = all_regrets[infoset_id, :n_actions].copy()
            strategies = all_strategies[infoset_id, :n_actions].astype(np.float64, copy=True)

            if infoset_id not in saved_action_sigs:
                raise ValueError(
                    f"Missing action signatures for infoset ID {infoset_id} in checkpoint. "
                    "Cannot reconstruct legal actions."
                )

            action_sigs = saved_action_sigs[infoset_id]
            legal_actions = [
                _reconstruct_action(action_type_name, amount)
                for action_type_name, amount in action_sigs
            ]

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
