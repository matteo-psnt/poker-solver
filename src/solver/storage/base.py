from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional

from src.bucketing.utils.infoset import InfoSet, InfoSetKey
from src.game.actions import Action


class Storage(ABC):
    """Abstract base class for infoset storage."""

    checkpoint_dir: Optional[Path] = None

    @abstractmethod
    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: list[Action]) -> InfoSet:
        """Get existing infoset or create new one."""
        pass

    @abstractmethod
    def get_infoset(self, key: InfoSetKey) -> Optional[InfoSet]:
        """Get existing infoset or None if not found."""
        pass

    @abstractmethod
    def num_infosets(self) -> int:
        """Get total number of stored infosets."""
        pass

    @abstractmethod
    def iter_infosets(self) -> Iterable[InfoSet]:
        """Iterate over infosets owned by this storage."""
        pass

    def is_owned(self, key: InfoSetKey) -> bool:
        """
        Check if this storage instance owns the given infoset key.

        For non-partitioned storage, all keys are "owned" (returns True).
        For partitioned storage, only keys mapping to this worker's partition are owned.
        """
        return True

    @abstractmethod
    def checkpoint(self, iteration: int):
        """Save a checkpoint at given iteration."""
        pass
