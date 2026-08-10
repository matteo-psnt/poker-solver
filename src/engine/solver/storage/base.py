from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from pathlib import Path

from src.core.game.actions import Action
from src.engine.solver.infoset.model import InfoSet, InfoSetKey


class Storage(ABC):
    """Abstract base class for infoset storage."""

    checkpoint_dir: Path | None = None

    @abstractmethod
    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: Sequence[Action]) -> InfoSet:
        """Get existing infoset or create new one."""

    @abstractmethod
    def get_infoset(self, key: InfoSetKey) -> InfoSet | None:
        """Get existing infoset or None if not found."""

    @abstractmethod
    def num_infosets(self) -> int:
        """Get total number of stored infosets."""

    @abstractmethod
    def iter_infosets(self) -> Iterable[InfoSet]:
        """Iterate over infosets owned by this storage."""

    @abstractmethod
    def checkpoint(self, iteration: int) -> None:
        """Save a checkpoint at given iteration."""
