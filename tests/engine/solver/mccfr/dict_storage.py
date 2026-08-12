"""Dict-backed :class:`KeyedStorage` for the Kuhn/Leduc conformance harness.

Lives in tests/ because Kuhn and Leduc have no betting tree, so the production
static backend cannot back them -- and a second storage implementation in
``src/`` that nothing ships would be worse than a dict here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.engine.solver.infoset.model import InfoSet, InfoSetKey

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from src.core.game.actions import Action


class DictStorage:
    """Key-addressed infoset storage over a plain dict."""

    def __init__(self) -> None:
        self._infosets: dict[InfoSetKey, InfoSet] = {}

    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: Sequence[Action]) -> InfoSet:
        infoset = self._infosets.get(key)
        if infoset is None:
            infoset = InfoSet(key, tuple(legal_actions))
            self._infosets[key] = infoset
        return infoset

    def get_infoset(self, key: InfoSetKey) -> InfoSet | None:
        return self._infosets.get(key)

    def num_infosets(self) -> int:
        return len(self._infosets)

    def iter_infosets(self) -> Iterable[InfoSet]:
        return list(self._infosets.values())

