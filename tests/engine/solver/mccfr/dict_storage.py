"""A dict-backed :class:`Storage` for the Kuhn/Leduc conformance harness.

WHY THIS IS IN tests/. The conformance harness validates the CFR kernel against
games with exact analytic equilibria, and it does so by driving ``MCCFRSolver``
over a generic ``ExtensiveGame`` rather than HUNL. Those games have no betting
tree to enumerate, so the production static backend -- which addresses infosets
by ``(node_id, bucket)`` over a tree built from a poker config -- cannot back
them at all.

It used to borrow the dynamic backend's key-addressed storage. That backend is
gone, and re-adding one to ``src/`` purely to keep this harness running would
put a second, unused storage implementation back into production code. So the
harness carries its own: a dict, which is all a few-thousand-infoset toy game
ever needed, and which cannot drift from a production backend because it is not
pretending to be one.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from src.core.game.actions import Action
from src.engine.solver.infoset import InfoSet, InfoSetKey
from src.engine.solver.storage.base import Storage


class DictStorage(Storage):
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

    def checkpoint(self, iteration: int) -> None:
        """No-op: the harness asserts on in-memory strategies, never on disk."""
