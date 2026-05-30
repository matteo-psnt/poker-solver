"""What the CFR kernel needs from storage, in two structural pieces.

Protocols, not base classes. Nothing here is inherited: `StaticArrayStorage` is
a plain class and the Kuhn/Leduc harness's `DictStorage` is a plain class, and
each satisfies what it satisfies by shape. An ABC here bought nominal
inheritance for two test doubles and cost a `cast` in `StaticTreeSolver`, which
had to lie about its own storage to pass it to a base class that did not
describe it.

The split is the one real distinction: **counting is universal, keying is not.**
`StaticArrayStorage` addresses a row by ``(node_id, bucket)`` and deliberately
has no key-addressed lookup -- reintroducing one would restore the string
hashing the static design removes, which
`tests/engine/solver/mccfr/test_static_storage_contract.py` asserts against.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from src.core.game.actions import Action
    from src.engine.solver.infoset.model import InfoSet, InfoSetKey


class CountsInfosets(Protocol):
    """The whole of what :class:`MCCFRSolver` itself asks of storage.

    Both backends satisfy it, which is what lets the solver be generic over
    storage instead of casting.
    """

    def num_infosets(self) -> int: ...


class KeyedStorage(CountsInfosets, Protocol):
    """Key-addressed storage: the small-game oracle path, and nothing shipped.

    `traversal.keyed_infoset_context` is the only reader, reached only through
    the generic `MCCFRSolver.lookup_infoset` that `StaticTreeSolver` overrides.
    Production never takes this path.
    """

    def get_or_create_infoset(self, key: InfoSetKey, legal_actions: Sequence[Action]) -> InfoSet: ...

    def get_infoset(self, key: InfoSetKey) -> InfoSet | None: ...

    def iter_infosets(self) -> Iterable[InfoSet]: ...
