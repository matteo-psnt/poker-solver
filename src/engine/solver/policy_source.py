"""How a consumer asks "what is the blueprint's infoset here?".

The two storage backends answer that question in incompatible ways. The dynamic
backend is addressed by a hashed :class:`InfoSetKey`; the static backend is
addressed by ``(node_id, bucket)`` and has no keys at all — deliberately, since
the string hashing a key exists to support is the cost that design removes.

Evaluation code should not have to know which. Before this seam, the exact-BR
engine constructed ``InfoSetKey`` objects inline, which both hard-wired it to
the dynamic backend and put key construction — a solver concern — inside the
evaluation layer. A policy source moves that back behind one method.

The seam is deliberately ``(state, bucket) -> InfoSet`` rather than
``-> distribution``: consumers already own the filtering and fallback policy via
``policy_lookup.blueprint_action_distribution``, and duplicating that decision
per backend is exactly how the "every consumer picks its own restriction"
problem that module was written to end would come back.
"""

from __future__ import annotations

from typing import Protocol

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import GameState, Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.infoset import InfoSet, InfoSetKey
from src.engine.solver.infoset_encoder import get_spr_bucket
from src.engine.solver.infoset_index import NUM_PREFLOP_HANDS, preflop_hand_string_at
from src.engine.solver.protocols import BucketingStrategy
from src.engine.solver.storage.base import Storage
from src.engine.solver.storage.static_array import StaticArrayStorage


class ScorableBlueprint(Protocol):
    """The minimum an evaluator needs from a blueprint.

    Deliberately narrower than :class:`~src.engine.solver.protocols.Blueprint`,
    and in particular it does NOT declare ``storage``. The two backends type
    that attribute incompatibly — ``Storage`` vs ``StaticArrayStorage`` — and a
    mutable protocol attribute cannot be satisfied by both. Since evaluation
    reaches policy through :func:`policy_source_for`, which dispatches on the
    concrete backend, the attribute never needs to appear in the contract.
    """

    action_model: ActionModel
    card_abstraction: BucketingStrategy
    rules: GameRules


class PolicySource(Protocol):
    """Resolves a stored infoset from a public state and a card bucket."""

    def num_buckets(self, street: Street) -> int:
        """Buckets on ``street`` — the range a consumer may enumerate."""
        ...

    def infoset_at(self, state: GameState, bucket: int) -> InfoSet | None:
        """The acting player's stored infoset at ``state`` holding ``bucket``.

        ``None`` means the blueprint has no entry here (untrained or off-tree);
        the caller owns the fallback.
        """
        ...


class KeyedPolicySource:
    """Policy source over the key-addressed (dynamic) backend."""

    def __init__(self, storage: Storage, card_abstraction: BucketingStrategy):
        self._storage = storage
        self._abstraction = card_abstraction

    def num_buckets(self, street: Street) -> int:
        if street == Street.PREFLOP:
            return NUM_PREFLOP_HANDS
        return self._abstraction.num_buckets(street)

    def infoset_at(self, state: GameState, bucket: int) -> InfoSet | None:
        spr = min(state.stacks) / state.pot if state.pot > 0 else 0
        preflop = state.street == Street.PREFLOP
        key = InfoSetKey(
            player_position=state.current_player,
            street=state.street,
            betting_sequence=state.normalized_betting_sequence(),
            preflop_hand=preflop_hand_string_at(bucket) if preflop else None,
            postflop_bucket=None if preflop else bucket,
            spr_bucket=get_spr_bucket(spr),
        )
        return self._storage.get_infoset(key)


class TreePolicySource:
    """Policy source over the tree-addressed (static) backend.

    An off-tree state raises rather than returning ``None``: the tree covers
    every state its config admits, so a miss means the caller is evaluating a
    different game than the blueprint was trained on — which a uniform fallback
    would quietly average away into a plausible-looking score.
    """

    def __init__(self, tree: BettingTree, storage: StaticArrayStorage):
        self._tree = tree
        self._storage = storage

    def num_buckets(self, street: Street) -> int:
        return self._tree.num_buckets(street)

    def infoset_at(self, state: GameState, bucket: int) -> InfoSet | None:
        node_id = self._tree.node_id(state)
        if not 0 <= bucket < self._tree.buckets_per_node[node_id]:
            return None
        # view(), not infoset_at(): evaluation must not mark coverage, or a
        # scoring pass would report an untrained tree as fully explored.
        return self._storage.view(node_id, bucket)


def policy_source_for(blueprint: object) -> PolicySource:
    """Pick the policy source matching a blueprint's storage backend.

    Typed ``object`` because the dispatch IS on the concrete backend; declaring
    a ``storage`` attribute here would reintroduce the incompatibility
    ``ScorableBlueprint`` exists to avoid.
    """
    storage = getattr(blueprint, "storage")
    if isinstance(storage, StaticArrayStorage):
        tree = getattr(blueprint, "tree", None) or storage.tree
        return TreePolicySource(tree, storage)
    return KeyedPolicySource(storage, getattr(blueprint, "card_abstraction"))


__all__ = (
    "KeyedPolicySource",
    "PolicySource",
    "ScorableBlueprint",
    "TreePolicySource",
    "policy_source_for",
)
