"""How a consumer asks "what is the blueprint's infoset here?".

A blueprint is addressed by ``(node_id, bucket)`` — an index into the betting
tree, with no keys anywhere.

Evaluation code should not have to know that. Before this seam the exact-BR
engine built infoset keys inline, which put a solver concern inside the
evaluation layer and hard-wired that layer to one storage layout; changing the
layout meant editing every scorer. A policy source moves it back behind one
method, and is why the scorers survived the storage layout being replaced
underneath them.

The seam is deliberately ``(state, bucket) -> InfoSet`` rather than
``-> distribution``: consumers already own the filtering and fallback policy via
``policy_lookup.blueprint_action_distribution``, and duplicating that decision
per backend is exactly how the "every consumer picks its own restriction"
problem that module was written to end would come back.
"""

from __future__ import annotations

from collections.abc import Hashable
from typing import Protocol

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action
from src.core.game.rules import GameRules
from src.core.game.state import GameState, Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.infoset import InfoSet
from src.engine.solver.infoset_index import (
    bucket_of,
)
from src.engine.solver.protocols import BucketingStrategy
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.shared.config import Config


class ScorableBlueprint(Protocol):
    """The minimum an evaluator needs from a blueprint.

    Declares ``policy_source``, not ``storage``. Every consumer here wants to
    ask "what does the blueprint play at this state", and routing that through
    the source rather than the table is what keeps them from re-implementing
    infoset addressing -- the habit that once hard-wired the exact-BR engine to
    one backend.
    """

    action_model: ActionModel
    card_abstraction: BucketingStrategy
    rules: GameRules
    config: Config
    policy_source: PolicySource

    def sample_action_from_strategy(self, state: GameState, *, use_average: bool = True) -> Action:
        """Sample an action from the blueprint policy at ``state``."""
        ...

    def is_chance_node(self, state: GameState) -> bool:
        """Whether ``state`` awaits a board card rather than a player action."""
        ...

    def sample_chance_outcome(self, state: GameState) -> GameState:
        """Deal the pending street and return the advanced state."""
        ...


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

    def infoset_for(self, state: GameState, player: int) -> InfoSet | None:
        """``player``'s stored infoset at ``state``, bucketing their real hand.

        The common runtime case: consumers that hold a concrete hand rather than
        a bucket to enumerate.
        """
        ...

    def identity(self, state: GameState, player: int) -> Hashable:
        """An opaque, hashable identifier for that infoset — for caching only.

        Deliberately not the bucket. Several consumers memoize ACROSS states, and
        a bucket alone omits the street and betting sequence, so two unrelated
        nodes sharing a bucket would collide in a cross-call memo. Comparable
        only within one policy source; never persist it.
        """
        ...

    def bucket_for(self, state: GameState, player: int) -> int:
        """The card bucket ``player``'s actual hole cards fall in at ``state``.

        Runtime consumers (the resolver, the heads-up session, range inference)
        have a concrete hand rather than a bucket to enumerate, so they need this
        to reach :meth:`infoset_at`. It doubles as their cache key: for a fixed
        state the bucket is the only part of infoset identity that varies with
        the hand, which is exactly what those caches were keying on when they
        built a whole ``InfoSetKey`` to hash.
        """
        ...


class TreePolicySource:
    """Policy source over the tree-addressed (static) backend.

    An off-tree state raises rather than returning ``None``: the tree covers
    every state its config admits, so a miss means the caller is evaluating a
    different game than the blueprint was trained on — which a uniform fallback
    would quietly average away into a plausible-looking score.
    """

    def __init__(
        self,
        tree: BettingTree,
        storage: StaticArrayStorage,
        card_abstraction: BucketingStrategy,
    ):
        self._tree = tree
        self._storage = storage
        self._abstraction = card_abstraction

    def num_buckets(self, street: Street) -> int:
        return self._tree.num_buckets(street)

    def bucket_for(self, state: GameState, player: int) -> int:
        return bucket_of(state, player, self._abstraction)

    def infoset_for(self, state: GameState, player: int) -> InfoSet | None:
        return self.infoset_at(state, self.bucket_for(state, player))

    def identity(self, state: GameState, player: int) -> Hashable:
        return (self._tree.node_id(state), self.bucket_for(state, player))

    def infoset_at(self, state: GameState, bucket: int) -> InfoSet | None:
        node_id = self._tree.node_id(state)
        if not 0 <= bucket < self._tree.buckets_per_node[node_id]:
            return None
        # view(), not infoset_at(): evaluation must not mark coverage, or a
        # scoring pass would report an untrained tree as fully explored.
        infoset = self._storage.view(node_id, bucket)
        # An UNVISITED row is not an answer, it is an allocation. The static
        # table holds every row from the start, so without this check a caller
        # asking "does the blueprint have a policy here?" is always told yes --
        # and the fallback-mass diagnostic, which exists to reveal exactly how
        # much of a score came from untrained regions, silently reads zero on
        # every run. Numerically this changes nothing (a zeroed row already
        # yields the uniform distribution the caller falls back to); it restores
        # the caller's ability to KNOW that is what happened.
        if not self._storage.visited[infoset.row]:
            return None
        return infoset


__all__ = (
    "PolicySource",
    "ScorableBlueprint",
    "TreePolicySource",
)
