"""MCCFR over a statically enumerated betting tree.

Same regret math, same sampling, same averaging as :class:`MCCFRSolver` — the
only thing that changes is how an infoset is *found*. That difference is the
whole point, so it is worth being precise about what disappears from the hot
path here:

    string key construction      normalized betting sequence + hand label
    key hashing + dict lookup    one xxhash-owner probe and one dict get
    id allocation / reconcile    absent by construction
    get_legal_actions            the tree already recorded the exact list
    filter_stored_actions        stored and live actions can no longer disagree

What remains is two integer lookups and a slice.

Two fields of the old key are gone as well, both verified redundant (see
``infoset_index``): ``spr_bucket``, a pure function of the betting sequence, and
``player_position``, which duplicated the entire infoset space because the tree
is button-symmetric. Dropping the latter halves the number of infosets the same
iteration budget has to fill — a direct 2x on updates per infoset, which is the
quantity that was starving.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import cast

import numpy as np

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action
from src.core.game.rules import GameRules
from src.core.game.state import GameState
from src.engine.solver.betting_tree import BettingTree, build_betting_tree
from src.engine.solver.infoset import InfoSet
from src.engine.solver.infoset_index import bucket_of
from src.engine.solver.protocols import BucketingStrategy
from src.engine.solver.storage.base import Storage
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import load_checkpoint, save_checkpoint
from src.shared.config import Config

from .solver import MCCFRSolver


class StaticTreeSolver(MCCFRSolver):
    """MCCFR whose infosets are preallocated rows of a static betting tree.

    ``dropped_unknown_id_updates`` stays at zero for the solver's whole lifetime:
    every row exists in every process from the start, so there is no state in
    which a worker knows an infoset but cannot write it. The counter is retained
    (rather than deleted) so metrics consumers keep working and so a nonzero
    value would be an unmissable signal that something reintroduced dynamic
    allocation.
    """

    # ``StaticArrayStorage`` deliberately does not implement the key-addressed
    # ``Storage`` ABC — reintroducing ``InfoSetKey`` here would restore the string
    # hashing this design removes. The base class only touches ``self.storage``
    # in ``checkpoint`` and ``num_infosets``, both overridden below, so nothing
    # reaches the ABC surface through this attribute.
    storage: StaticArrayStorage

    def __init__(
        self,
        action_model: ActionModel,
        card_abstraction: BucketingStrategy,
        storage: StaticArrayStorage,
        config: Config,
        *,
        tree: BettingTree | None = None,
        checkpoint_dir: Path | None = None,
        checkpoint_retain_every: int = 0,
        abstraction_id: str | None = None,
    ):
        super().__init__(action_model, card_abstraction, cast("Storage", storage), config)
        self.storage = storage
        self.tree = tree if tree is not None else storage.tree
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        # Retention is a property of the RUN, not of an individual call: a leg
        # that forgot to pass it would silently stop sparing measurement points.
        self.checkpoint_retain_every = checkpoint_retain_every
        # Identifies the bucket ASSIGNMENT, which the tree fingerprint cannot:
        # two abstractions with the same per-street counts produce an identical
        # fingerprint while mapping hands to different buckets.
        self.abstraction_id = abstraction_id

    @classmethod
    def build(
        cls,
        action_model: ActionModel,
        card_abstraction: BucketingStrategy,
        config: Config,
        *,
        session_id: str | None = None,
    ) -> StaticTreeSolver:
        """Enumerate the tree, allocate storage to fit it, and wire up a solver."""
        rules = GameRules(config.game.small_blind, config.game.big_blind)
        tree = build_betting_tree(
            rules,
            action_model,
            card_abstraction,
            starting_stack=config.game.starting_stack,
        )
        storage = StaticArrayStorage(tree, session_id=session_id)
        return cls(action_model, card_abstraction, storage, config, tree=tree)

    def lookup_infoset(
        self, state: GameState, current_player: int
    ) -> tuple[InfoSet, Sequence[Action], list[int], np.ndarray]:
        """Two integer lookups and a slice — no key, no hash, no allocation."""
        node_id = self.tree.node_id(state)
        bucket = bucket_of(state, current_player, self.card_abstraction)
        infoset = self.storage.infoset_at(node_id, bucket)

        legal_actions = infoset.legal_actions
        # The tree recorded this node's action list at enumeration time from the
        # same action model, so stored and live actions are the same objects in
        # the same order. The dynamic backend had to reconcile them because a
        # checkpoint could carry an action list from a different abstraction;
        # here a mismatched abstraction produces a different tree and is caught
        # at construction instead.
        valid_indices = _identity_index_list(len(legal_actions))
        strategy = infoset.get_filtered_strategy(valid_indices=valid_indices, use_average=False)
        return infoset, legal_actions, valid_indices, strategy

    def num_infosets(self) -> int:
        """Rows actually visited, not tree capacity.

        Capacity is a config constant here, so reporting it as "infosets" would
        turn a progress metric into a no-op. Visited rows preserve the meaning
        the dynamic backend's count had.
        """
        return self.storage.num_touched_infosets()

    def checkpoint(self) -> None:
        """Write a snapshot of the flat arrays and publish it atomically.

        ``checkpoint_retain_every`` spares one snapshot per band from pruning,
        which is what makes a within-run exploitability curve computable after
        the run ends. Without it a run finishes holding exactly one checkpoint —
        the reason no such curve has ever existed here.
        """
        if self.checkpoint_dir is None:
            raise ValueError(
                "StaticTreeSolver has no checkpoint_dir; pass one at construction "
                "to enable checkpointing."
            )
        save_checkpoint(
            self.storage,
            self.checkpoint_dir,
            self.iteration,
            retain_every=self.checkpoint_retain_every,
            abstraction_id=self.abstraction_id,
        )

    def restore(self, *, at_iteration: int | None = None) -> int:
        """Load a snapshot into this solver's storage and adopt its iteration."""
        if self.checkpoint_dir is None:
            raise ValueError("StaticTreeSolver has no checkpoint_dir to restore from.")
        self.iteration = load_checkpoint(
            self.storage,
            self.checkpoint_dir,
            at_iteration=at_iteration,
            abstraction_id=self.abstraction_id,
        )
        return self.iteration

    def __str__(self) -> str:
        return (
            f"StaticTreeSolver(iteration={self.iteration}, "
            f"visited={self.num_infosets():,}/{self.tree.num_rows:,} rows, "
            f"nodes={len(self.tree):,})"
        )


# Identity index lists are read-only by convention and shared across calls, so
# the hot path never allocates one. Sized on demand; there are only a handful of
# distinct action counts.
_IDENTITY_LISTS: dict[int, list[int]] = {}


def _identity_index_list(num_actions: int) -> list[int]:
    indices = _IDENTITY_LISTS.get(num_actions)
    if indices is None:
        indices = _IDENTITY_LISTS.setdefault(num_actions, list(range(num_actions)))
    return indices
