"""Monte Carlo Counterfactual Regret Minimization (MCCFR) solver."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from src.core.game.rules import GameRules
from src.engine.solver.infoset import encoder

from . import chance, traversal

if TYPE_CHECKING:
    from src.core.actions.action_model import ActionModel
    from src.core.game.state import GameState
    from src.engine.solver.infoset.model import InfoSetKey
    from src.engine.solver.protocols import BucketingStrategy
    from src.engine.solver.storage.base import CountsInfosets
    from src.shared.config import Config


class MCCFRSolver[StorageT: CountsInfosets]:
    """
    Monte Carlo CFR with external sampling.

    External sampling (default):
    - Explores all actions for traversing player
    - Samples single action for opponent
    - Samples chance outcomes

    Outcome sampling:
    - Samples single action for all players
    - Samples chance outcomes
    - Faster but higher variance
    """

    def __init__(
        self,
        action_model: ActionModel,
        card_abstraction: BucketingStrategy,
        storage: StorageT,
        config: Config,
    ):
        self.action_model = action_model
        self.card_abstraction = card_abstraction
        self.storage = storage
        self.config = config

        self.iteration = 0
        # Updates skipped because the infoset's global ID was not yet known
        # (non-owner placeholder views). Cumulative over the solver's lifetime;
        # workers report per-batch deltas so the drop rate is observable
        # instead of an assumed "brief propagation delay".
        self.dropped_unknown_id_updates = 0
        # Diagnostic counterpart: writable (applied) update sites, so the drop
        # RATE (dropped / (dropped + applied)) is observable, not just the count.
        self.applied_updates = 0
        self.rules = GameRules(self.config.game.small_blind, self.config.game.big_blind)

        # THE LEGACY `np.random` API IS DELIBERATE HERE -- do not "modernise" it
        # to `np.random.Generator` (what ruff's NPY002 would tell you to do).
        # Both streams this solver draws from are MT19937, and
        # `src/engine/solver/numba_random.py` reproduces MT19937 *inside* the
        # numba kernel by round-tripping `np.random.get_state()`, which is what
        # makes the compiled walk bit-identical to the traversal it replaced --
        # the property that let it deploy without re-baselining every published
        # number. `Generator` is PCG64, has no equivalent state hand-off, and
        # would break that module and every baseline at once.
        if self.config.system.seed is not None:
            random.seed(self.config.system.seed)
            np.random.seed(self.config.system.seed)

    def num_infosets(self) -> int:
        """Get total number of infosets discovered."""
        return self.storage.num_infosets()

    def train_iteration(self) -> float:
        """Execute one external-sampling MCCFR iteration."""
        state = self.deal_initial_state()
        traversing_player = self.iteration % 2
        util = self._cfr_external_sampling(state, traversing_player)

        self.iteration += 1
        if traversing_player == 1:
            util = -util
        return util

    def deal_initial_state(self) -> GameState:
        return chance.deal_initial_state(self)

    def is_chance_node(self, state: GameState) -> bool:
        return chance.is_chance_node(state)

    def sample_chance_outcome(self, state: GameState) -> GameState:
        return chance.sample_chance_outcome(state)

    def deal_remaining_cards(self, state: GameState) -> GameState:
        return chance.deal_remaining_cards(state)

    def encode_infoset_key(self, state: GameState, player: int) -> InfoSetKey:
        """The key under which ``player``'s decision at ``state`` is stored.

        Dispatched through the solver rather than called directly by the traversal
        so the CFR kernel can be exercised on games other than HUNL: this encoding
        is HUNL-specific (169 preflop classes, equity buckets, SPR) while the
        regret math below it is not. See
        ``tests/engine/solver/mccfr/extensive_game_solver.py``.
        """
        return encoder.encode_infoset_key(state, player, self.card_abstraction)

    def lookup_infoset(self, state: GameState, current_player: int):
        """Resolve ``current_player``'s infoset at ``state`` for the traversal.

        The identity seam (see ``traversal._infoset_context``). This generic
        implementation hashes an ``InfoSetKey``, which works for any game;
        ``StaticTreeSolver`` overrides it to index the betting tree instead.
        """
        return traversal.keyed_infoset_context(self, state, current_player)

    def _cfr_external_sampling(self, state: GameState, traversing_player: int) -> float:
        return traversal.cfr_external_sampling(self, state, traversing_player)

    def __str__(self) -> str:
        return (
            f"MCCFRSolver(iteration={self.iteration}, infosets={self.num_infosets()}, "
            f"stack={self.config.game.starting_stack})"
        )
