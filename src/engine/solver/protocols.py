"""Solver-facing protocols shared across engine modules."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from src.core.game.state import Card, Street

if TYPE_CHECKING:
    from src.core.actions.action_model import ActionModel
    from src.core.game.actions import Action
    from src.core.game.rules import GameRules
    from src.core.game.state import GameState
    from src.engine.solver.storage.base import Storage
    from src.shared.config import Config


class BucketingStrategy(Protocol):
    """Structural interface for card bucketing used by the solver."""

    def get_bucket(
        self, hole_cards: tuple[Card, Card], board: tuple[Card, ...], street: Street
    ) -> int:
        """Map hole cards + board context to an abstract bucket id."""

    def num_buckets(self, street: Street) -> int:
        """Return number of buckets for a specific street."""


class Blueprint(Protocol):
    """A trained strategy queryable at runtime.

    This is the contract every blueprint consumer — the subgame resolver, range
    inference, the LBR opponent models, the resolver gate — actually relies on;
    it was previously implicit (parameters typed ``object``), discoverable only
    by reading call sites, and silently breakable by any solver refactor. The
    only production implementation is ``MCCFRSolver``, which satisfies it
    structurally; tests may substitute lighter fakes.

    Consumers read strategies via ``storage.get_infoset`` keyed through
    ``card_abstraction`` (see ``policy_lookup`` for the canonical lookup), and
    use the game members to enumerate/validate actions and walk chance nodes
    with the blueprint's own dealing semantics.
    """

    action_model: ActionModel
    card_abstraction: BucketingStrategy
    storage: Storage
    rules: GameRules
    config: Config

    def sample_action_from_strategy(self, state: GameState, *, use_average: bool = True) -> Action:
        """Sample an action from the blueprint policy at ``state``."""
        ...

    def is_chance_node(self, state: GameState) -> bool:
        """Whether ``state`` awaits a board card rather than a player action."""
        ...

    def sample_chance_outcome(self, state: GameState) -> GameState:
        """Deal the pending street and return the advanced state."""
        ...
