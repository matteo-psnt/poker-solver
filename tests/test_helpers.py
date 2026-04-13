"""Test helpers for poker solver tests."""

from typing import Any

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action
from src.core.game.rules import GameRules
from src.core.game.state import Card, GameState
from src.engine.search.range_inference import replace_actor_hole_cards
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.mccfr import MCCFRSolver
from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.pipeline.abstraction.base import BucketingStrategy
from src.shared import records as record_store
from src.shared.config import Config


def build_test_solver(
    config: Config | None = None,
    card_abstraction: BucketingStrategy | None = None,
    *,
    checkpoint_dir: Any = None,
) -> tuple[StaticTreeSolver, StaticArrayStorage]:
    """A solver and its storage over a tree enumerated for ``config``.

    Tree, table and solver are built together because on the static backend they
    are not independently meaningful: the table's shape IS the tree's row layout,
    so a storage built against a different tree cannot be attached to this solver.
    That is why this returns a pair rather than offering a bare-storage helper —
    there is no longer such a thing as storage you can build without a tree.
    """
    config = config or make_test_config()
    abstraction = card_abstraction or DummyCardAbstraction()
    action_model = ActionModel(config)
    tree = build_betting_tree(
        GameRules(config.game.small_blind, config.game.big_blind),
        action_model,
        abstraction,
        starting_stack=config.game.starting_stack,
    )
    storage = StaticArrayStorage(tree)
    solver = StaticTreeSolver(
        action_model, abstraction, storage, config, tree=tree, checkpoint_dir=checkpoint_dir
    )
    return solver, storage


def build_trained_test_solver(
    iterations: int,
    *,
    starting_stack: int = 400,
    **config_overrides,
) -> StaticTreeSolver:
    """A minimally trained (deliberately weak) blueprint on the static tree.

    Training is seeded (config seed=42), so repeated builds are
    strategy-identical and a test can rebuild one instead of sharing it.
    """
    config = make_test_config(
        seed=42,
        small_blind=50,
        big_blind=100,
        starting_stack=starting_stack,
        **config_overrides,
    )
    solver, _ = build_test_solver(config)
    for _ in range(iterations):
        solver.train_iteration()
    return solver


def make_test_config(**overrides) -> Config:
    """
    Create a Config object for tests with optional overrides.

    Examples:
        make_test_config(seed=42)
        make_test_config(starting_stack=100)
    """
    # Map common shorthand overrides to nested dict structure
    shorthand_map = {
        "seed": ("system", "seed"),
        "starting_stack": ("game", "starting_stack"),
        "small_blind": ("game", "small_blind"),
        "big_blind": ("game", "big_blind"),
        "cfr_plus": ("solver", "cfr_plus"),
        "iteration_weighting": ("solver", "iteration_weighting"),
        # DCFR parameters
        "dcfr_alpha": ("solver", "dcfr_alpha"),
        "dcfr_beta": ("solver", "dcfr_beta"),
        "dcfr_gamma": ("solver", "dcfr_gamma"),
        # Pruning parameters
        "enable_pruning": ("solver", "enable_pruning"),
        "pruning_threshold": ("solver", "pruning_threshold"),
        "prune_start_iteration": ("solver", "prune_start_iteration"),
        "prune_reactivate_frequency": ("solver", "prune_reactivate_frequency"),
    }

    # Build nested dict from overrides
    nested: dict[str, dict[str, Any]] = {}
    for key, value in overrides.items():
        if key in shorthand_map:
            section, field = shorthand_map[key]
            if section not in nested:
                nested[section] = {}
            nested[section][field] = value
        else:
            # Assume it's already a section.field format or top-level
            parts = key.split(".")
            if len(parts) == 2:
                section, field = parts
                if section not in nested:
                    nested[section] = {}
                nested[section][field] = value
            else:
                nested[key] = value

    return Config.default().merge(nested) if nested else Config.default()


def skew_preflop_infoset(
    blueprint: MCCFRSolver,
    state: GameState,
    *,
    actor: int,
    combo: tuple[Card, Card],
    action: Action,
) -> None:
    """Force the blueprint to play ``action`` with certainty for one hand class.

    Manufactures the preflop infoset ``actor`` would hold with ``combo`` and puts
    all average-strategy mass on ``action`` (an in-place ``strategy_sum`` write —
    the array is a view into the static table, so later blueprint lookups see
    it). Observing ``action`` then provably up-weights that hand class in range
    inference, with no training. Tiny trained test blueprints are near-uniform,
    which gives a Bayes update nothing to grip.
    """
    hypo = replace_actor_hole_cards(state, actor=actor, combo=combo)
    infoset, _, _, _ = blueprint.lookup_infoset(hypo, actor)
    infoset.strategy_sum[:] = 0.0
    infoset.strategy_sum[infoset.legal_actions.index(action)] = 1.0


class DummyCardAbstraction(BucketingStrategy):
    """
    Minimal card abstraction for testing.

    All hands map to bucket 0 (single bucket per street).
    Used when card abstraction logic isn't being tested.
    """

    def get_bucket(self, hole_cards, board, street):
        """All hands map to bucket 0."""
        return 0

    def num_buckets(self, street):
        """Single bucket per street."""
        return 1


def seed_ledger(path: Any, *rows: dict[str, Any]) -> None:
    """Write a derived eval index containing exactly ``rows``.

    Production never appends to an index -- `record_evaluation` writes one
    document per eval and `rebuild_ledger` derives the index from those on every
    read. Reader tests still need an index to read, and building one through a
    full rebuild would test the rebuild rather than the reader, so this writes
    the rows directly through the same record substrate.
    """
    for row in rows:
        record_store.append_log(path, row, record_store.REGISTRY["eval_ledger.jsonl"])
