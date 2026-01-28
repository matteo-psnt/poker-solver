"""Test helpers for poker solver tests."""

from typing import Any

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action
from src.core.game.state import Card, GameState
from src.engine.search.range_inference import replace_actor_hole_cards
from src.engine.solver.infoset_encoder import encode_infoset_key
from src.engine.solver.mccfr import MCCFRSolver
from src.engine.solver.storage.shared_array import SharedArrayStorage
from src.pipeline.abstraction.base import BucketingStrategy
from src.shared.config import Config, StorageConfig


def build_test_storage(session_id: str = "test", **overrides: Any) -> SharedArrayStorage:
    """A SharedArrayStorage carrying the declared ``StorageConfig`` defaults.

    The zarr knobs are required at the constructor, but tests that never tune
    checkpointing shouldn't restate their values — that would just re-scatter the
    literals ``StorageConfig`` exists to own. Take them from it instead.
    """
    defaults = StorageConfig()
    overrides.setdefault("num_workers", 1)
    overrides.setdefault("worker_id", 0)
    overrides.setdefault("is_coordinator", True)
    return SharedArrayStorage(
        session_id=session_id,
        zarr_compression_level=defaults.zarr_compression_level,
        zarr_chunk_size=defaults.zarr_chunk_size,
        **overrides,
    )


def build_trained_test_solver(
    iterations: int,
    *,
    starting_stack: int = 400,
    session_id: str = "test-solver",
    **config_overrides,
):
    """A minimally trained (deliberately weak) blueprint on shared-array storage.

    Training is seeded (config seed=42) so repeated builds are strategy-identical;
    ``session_id`` only names the backing shared memory — pass a unique one when
    solvers are rebuilt inside parallel worker processes.
    """
    config = make_test_config(
        seed=42,
        small_blind=50,
        big_blind=100,
        starting_stack=starting_stack,
        **config_overrides,
    )
    storage = build_test_storage(session_id)
    solver = MCCFRSolver(ActionModel(config), DummyCardAbstraction(), storage, config=config)
    for _ in range(iterations):
        solver.train_iteration()
    return solver


def make_test_config(**overrides) -> Config:
    """
    Create a Config object for tests with optional overrides.

    Examples:
        make_test_config(seed=42)
        make_test_config(seed=42, sampling_method="outcome")
        make_test_config(starting_stack=100)
    """
    # Map common shorthand overrides to nested dict structure
    shorthand_map = {
        "seed": ("system", "seed"),
        "starting_stack": ("game", "starting_stack"),
        "small_blind": ("game", "small_blind"),
        "big_blind": ("game", "big_blind"),
        "sampling_method": ("solver", "sampling_method"),
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
    the array is a view into shared-array storage, so later blueprint lookups see
    it). Observing ``action`` then provably up-weights that hand class in range
    inference, with no training. Tiny trained test blueprints are near-uniform,
    which gives a Bayes update nothing to grip.
    """
    hypo = replace_actor_hole_cards(state, actor=actor, combo=combo)
    key = encode_infoset_key(hypo, actor, blueprint.card_abstraction)
    legal = blueprint.rules.get_legal_actions(hypo, action_model=blueprint.action_model)
    infoset = blueprint.storage.get_or_create_infoset(key, legal)
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
