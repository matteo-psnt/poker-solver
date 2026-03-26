"""
Shared builder functions for training components.

Provides centralized, reusable functions for building solver components
(abstractions, storage, solver) from configuration. Used by TrainingSession
to eliminate code duplication.
"""

from pathlib import Path

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import load_checkpoint
from src.pipeline.abstraction.base import BucketingStrategy
from src.pipeline.abstraction.postflop.precompute import PostflopPrecomputer
from src.pipeline.training.abstraction_resolver import ComboAbstractionResolver
from src.shared.config import Config


def build_card_abstraction(
    config: Config,
    abstractions_dir: Path | None = None,
    abstraction_hash: str | None = None,
) -> BucketingStrategy:
    """
    Build card abstraction from config.

    Uses combo-level abstraction with suit isomorphism for correct postflop bucketing.

    Args:
        config: Configuration object
        abstractions_dir: Optional directory containing precomputed abstractions
        abstraction_hash: Optional exact abstraction config hash to pin resolution to.
            Required to faithfully evaluate a checkpoint whose abstraction has since
            been recomputed under the same config name.

    Returns:
        BucketingStrategy instance (DenseBucketer)

    Raises:
        ValueError: If config is invalid
        FileNotFoundError: If abstraction file doesn't exist
    """
    resolver = ComboAbstractionResolver(
        abstractions_dir=abstractions_dir,
        loader=PostflopPrecomputer.load,
    )
    return resolver.load(
        abstraction_config=config.card_abstraction.config,
        abstraction_hash=abstraction_hash,
    )


def resolve_card_abstraction_hash(
    config: Config,
    abstractions_dir: Path | None = None,
) -> str | None:
    """Config hash of the abstraction ``config`` currently resolves to.

    Recorded on a run so evaluation can pin the exact abstraction it trained against.
    """
    resolver = ComboAbstractionResolver(
        abstractions_dir=abstractions_dir,
        loader=PostflopPrecomputer.load,
    )
    return resolver.resolved_hash(abstraction_config=config.card_abstraction.config)


def build_static_evaluation_solver(
    config: Config,
    *,
    checkpoint_dir: Path,
    abstractions_dir: Path | None = None,
    abstraction_hash: str | None = None,
    at_iteration: int | None = None,
) -> tuple[StaticTreeSolver, StaticArrayStorage]:
    """Build a read-only blueprint over a STATIC checkpoint.

    ``session_id=None`` allocates process-local arrays rather than shared memory:
    an evaluation is a single read-only process, and taking a named segment would
    collide with a training run of the same id.
    """
    action_model = ActionModel(config)
    card_abstraction = build_card_abstraction(
        config,
        abstractions_dir=abstractions_dir,
        abstraction_hash=abstraction_hash,
    )
    rules = GameRules(config.game.small_blind, config.game.big_blind)
    tree = build_betting_tree(
        rules, action_model, card_abstraction, starting_stack=config.game.starting_stack
    )
    storage = StaticArrayStorage(tree)
    # Verifies the tree fingerprint, so a checkpoint written against a different
    # tree is refused rather than silently reinterpreted row-for-row.
    load_checkpoint(storage, checkpoint_dir, at_iteration=at_iteration)
    solver = StaticTreeSolver(action_model, card_abstraction, storage, config, tree=tree)
    return solver, storage
