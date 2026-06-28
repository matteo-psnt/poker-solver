"""Construction of a blueprint -- abstraction, betting tree, storage, solver.

One place that turns a config (plus, for a trained one, a checkpoint) into the
objects a blueprint is made of, so every consumer builds the same objects the
same way rather than assembling its own.

**This sits beside `training/`, not inside it.** It lived under `training` while
training was the only thing that built a solver, but the consumers are now
training, evaluation and anything that serves a run for reading -- and reading
outnumbers training here. Filing the constructor under one consumer made the
others reach across the `training`/`evaluation` independence contract to get at
it, which is exactly the coupling that contract exists to prevent.
"""

from pathlib import Path
from typing import Any

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.protocols import BucketingStrategy
from src.engine.solver.storage.policy_assembly import PolicyIterate, assemble_policy
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import load_checkpoint
from src.pipeline.abstraction.postflop.precompute import PostflopPrecomputer
from src.pipeline.abstraction.resolver import ComboAbstractionResolver
from src.shared.config import Config


def build_card_abstraction(
    config: Config,
    abstractions_dir: Path | None = None,
    abstraction_hash: str | None = None,
) -> BucketingStrategy:
    """Build the card abstraction from config: combo-level, with suit isomorphism.

    ``abstraction_hash`` pins resolution to one exact abstraction config hash, which
    is required to faithfully evaluate a checkpoint whose abstraction has since been
    recomputed under the same config name.

    Raises:
        ValueError: If config is invalid.
        FileNotFoundError: If the abstraction file does not exist.
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
    policy_iterate: PolicyIterate = "average",
    avg_window_from: int | None = None,
    avg_gamma: float | None = None,
) -> tuple[StaticTreeSolver, StaticArrayStorage, dict[str, Any]]:
    """Build a read-only blueprint over a STATIC checkpoint.

    ``session_id=None`` allocates process-local arrays rather than shared memory:
    an evaluation is a single read-only process, and taking a named segment would
    collide with a training run of the same id.

    The third return value is what the eval record needs to tell a non-default
    ``policy_iterate``/``avg_window_from`` apart from the plain average.
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
    loaded = load_checkpoint(storage, checkpoint_dir, at_iteration=at_iteration)
    policy_record = assemble_policy(
        storage,
        checkpoint_dir,
        iterate=policy_iterate,
        window_from=avg_window_from,
        avg_gamma=avg_gamma,
        source_gamma=config.solver.dcfr_gamma,
        loaded_iteration=loaded,
    )
    solver = StaticTreeSolver(action_model, card_abstraction, storage, config, tree=tree)
    return solver, storage, policy_record
