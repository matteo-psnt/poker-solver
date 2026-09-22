"""Test helpers for poker solver tests."""

import enum
import types
from pathlib import Path
from typing import Any, Literal, Union, get_args, get_origin

from pydantic import BaseModel

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action
from src.core.game.rules import GameRules
from src.core.game.state import Card, GameState
from src.engine.search.range_inference import replace_actor_hole_cards
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.mccfr import MCCFRSolver
from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.protocols import BucketingStrategy
from src.engine.solver.storage.static_array import StaticArrayStorage
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


class DummyCardAbstraction:
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


def seed_eval_document(run_dir: Any, document: dict[str, Any], slug: str) -> Any:
    """Write one ``evals/<slug>.json`` the way the pre-sink corpus was written.

    Nothing writes that layout any more -- `record_evaluation` hands the sink a
    document and the last file writer is gone -- but `rebuild_ledger` still
    READS it, and 2,238 historical documents are what it reads. A reader whose
    only corpus came from a writer that no longer exists needs the corpus laid
    down here instead.
    """
    evals_dir = Path(run_dir) / "evals"
    evals_dir.mkdir(parents=True, exist_ok=True)
    path = evals_dir / f"{slug}.json"
    record_store.write_snapshot(path, document, record_store.REGISTRY["evals/*.json"])
    return path


def example_of[M: BaseModel](model: type[M], **overrides: Any) -> M:
    """An instance of a payload model, built from its own field declarations.

    Required fields get the smallest value of their type; optional ones keep
    their defaults; a list or dict of models holds ONE member, so a renderer
    that iterates prints a row. Overrides pin whatever a test asserts on. This
    replaces a hand-written fixture that every new command had to be added to.
    """
    values: dict[str, Any] = {}
    for name, field in model.model_fields.items():
        if name in overrides or not field.is_required():
            continue
        values[name] = _example_value(field.annotation, name)
    return model(**values, **overrides)


def _example_value(annotation: Any, name: str) -> Any:
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin is Literal:
        return args[0]
    if origin in (types.UnionType, Union):
        inner = [arg for arg in args if arg is not type(None)]
        return None if len(inner) < len(args) else _example_value(inner[0], name)
    if origin in (list, tuple, set, frozenset):
        item = args[0] if args else Any
        members = [_example_value(item, name)] if _is_model(item) else []
        return origin(members)
    if origin is dict:
        key, value = args if args else (str, Any)
        if _is_model(value):
            return {_example_value(key, name): _example_value(value, name)}
        return {}
    if _is_model(annotation):
        return example_of(annotation)
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return next(iter(annotation))
    if annotation is bool:
        return False
    if annotation is int:
        return 1
    if annotation is float:
        return 1.0
    if annotation is str:
        return f"{name}-1"
    if annotation is Path:
        return Path(name)
    return None


def _is_model(annotation: Any) -> bool:
    return isinstance(annotation, type) and issubclass(annotation, BaseModel)
