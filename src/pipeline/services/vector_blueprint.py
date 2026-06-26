"""Train a board-free blueprint and store it as an ordinary static checkpoint.

The board-free kernel updates every row of the tree on every iteration, where
external sampling touches 88-95. ``BucketVectorCFR`` and ``StaticArrayStorage``
both store ``regrets``/``strategy_sum`` as flat arrays of ``tree.num_slots``
under the identical ``slot_base[n] + b*slot_stride[n] + a`` layout, so the
strategy is already in the shape a checkpoint wants. This service is that
bridge: train board-free, write a real ``STATIC_CHECKPOINT.json`` ladder, and
``evaluate``, ``curve`` and ``ledger`` all work on it unchanged.

Two things here are load-bearing rather than incidental.

``visited`` must be written. ``TreePolicySource.infoset_at`` returns ``None``
for any row whose ``visited`` flag is false and the caller reads that as *play
uniform*, so a checkpoint from a vector kernel left at zeros scores exactly like
an untrained blueprint -- no error, no warning. The flag is set from the
strategy itself: a row has an answer exactly when it accumulated mass.

**The universe is part of the game.** Chance here is a bucket-to-bucket
transition matrix estimated from sampled boards, so two runs with different
boards are solving DIFFERENT GAMES. ``universe_seed``/``universe_boards`` are
recorded and verified on resume, as the card abstraction hash is.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from pydantic import BaseModel

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Street
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import load_checkpoint, save_checkpoint
from src.engine.solver.vector import bucket_game, compile_tree
from src.engine.solver.vector.bucket_kernel import BucketVectorCFR
from src.pipeline.abstraction.vector_universe import build_universe
from src.pipeline.blueprint import construction
from src.pipeline.training.run_tracker import RunTracker
from src.pipeline.training.run_tracker.attempts import ExperimentTag
from src.shared import records
from src.shared.config.loader import load_training_config
from src.shared.log import configure_logging

PROGRESS_ARTIFACT = "train-progress.json"

if TYPE_CHECKING:
    from src.shared.config import Config

logger = logging.getLogger(__name__)

PREFLOP_BUCKETS = 169
UNIVERSE_FILENAME = "vector_universe.json"


class VectorBlueprintOutput(BaseModel):
    """Portable summary; the CLI renders this and the cloud leg returns it."""

    run_id: str
    runs_dir: str
    config_name: str
    iterations: int
    num_rows: int
    touched_rows: int
    coverage: float
    runtime_seconds: float
    seconds_per_iteration: float
    abstract_exploitability: float
    universe_boards: int
    universe_seed: int
    dtype: str
    status: str


def _verify_universe(run_dir: Path, boards: int, seed: int) -> None:
    """Refuse to continue a run whose chance layer was estimated differently.

    The transition matrices ARE the game. Continuing one run's regrets under
    another run's transitions converges to neither game, and nothing downstream
    could detect it: the tree fingerprint covers node layout and bucket counts,
    not the matrices.
    """
    recorded = json.loads((run_dir / UNIVERSE_FILENAME).read_text())
    if (recorded["boards"], recorded["seed"]) != (boards, seed):
        raise ValueError(
            f"Run was built on {recorded['boards']} boards at seed {recorded['seed']}, "
            f"but this leg asks for {boards} at seed {seed}. The universe defines the "
            "chance layer, so continuing would blend two different games."
        )


def train_vector_blueprint(
    config_name: str,
    *,
    iterations: int,
    universe_boards: int = 2000,
    universe_seed: int = 7,
    checkpoint_every: int = 25,
    retain_every: int = 1,
    dtype: str = "float32",
    config_overrides: dict[str, object] | None = None,
    runs_dir: Path | None = None,
    run_id: str | None = None,
    experiment: ExperimentTag | None = None,
    progress_file: Path | None = None,
) -> VectorBlueprintOutput:
    """Train the board-free kernel to an ABSOLUTE iteration target.

    Absolute, like ``train_static``: continuing past the target is a no-op, so a
    retried cloud leg converges instead of training twice.

    Args:
        config_name: Stem of a config under ``config/training``.
        iterations: ABSOLUTE target. The kernel's real-game quality is U-shaped,
            so this is a budget rather than a convergence criterion — retain
            rungs and score them, then take the minimum.
        universe_boards: Real boards the transition and showdown matrices are
            estimated from. More boards is a better estimate of the same game.
        universe_seed: Draws those boards. Part of the game's identity.
        checkpoint_every: Write a rung every N iterations.
        retain_every: Keep every Nth rung addressable by ``evaluate --at``.
        dtype: ``float32`` (default, measured to cost nothing in strategy
            quality) or ``float64`` (sharper zero-sum residual, 2x memory).
        progress_file: Where to publish iterations done while they are being
            done. A rung is the durable answer, but rungs are ``checkpoint_every``
            apart and an iteration here updates every row of the tree -- so
            between two of them the work is otherwise unobservable.
    """
    config: Config = load_training_config(config_name, **(config_overrides or {}))
    configure_logging(config.system.log_level)

    base_dir = Path(runs_dir) if runs_dir is not None else Path(config.training.runs_dir)
    if run_id is None:
        run_id = f"vec-{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = base_dir / run_id
    resuming = (run_dir / UNIVERSE_FILENAME).exists()

    action_model = ActionModel(config)
    abstraction = construction.build_card_abstraction(config)
    abstraction_hash = construction.resolve_card_abstraction_hash(config)
    rules = GameRules(config.game.small_blind, config.game.big_blind)
    tree = build_betting_tree(
        rules, action_model, abstraction, starting_stack=config.game.starting_stack
    )

    if resuming:
        _verify_universe(run_dir, universe_boards, universe_seed)
        tracker = RunTracker.load(run_dir)
        tracker.verify_action_config_hash(action_model.get_config_hash())
        tracker.mark_resumed()
    else:
        tag = experiment or ExperimentTag()
        tracker = RunTracker(
            run_dir=run_dir,
            config_name=config.system.config_name,
            config=config,
            action_config_hash=action_model.get_config_hash(),
            card_abstraction_hash=abstraction_hash,
            experiment_id=tag.experiment_id,
            arm=tag.arm,
            parent_run_id=tag.parent_run_id,
            kernel="board-free",
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker.initialize()
    (run_dir / UNIVERSE_FILENAME).write_text(
        json.dumps({"boards": universe_boards, "seed": universe_seed}, indent=2)
    )

    started = time.time()
    try:
        compiled = compile_tree(tree, rules)
        rng = np.random.default_rng(universe_seed)
        universe = build_universe(abstraction, universe_boards, rng=rng)
        counts = {
            Street.PREFLOP: PREFLOP_BUCKETS,
            **{s: tree.num_buckets(s) for s in (Street.FLOP, Street.TURN, Street.RIVER)},
        }
        game = bucket_game.derive(universe, counts)
        logger.info(
            "board-free game derived from %d boards; showdown signal max|S|=%.4f",
            universe_boards,
            float(np.abs(game.showdown).max()),
        )

        kernel = BucketVectorCFR(compiled, game, cfr_plus=True, dtype=getattr(np, dtype))
        storage = StaticArrayStorage(tree)
        if resuming:
            kernel.iteration = load_checkpoint(storage, run_dir)
            kernel.regrets[:] = storage.regrets
            kernel.strategy_sum[:] = storage.strategy_sum
            logger.info("resumed board-free run at iteration %d", kernel.iteration)

        initial = np.ones(counts[Street.PREFLOP], dtype=kernel.dtype)
        row_starts = tree.row_slot_starts

        def publish(at: int) -> float:
            storage.regrets[:] = kernel.regrets
            storage.strategy_sum[:] = kernel.strategy_sum
            mass = np.add.reduceat(np.asarray(kernel.strategy_sum, dtype=np.float64), row_starts)
            # Without this every row reads as unvisited and the blueprint plays
            # uniform everywhere. See the module docstring.
            storage.visited[:] = mass > 0.0
            save_checkpoint(
                storage,
                run_dir,
                iteration=at,
                retain_every=retain_every,
                abstraction_id=abstraction_hash,
            )
            return float((mass > 0.0).mean())

        # Per ITERATION, with no timer and no thread: this kernel is one process
        # and one iteration updates every row of the tree, so a small local write
        # beside it is free. The scalar trainer needs both because its work is in
        # other processes and its coordinator is blocked while they run.
        report = records.progress_writer(progress_file, records.REGISTRY[PROGRESS_ARTIFACT])

        coverage = 0.0
        while kernel.iteration < iterations:
            target = min(iterations, kernel.iteration + max(1, checkpoint_every))
            while kernel.iteration < target:
                kernel.iterate(initial)
                if report is not None:
                    report(kernel.iteration, iterations)
            coverage = publish(kernel.iteration)
            logger.info(
                "rung %d written, %.2f%% of rows have an answer", kernel.iteration, coverage * 100.0
            )

        if coverage == 0.0:
            coverage = publish(kernel.iteration)
        abstract = float(kernel.exploitability(initial))
    except Exception:
        tracker.mark_failed(cleanup_if_empty=True)
        raise

    runtime = time.time() - started
    touched = round(coverage * tree.num_rows)
    tracker.update(
        iterations=kernel.iteration,
        runtime_seconds=runtime,
        num_infosets=touched,
        storage_capacity=tree.num_rows,
    )
    tracker.mark_completed()

    return VectorBlueprintOutput(
        run_id=run_id,
        runs_dir=str(base_dir),
        config_name=config.system.config_name,
        iterations=kernel.iteration,
        num_rows=tree.num_rows,
        touched_rows=touched,
        coverage=coverage,
        runtime_seconds=runtime,
        seconds_per_iteration=runtime / max(1, kernel.iteration),
        abstract_exploitability=abstract,
        universe_boards=universe_boards,
        universe_seed=universe_seed,
        dtype=dtype,
        status="completed",
    )


__all__ = ("VectorBlueprintOutput", "train_vector_blueprint")
