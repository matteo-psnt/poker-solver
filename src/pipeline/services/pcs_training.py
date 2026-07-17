"""Train a blueprint by public chance sampling, as an ordinary static run.

The service seam only -- run directory, provenance, the absolute target --
mirroring :mod:`static_training`. The work is
:func:`src.pipeline.training.pcs_parallel.pcs_worker` under the scalar
trainer's own coordinator, and the result is the same checkpoint ladder
``evaluate`` and ``score`` already read.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.vector import compile_tree
from src.pipeline import blueprint
from src.pipeline.training import pcs_parallel
from src.pipeline.training.run_tracker import ExperimentTag, RunTracker
from src.pipeline.training.static_parallel import train_static_parallel
from src.shared import records, run_events
from src.shared.config.loader import load_training_config
from src.shared.log import configure_logging

if TYPE_CHECKING:
    from src.shared.config import Config

PROGRESS_ARTIFACT = "train-progress.json"
KERNEL = "pcs"

logger = logging.getLogger(__name__)


class PcsTrainingOutput(BaseModel):
    """Machine-readable summary of a public-chance-sampling run.

    ``iterations`` counts sampled boards (times ``pcs.runouts_per_flop`` for
    board passes), not the scalar trainer's deals: one iteration here updates
    every live hand at every node the board reaches.
    """

    run_id: str
    runs_dir: str
    config_name: str
    iterations: int
    board_passes: int
    workers: int
    num_rows: int
    touched_rows: int
    coverage: float
    runtime_seconds: float
    iterations_per_second: float
    status: str


# The config sections that decide what the PCS trainer IS. `--set` overrides do
# not carry into a continuation, so the run's own record is the only thing that
# knows; RunTracker.verify_trainer_knobs is where that is enforced.
TRAINER_BLOCKS = ("solver", "pcs")


def train_pcs(
    config_name: str,
    *,
    iterations: int,
    num_workers: int | None = None,
    seed: int | None = None,
    config_overrides: dict[str, object] | None = None,
    experiment: ExperimentTag | None = None,
    runs_dir: Path | None = None,
    checkpoint_every: int = 200,
    retain_every: int = 0,
    run_id: str | None = None,
    progress_file: Path | None = None,
) -> PcsTrainingOutput:
    """Train to an ABSOLUTE iteration target; continuing past it is a no-op.

    ``num_workers`` is a ceiling: the effective count is clamped to what the
    node's RAM holds (``pcs_parallel.ram_safe_workers``), since each worker's
    hand-space scratch is private. ``retain_every`` of 0 keeps EVERY rung --
    a ladder is the only way to find a sampling trainer's best point.
    """
    overrides: dict[str, object] = dict(config_overrides or {})
    if seed is not None:
        overrides["system__seed"] = seed
    config: Config = load_training_config(config_name, **overrides)
    configure_logging(config.system.log_level)

    base_dir = Path(runs_dir) if runs_dir is not None else Path(config.training.runs_dir)
    if run_id is None:
        run_id = f"pcs-{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = base_dir / run_id
    resuming = run_events.log_path(run_dir).exists() or (run_dir / ".run.json").exists()

    action_model = ActionModel(config)
    abstraction = blueprint.build_card_abstraction(config)
    abstraction_hash = blueprint.resolve_card_abstraction_hash(config)
    if resuming:
        tracker = RunTracker.load(run_dir)
        tracker.verify_action_config_hash(action_model.get_config_hash())
        if tracker.metadata.kernel != KERNEL:
            raise ValueError(
                f"Run '{run_id}' was trained by the {tracker.metadata.kernel!r} kernel; "
                "continuing it by public chance sampling would mix two lineages in one ladder."
            )
        tracker.verify_trainer_knobs(config, TRAINER_BLOCKS)
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
            kernel=KERNEL,
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    tracker.initialize()

    rules = GameRules(config.game.small_blind, config.game.big_blind)
    tree = build_betting_tree(
        rules, action_model, abstraction, starting_stack=config.game.starting_stack
    )
    compiled = compile_tree(tree, rules)
    if config.pcs.cfr_br != "off" and config.pcs.alternating:
        raise ValueError(
            "pcs.alternating and pcs.cfr_br cannot both be on: CFR-BR already updates one "
            "seat at a time against that seat's own best-responding opponent, and alternating "
            "on top of it would halve each seat's updates for nothing."
        )
    extra = pcs_parallel.trunk_arrays(config, tree)
    shared = 2 * tree.num_slots * 4 + tree.num_rows * (8 + 8 + 1) + 4 * sum(extra.values())
    safe = pcs_parallel.ram_safe_workers(
        tree, compiled.num_terminals, shared_bytes=shared, br_streets=config.pcs.cfr_br
    )
    requested = num_workers or (os.cpu_count() or 1)
    workers = min(requested, safe)
    logger.info(
        "[pcs] %d workers (%d requested, %d RAM-safe at %.2f GB each), %d runouts per flop, "
        "cfr_br=%s",
        workers,
        requested,
        safe,
        pcs_parallel.worker_bytes(
            tree,
            compiled.num_terminals,
            br_streets=config.pcs.cfr_br,
            runouts=config.pcs.runouts_per_flop,
        )
        / 1e9,
        config.pcs.runouts_per_flop,
        config.pcs.cfr_br,
    )
    # The RESOLVED knobs, not the flags someone meant to pass. A run's identity
    # is what `--config` plus its `--set` list actually produced, and reading it
    # back off a finished run used to mean unpacking the metadata by hand.
    logger.info(
        "[pcs] resolved trainer knobs: solver=%r pcs=%r",
        config.solver.model_dump(),
        config.pcs.model_dump(),
    )
    # The kernel is numpy and numba on one core per worker; BLAS threads on
    # top of that only oversubscribe the node.
    for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[variable] = "1"

    started = time.time()
    try:
        result = train_static_parallel(
            config,
            num_iterations=iterations,
            num_workers=workers,
            session_id=run_id,
            checkpoint_dir=run_dir,
            base_seed=config.system.seed if config.system.seed is not None else 42,
            checkpoint_retain_every=retain_every or checkpoint_every,
            abstraction=abstraction,
            checkpoint_every=checkpoint_every,
            resume=resuming,
            on_progress=records.progress_writer(progress_file, records.REGISTRY[PROGRESS_ARTIFACT]),
            worker=pcs_parallel.pcs_worker,
            before_checkpoint=pcs_parallel.mark_visited_from_strategy,
            extra_arrays=extra,
        )
    except Exception:
        tracker.mark_failed(cleanup_if_empty=True)
        raise

    tracker.update(
        iterations=result.iterations,
        runtime_seconds=time.time() - started,
        num_infosets=result.touched_rows,
        storage_capacity=result.num_rows,
    )
    tracker.mark_completed()

    return PcsTrainingOutput(
        run_id=run_id,
        runs_dir=str(base_dir),
        config_name=config.system.config_name,
        iterations=result.iterations,
        board_passes=result.iterations * config.pcs.runouts_per_flop,
        workers=workers,
        num_rows=result.num_rows,
        touched_rows=result.touched_rows,
        coverage=result.coverage,
        runtime_seconds=result.elapsed_s,
        iterations_per_second=result.iterations_per_second,
        status="completed",
    )


__all__ = ("KERNEL", "TRAINER_BLOCKS", "PcsTrainingOutput", "train_pcs")
