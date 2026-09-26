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
from functools import partial
from typing import TYPE_CHECKING, TypedDict

from pydantic import BaseModel

from src.core.game.rules import GameRules
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.vector import compile_tree
from src.pipeline import blueprint
from src.pipeline.services.run_opening import open_run, train_recorded
from src.pipeline.training import pcs_parallel
from src.pipeline.training.static_parallel import train_static_parallel
from src.shared import records

if TYPE_CHECKING:
    from pathlib import Path

    from src.pipeline.training.run_tracker import (
        ExperimentTag,
    )
    from src.shared.config import Config
    from src.shared.ports.record import RecordSink, RecordSource

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


class WorkerFootprint(TypedDict):
    br_streets: str
    runouts: int
    kernels: int


def worker_footprint(config: Config) -> WorkerFootprint:
    """The kwargs describing one worker's memory, for the clamp AND the log line.

    A turn BEST RESPONSE holds one kernel per runout, because its joint
    maximisation needs their values at the same time. Plain PCS has no argmax
    and rebinds one kernel per runout whatever the sampler shares.
    """
    joint = config.pcs.runout_mode == "turn" and config.pcs.cfr_br != "off"
    return {
        "br_streets": config.pcs.cfr_br,
        "runouts": config.pcs.runouts_per_flop,
        "kernels": config.pcs.runouts_per_flop if joint else 1,
    }


def train_pcs(
    config_name: str | None,
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
    # Dual write. `None` writes files only, which is what every task did
    # before the database existed and what one dispatched without a DSN
    # still does. Constructed by the COMMAND, never here: the composition
    # root is the only layer allowed to know which adapter this is.
    sink: RecordSink | None = None,
    record_source: RecordSource | None = None,
) -> PcsTrainingOutput:
    """Train to an ABSOLUTE iteration target; continuing past it is a no-op.

    ``num_workers`` is a ceiling: the effective count is clamped to what the
    node's RAM holds (``pcs_parallel.ram_safe_workers``), since each worker's
    hand-space scratch is private. ``retain_every`` of 0 keeps EVERY rung --
    a ladder is the only way to find a sampling trainer's best point.

    A continuation trains what it was: the config comes off the run's own
    record and whatever this task carried is logged and ignored -- a Batch
    retry re-runs a FRESH submission's argv against the record its first
    attempt wrote, so refusing here would kill every retried run. The
    operator-facing refusal lives in `submit` and the task kinds.
    """
    run = open_run(
        config_name,
        kernel=KERNEL,
        pin_abstraction=True,
        run_id=run_id,
        runs_dir=runs_dir,
        seed=seed,
        config_overrides=config_overrides,
        experiment=experiment,
        sink=sink,
        record_source=record_source,
    )
    config, tracker, action_model = run.config, run.tracker, run.action_model
    # By the HASH the run recorded, fresh or continued: a preset renamed or
    # rebuilt since neither stops its ladder nor mixes a second abstraction in;
    # the resolver skips the YAML when pinned, and fails if the hash is gone.
    abstraction = blueprint.build_card_abstraction(
        config, abstraction_hash=tracker.metadata.card_abstraction_hash
    )

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
    if config.pcs.runout_mode == "turn" and config.pcs.runouts_per_flop < 2:
        raise ValueError(
            "pcs.runout_mode='turn' shares one turn across the iteration's runouts so a turn "
            "best response can maximise over them; with runouts_per_flop=1 there is nothing to "
            "maximise over and the sampler is just a slower way to draw one board."
        )
    if config.pcs.cfr_br in ("turn_river", "postflop") and config.pcs.runout_mode != "turn":
        raise ValueError(
            f"pcs.cfr_br={config.pcs.cfr_br!r} best-responds on the turn, but "
            "pcs.runout_mode='flop' gives every runout its own turn -- the argmax would read a "
            "river that has not been dealt. Set pcs__runout_mode=turn, or use cfr_br=river."
        )
    extra = pcs_parallel.trunk_arrays(config, tree)
    shared = 2 * tree.num_slots * 4 + tree.num_rows * (8 + 8 + 1) + 4 * sum(extra.values())
    footprint = worker_footprint(config)
    safe = pcs_parallel.ram_safe_workers(compiled, shared_bytes=shared, **footprint)
    requested = num_workers or (os.cpu_count() or 1)
    workers = min(requested, safe)
    logger.info("[pcs] transparent huge pages: %s", pcs_parallel.huge_page_modes())
    logger.info(
        "[pcs] %d workers (%d requested, %d RAM-safe at %.2f GB each), %d runouts per flop, "
        "cfr_br=%s",
        workers,
        requested,
        safe,
        pcs_parallel.worker_bytes(compiled, **footprint) / 1e9,
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

    result = train_recorded(
        tracker,
        lambda: train_static_parallel(
            config,
            num_iterations=iterations,
            num_workers=workers,
            session_id=run.run_id,
            checkpoint_dir=run.run_dir,
            abstraction_id=tracker.metadata.card_abstraction_hash,
            base_seed=config.system.seed if config.system.seed is not None else 42,
            checkpoint_retain_every=retain_every or checkpoint_every,
            abstraction=abstraction,
            checkpoint_every=checkpoint_every,
            resume=run.resuming,
            on_progress=records.progress_writer(progress_file, records.REGISTRY[PROGRESS_ARTIFACT]),
            on_checkpoint=tracker.record_checkpoint,
            worker=pcs_parallel.pcs_worker,
            before_checkpoint=partial(pcs_parallel.prepare_checkpoint, config=config),
            extra_arrays=extra,
            dynamic=True,
        ),
    )

    return PcsTrainingOutput(
        run_id=run.run_id,
        runs_dir=str(run.base_dir),
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


__all__ = ("KERNEL", "PcsTrainingOutput", "train_pcs")
