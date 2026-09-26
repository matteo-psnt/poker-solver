"""Training over the statically-enumerated betting tree.

An infoset is ``(node_id, bucket)`` -- an array index into a table allocated
once at full size, ~16.8M rows and ~1.7 GB, with no per-worker dicts. So 1M
iterations and 300M iterations cost the same memory, which is the difference
between a long run being a budgeting exercise and a long run being possible.
Discovering infosets at runtime instead grows with the KEYING rather than the
parallelism, and no worker count fixes that.

This module is the service seam only: run directory, provenance, metadata. The
solver work lives in :mod:`src.pipeline.training.static_parallel`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.pipeline.services.run_opening import open_run, train_recorded
from src.pipeline.training.static_parallel import train_static_parallel
from src.shared import records

if TYPE_CHECKING:
    from pathlib import Path

    from src.pipeline.training.run_tracker import (
        ExperimentTag,
    )
    from src.shared.ports.record import RecordSink, RecordSource

PROGRESS_ARTIFACT = "train-progress.json"

logger = logging.getLogger(__name__)


class StaticTrainingOutput(BaseModel):
    """Machine-readable summary of a static-storage training run.

    ``coverage`` -- the fraction of the infoset space training reached -- is
    meaningful only because the table's size is known up front.
    """

    run_id: str
    runs_dir: str
    config_name: str
    iterations: int
    num_rows: int
    touched_rows: int
    coverage: float
    mean_visits_per_touched: float
    runtime_seconds: float
    iterations_per_second: float
    dropped_updates: int
    status: str


def train_static(
    config_name: str | None,
    *,
    num_workers: int = 1,
    num_iterations: int | None = None,
    seed: int | None = None,
    config_overrides: dict[str, object] | None = None,
    experiment: ExperimentTag | None = None,
    runs_dir: Path | None = None,
    checkpoint_every: int = 5_000_000,
    run_id: str | None = None,
    progress_file: Path | None = None,
    # Dual write. `None` writes files only, which is what every task did
    # before the database existed and what one dispatched without a DSN
    # still does. Constructed by the COMMAND, never here: the composition
    # root is the only layer allowed to know which adapter this is.
    sink: RecordSink | None = None,
    record_source: RecordSource | None = None,
) -> StaticTrainingOutput:
    """Train a static-tree solver from a named config and return a portable summary.

    Args:
        config_name: Stem of a config under ``config/training``. Ignored on a
            continuation, which trains the config on the run's own record.
        num_workers: Worker processes. A pure throughput knob: the table is
            shared and there are no per-worker maps, so raising it does not
            raise memory.
        num_iterations: ABSOLUTE iteration target. Continuing past it is a
            no-op, so a retried task converges rather than repeating.
        checkpoint_every: Checkpoint every N iterations (0 = only at the end).
            The bound on what a killed run loses, traded against a full-table
            write and a worker respawn per chunk. Measured 08-22 at 16 workers:
            a 1M chunk is ~18 s and its checkpoint ~2 s, so 1M paid ~20% and
            5M pays under 5% while losing at most ~90 s of work.
        run_id: Continue an EXISTING run directory instead of creating one. The
            checkpoint there is loaded first and training continues from it.
        seed: Overrides ``system.seed``.
        config_overrides: Nested config overrides (``__`` separator).
        experiment: Experiment/arm/parent recorded on the run.
        runs_dir: Base runs directory (default `data/runs`, node-relative).
        progress_file: Where to publish iterations done while they are being
            done. The checkpoint is the durable answer, but one lands every
            million iterations -- minutes to half an hour apart -- and that is
            not a cadence anything can watch.

    Raises:
        FileNotFoundError: The card abstraction is missing (precompute it first).
        AbstractionHashMismatchError: The abstraction on disk is stale (recompute it).
    """
    run = open_run(
        config_name,
        kernel="scalar",
        pin_abstraction=False,
        run_id=run_id,
        runs_dir=runs_dir,
        seed=seed,
        config_overrides=config_overrides,
        experiment=experiment,
        sink=sink,
        record_source=record_source,
    )
    config, tracker = run.config, run.tracker
    iterations = num_iterations or config.training.num_iterations
    result = train_recorded(
        tracker,
        lambda: train_static_parallel(
            config,
            num_iterations=iterations,
            num_workers=num_workers,
            session_id=run.run_id,
            checkpoint_dir=run.run_dir,
            # seed is optional in config; the static path needs a concrete
            # value because worker seeds are derived from it deterministically.
            base_seed=config.system.seed if config.system.seed is not None else 42,
            checkpoint_retain_every=config.storage.checkpoint_retain_every,
            # The bucket ASSIGNMENT, which the tree fingerprint deliberately does
            # not cover. Without it `load_checkpoint`'s AbstractionMismatchError
            # cannot fire at all -- it needs the id on BOTH sides.
            abstraction_id=tracker.metadata.card_abstraction_hash,
            checkpoint_every=checkpoint_every,
            resume=run.resuming,
            on_progress=records.progress_writer(progress_file, records.REGISTRY[PROGRESS_ARTIFACT]),
            on_checkpoint=tracker.record_checkpoint,
        ),
    )

    return StaticTrainingOutput(
        run_id=run.run_id,
        runs_dir=str(run.base_dir),
        config_name=config.system.config_name,
        iterations=result.iterations,
        num_rows=result.num_rows,
        touched_rows=result.touched_rows,
        coverage=result.coverage,
        mean_visits_per_touched=result.mean_visits_per_touched,
        runtime_seconds=result.elapsed_s,
        iterations_per_second=result.iterations_per_second,
        dropped_updates=result.dropped_updates,
        status="completed",
    )
