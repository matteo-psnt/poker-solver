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
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import BaseModel

from src.core.actions.action_model import ActionModel
from src.pipeline import blueprint
from src.pipeline.abstraction.resolver import AbstractionHashMismatchError
from src.pipeline.training.run_tracker import (
    ExperimentTag,
    RunTracker,
    has_run_record,
    refuse_config_on_continue,
)
from src.pipeline.training.static_parallel import train_static_parallel
from src.shared import records
from src.shared.config import DEFAULT_RUNS_DIR
from src.shared.config.loader import load_training_config
from src.shared.log import configure_logging

if TYPE_CHECKING:
    from src.shared.config import Config
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
        config_name: Stem of a config under ``config/training``. Refused on a
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
    base_dir = Path(runs_dir) if runs_dir is not None else Path(DEFAULT_RUNS_DIR)
    # Random suffix: second resolution collides, and two runs sharing a
    # directory interleave their checkpoints silently.
    if run_id is None:
        run_id = f"run-{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = base_dir / run_id
    # A named run that does not exist yet is a fresh start, not an error --
    # that is what makes a scheduler retry continue rather than restart.
    # BOTH layouts. A run written before the event log has a .run.json and no
    # log, and missing it here is not a cosmetic bug: `resuming` False mints
    # fresh metadata over a live run, skips verify_action_config_hash, and
    # restarts training from zero into a directory holding a real ladder --
    # which save_checkpoint then extends with mixed-lineage rungs and prunes.
    # THROUGH THE RECORD, not the filesystem. `run.jsonl` is no longer
    # written, so a directory check answers `False` for every run created
    # after the flip -- which mints fresh metadata over a live ladder and
    # restarts training from zero.
    resuming = has_run_record(run_dir, record_source)
    if resuming:
        refuse_config_on_continue(run_id, config_name, config_overrides, seed)
        tracker = RunTracker.load(run_dir, record_source, sink)
        config: Config = tracker.metadata.config
    else:
        if not config_name:
            raise ValueError("a fresh run needs a config name; only a continuation goes without")
        overrides: dict[str, object] = dict(config_overrides or {})
        if seed is not None:
            overrides["system__seed"] = seed
        config = load_training_config(config_name, **overrides)
    # The run's own verbosity. Workers repeat this from the same field, so all
    # processes agree; --log-level still outranks it via the environment.
    configure_logging(config.system.log_level)
    iterations = num_iterations or config.training.num_iterations

    action_model = ActionModel(config)
    if resuming:
        tracker.verify_action_config_hash(action_model.get_config_hash())
        tracker.mark_resumed()
    else:
        tag = experiment or ExperimentTag()
        # Resolve the abstraction BEFORE anything is written, and translate its
        # two failure modes into messages that name the fix. Bare "no such file"
        # from deep inside the resolver does not tell a caller that the answer is
        # `precompute`, and this is the first thing a fresh checkout hits.
        try:
            abstraction_hash = blueprint.resolve_card_abstraction_hash(config)
        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Card abstraction '{config.card_abstraction.config}' for training config "
                f"'{config_name}' is missing. Precompute it before training. ({e})"
            ) from e
        except AbstractionHashMismatchError as e:
            raise AbstractionHashMismatchError(
                f"Card abstraction '{config.card_abstraction.config}' for training config "
                f"'{config_name}' is stale (config hash mismatch). Recompute it. ({e})"
            ) from e
        tracker = RunTracker(
            sink=sink,
            source=record_source,
            run_dir=run_dir,
            config_name=config.system.config_name,
            config=config,
            action_config_hash=action_model.get_config_hash(),
            card_abstraction_hash=abstraction_hash,
            experiment_id=tag.experiment_id,
            arm=tag.arm,
            parent_run_id=tag.parent_run_id,
            kernel="scalar",
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    # Before training, so `created` is genuinely the log's first event -- a run
    # listing reads identity from that one line rather than folding.
    tracker.initialize()

    started = time.time()
    try:
        result = train_static_parallel(
            config,
            num_iterations=iterations,
            num_workers=num_workers,
            session_id=run_id,
            checkpoint_dir=run_dir,
            # seed is optional in config; the static path needs a concrete
            # value because worker seeds are derived from it deterministically.
            base_seed=config.system.seed if config.system.seed is not None else 42,
            checkpoint_retain_every=config.storage.checkpoint_retain_every,
            # The bucket ASSIGNMENT, which the tree fingerprint deliberately does
            # not cover. Without it `load_checkpoint`'s AbstractionMismatchError
            # cannot fire at all -- it needs the id on BOTH sides -- so the guard
            # its own docstring calls "the only way that failure is ever visible"
            # was inert for every run this trainer ever wrote.
            abstraction_id=tracker.metadata.card_abstraction_hash,
            checkpoint_every=checkpoint_every,
            resume=resuming,
            on_progress=records.progress_writer(progress_file, records.REGISTRY[PROGRESS_ARTIFACT]),
            on_checkpoint=tracker.record_checkpoint,
        )
    except Exception:
        # cleanup_if_empty so a run that died before writing anything does not
        # leave an empty directory for `describe_runs` to report as a real run.
        tracker.mark_failed(cleanup_if_empty=True)
        raise

    tracker.update(
        iterations=result.iterations,
        runtime_seconds=time.time() - started,
        num_infosets=result.touched_rows,
        storage_capacity=result.num_rows,
    )
    tracker.mark_completed()

    return StaticTrainingOutput(
        run_id=run_id,
        runs_dir=str(base_dir),
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
