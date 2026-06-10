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
from src.pipeline.services import warm_start
from src.pipeline.training.run_tracker import ExperimentTag, RunTracker
from src.pipeline.training.static_parallel import train_static_parallel
from src.shared import records, run_events
from src.shared.config.loader import load_training_config
from src.shared.log import configure_logging

if TYPE_CHECKING:
    from src.shared.config import Config

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
    config_name: str,
    *,
    num_workers: int = 1,
    num_iterations: int | None = None,
    seed: int | None = None,
    config_overrides: dict[str, object] | None = None,
    experiment: ExperimentTag | None = None,
    runs_dir: Path | None = None,
    checkpoint_every: int = 1_000_000,
    run_id: str | None = None,
    warm_start_from: Path | None = None,
    warm_start_weight: int = warm_start.DEFAULT_EFFECTIVE_ITERATIONS,
    warm_start_at: int | None = None,
    progress_file: Path | None = None,
) -> StaticTrainingOutput:
    """Train a static-tree solver from a named config and return a portable summary.

    Args:
        config_name: Stem of a config under ``config/training``.
        num_workers: Worker processes. A pure throughput knob: the table is
            shared and there are no per-worker maps, so raising it does not
            raise memory.
        num_iterations: ABSOLUTE iteration target. Continuing past it is a
            no-op, so a retried task converges rather than repeating.
        checkpoint_every: Checkpoint every N iterations (0 = only at the end).
            The bound on what a killed run loses, traded against disk and write
            time: a full table is written each time. At 250k the writes were
            ~17% of a 30M run's wall clock and left 120 snapshots on the share;
            1M costs ~4% and loses at most ~5 minutes of work.
        run_id: Continue an EXISTING run directory instead of creating one. The
            checkpoint there is loaded first and training continues from it.
        warm_start_from: Seed a FRESH run from this run's average strategy
            before training. Ignored when continuing, so a retried leg does
            not re-seed over its own progress.
        warm_start_weight: How much accumulated regret the prior claims.
        warm_start_at: Which rung of the prior to seed from. Board-free
            quality is not monotone in iterations, so the LAST rung is not
            generally the best one; omitting this seeds from whatever the
            manifest calls current, which is rarely what was measured.
        seed: Overrides ``system.seed``.
        config_overrides: Nested config overrides (``__`` separator).
        experiment: Experiment/arm/parent recorded on the run.
        runs_dir: Base runs directory; defaults to the config's.
        progress_file: Where to publish iterations done while they are being
            done. The checkpoint is the durable answer, but one lands every
            million iterations -- minutes to half an hour apart -- and that is
            not a cadence anything can watch.

    Raises:
        FileNotFoundError: The card abstraction is missing (precompute it first).
        AbstractionHashMismatchError: The abstraction on disk is stale (recompute it).
    """
    overrides: dict[str, object] = dict(config_overrides or {})
    if seed is not None:
        overrides["system__seed"] = seed
    config: Config = load_training_config(config_name, **overrides)
    # The run's own verbosity. Workers repeat this from the same field, so all
    # processes agree; --log-level still outranks it via the environment.
    configure_logging(config.system.log_level)
    iterations = num_iterations or config.training.num_iterations

    base_dir = Path(runs_dir) if runs_dir is not None else Path(config.training.runs_dir)
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
    resuming = run_events.log_path(run_dir).exists() or (run_dir / ".run.json").exists()

    action_model = ActionModel(config)
    if resuming:
        tracker = RunTracker.load(run_dir)
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
            run_dir=run_dir,
            config_name=config.system.config_name,
            config=config,
            action_config_hash=action_model.get_config_hash(),
            card_abstraction_hash=abstraction_hash,
            experiment_id=tag.experiment_id,
            arm=tag.arm,
            parent_run_id=tag.parent_run_id,
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    # Before training, so `created` is genuinely the log's first event -- a run
    # listing reads identity from that one line rather than folding.
    tracker.initialize()

    # Seeding is a FRESH-run act: a retry finds the run dir populated, resumes,
    # and must not lay the prior back over the progress it already made.
    #
    # But "resuming" and "was seeded" are different questions, and conflating
    # them cost two 30M sweeps. The first attempt died before seeding (the rung
    # it wanted had not been fetched); the retry saw a populated directory,
    # skipped the prior, and trained a perfectly good CONTROL under the arm's
    # name. Every arm agreed to a tenth of a percent and nothing failed.
    #
    # So a resume records what it inherited. A run that asked for a prior and is
    # resuming one that never got seeded is not a smaller version of the
    # experiment -- it is a different arm wearing its label.
    seeded = False
    if (
        warm_start_from is not None
        and resuming
        and not (run_dir / warm_start.SEEDED_MARKER).exists()
    ):
        raise ValueError(
            f"Run '{run_id}' asked to seed from '{warm_start_from}' but is resuming a "
            "directory that was never seeded. Continuing would train an unseeded arm "
            "under a warm-start label. Delete the run directory to start it cleanly."
        )
    if warm_start_from is not None and not resuming:
        # A bare run id resolves under runs_dir, exactly as --run does; an
        # explicit path is taken as given. A node passes the id, because the
        # directory it lands in is the node's business, not the submitter's.
        source = Path(warm_start_from)
        if not source.exists():
            source = base_dir / str(warm_start_from)
        warm_start.seed_checkpoint(
            config,
            source_run=source,
            run_dir=run_dir,
            effective_iterations=warm_start_weight,
            abstraction_hash=tracker.metadata.card_abstraction_hash,
            at_iteration=warm_start_at,
        )
        (run_dir / warm_start.SEEDED_MARKER).write_text(
            f"{warm_start_from}@{warm_start_at or 'current'} weight={warm_start_weight}\n"
        )
        seeded = True

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
            checkpoint_every=checkpoint_every,
            resume=resuming or seeded,
            on_progress=records.progress_writer(progress_file, records.REGISTRY[PROGRESS_ARTIFACT]),
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
