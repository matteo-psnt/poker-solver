"""Training over the statically-enumerated betting tree.

WHY THIS EXISTS. The approach this replaced discovered infosets as it went, so
the space never stopped growing and every worker held dicts proportional to it.
Fitted against a live run (``infosets ~ 1.96 * iters^1.058``):

      iters      infosets   shared GB   per-worker GB   8w node GB
  5,000,000    24,169,390        2.4            2.6         23.3
 10,000,000    50,335,061        5.0            4.1         37.6
 30,000,000   161,007,993       16.1           10.2         98.0

A 30M-iteration run needs ~98 GB on 8 workers. No node we have reaches it, and
no worker count fixes it, because the growth is in the KEYING, not the
parallelism.

The static tree removes the growth rather than budgeting for it. The public
betting tree is small and finite -- 57,604 decision nodes under
``config/training/production.yaml`` -- so an infoset is ``(node_id, bucket)``,
which is an array index. The table is allocated once at full size:

    bounded infoset space  ~16.8M rows, ~1.7 GB, FIXED
    per-worker dicts       none

1M iterations and 300M iterations cost the same memory. That is the difference
between a long run being a budgeting exercise and a long run being possible.

This module is the service seam only: run directory, provenance, metadata. The
solver work lives in :mod:`src.pipeline.training.static_parallel`.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from src.core.actions.action_model import ActionModel
from src.pipeline.training import components
from src.pipeline.training.abstraction_resolver import AbstractionHashMismatchError
from src.pipeline.training.run_tracker import ExperimentTag, RunTracker
from src.pipeline.training.static_parallel import train_static_parallel
from src.shared.config import Config
from src.shared.config_loader import load_training_config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StaticTrainingOutput:
    """Machine-readable summary of a static-storage training run.

    Mirrors :class:`~src.pipeline.services.training.TrainingOutput` where the
    fields mean the same thing, and adds what only the static path can report:
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
) -> StaticTrainingOutput:
    """Train a static-tree solver from a named config and return a portable summary.

    Args:
        config_name: Stem of a config under ``config/training``.
        num_workers: Worker processes. A pure throughput knob: the table is
            shared and there are no per-worker maps, so raising it does not
            raise memory.
        num_iterations: ABSOLUTE iteration target. Continuing past it is a
            no-op, so a retried leg converges rather than repeating.
        checkpoint_every: Checkpoint every N iterations (0 = only at the end).
            The bound on what a killed run loses, traded against disk and write
            time: a full table is written each time. At 250k the writes were
            ~17% of a 30M run's wall clock and left 120 snapshots on the share;
            1M costs ~4% and loses at most ~5 minutes of work.
        run_id: Continue an EXISTING run directory instead of creating one. The
            checkpoint there is loaded first and training continues from it.
        seed: Overrides ``system.seed``.
        config_overrides: Nested config overrides (``__`` separator).
        experiment: Experiment/arm/parent recorded on the run.
        runs_dir: Base runs directory; defaults to the config's.

    Raises:
        FileNotFoundError: The card abstraction is missing (precompute it first).
        AbstractionHashMismatchError: The abstraction on disk is stale (recompute it).
    """
    overrides: dict[str, object] = dict(config_overrides or {})
    if seed is not None:
        overrides["system__seed"] = seed
    config: Config = load_training_config(config_name, **overrides)
    iterations = num_iterations or config.training.num_iterations

    base_dir = Path(runs_dir) if runs_dir is not None else Path(config.training.runs_dir)
    # Random suffix: second resolution collides, and two runs sharing a
    # directory interleave their checkpoints silently.
    if run_id is None:
        run_id = f"run-{datetime.now().strftime('%Y%m%d_%H%M%S')}-{uuid.uuid4().hex[:6]}"
    run_dir = base_dir / run_id
    # A named run that does not exist yet is a fresh start, not an error --
    # that is what makes a scheduler retry continue rather than restart.
    resuming = (run_dir / ".run.json").exists()

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
            abstraction_hash = components.resolve_card_abstraction_hash(config)
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
            resume=resuming,
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
