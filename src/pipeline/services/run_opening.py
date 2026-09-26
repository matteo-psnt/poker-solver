"""Opening a training run, fresh or continued, with its identity checked first.

Both trainers did this in near-copies that had drifted: one refused a run the
other kernel trained and one did not; one named the fix for a missing
abstraction and one did not. Nothing is written before every check passes.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from src.core.actions.action_model import ActionModel
from src.pipeline import blueprint
from src.pipeline.abstraction.resolver import AbstractionHashMismatchError
from src.pipeline.training.run_tracker import (
    ExperimentTag,
    RunTracker,
    continued_config,
    has_run_record,
)
from src.shared.config import DEFAULT_RUNS_DIR
from src.shared.config.loader import load_training_config
from src.shared.log import configure_logging

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.pipeline.training.static_parallel import StaticTrainingResult
    from src.shared.config import Config
    from src.shared.ports.record import RecordSink, RecordSource

# What a run recorded before the kernel was a field. PCS arrived in the same
# commit as the field, so every run without one was trained by the scalar kernel.
LEGACY_KERNEL = "scalar"


@dataclass(frozen=True)
class OpenedRun:
    run_id: str
    base_dir: Path
    run_dir: Path
    config: Config
    tracker: RunTracker
    resuming: bool
    action_model: ActionModel


def open_run(
    config_name: str | None,
    *,
    kernel: str,
    pin_abstraction: bool,
    run_id: str | None,
    runs_dir: Path | None,
    seed: int | None,
    config_overrides: dict[str, object] | None,
    experiment: ExperimentTag | None,
    sink: RecordSink | None,
    record_source: RecordSource | None,
) -> OpenedRun:
    """The run to train, its tracker initialised and its identity verified.

    ``pin_abstraction`` decides what a continuation trusts: the hash the run
    recorded (PCS loads exactly that, so a moved preset cannot enter its ladder)
    or the preset's CURRENT resolution, which must still match (scalar).
    """
    base_dir = Path(runs_dir) if runs_dir is not None else Path(DEFAULT_RUNS_DIR)
    # Random suffix: second resolution collides, and two runs sharing a
    # directory interleave their checkpoints silently.
    if run_id is None:
        stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        run_id = f"{'pcs' if kernel == 'pcs' else 'run'}-{stamp}-{uuid.uuid4().hex[:6]}"
    run_dir = base_dir / run_id
    # THROUGH THE RECORD, not the filesystem: `run.jsonl` is no longer written,
    # so a directory check answers False for a live run, which would mint fresh
    # metadata over its ladder and restart training from zero.
    resuming = has_run_record(run_dir, record_source)
    if resuming:
        tracker = RunTracker.load(run_dir, record_source, sink)
        config: Config = continued_config(tracker, config_name, config_overrides, seed)
    else:
        if not config_name:
            raise ValueError("a fresh run needs a config name; only a continuation goes without")
        overrides: dict[str, object] = dict(config_overrides or {})
        if seed is not None:
            overrides["system__seed"] = seed
        config = load_training_config(config_name, **overrides)
    # The run's own verbosity; workers repeat it from the same field, and
    # --log-level still outranks it via the environment.
    configure_logging(config.system.log_level)

    action_model = ActionModel(config)
    if resuming:
        tracker.verify_action_config_hash(action_model.get_config_hash())
        recorded = tracker.metadata.kernel or LEGACY_KERNEL
        if recorded != kernel:
            raise ValueError(
                f"Run '{run_id}' was trained by the {recorded!r} kernel; continuing it "
                f"with {kernel!r} would mix two lineages in one ladder."
            )
        if not pin_abstraction:
            tracker.verify_card_abstraction_hash(blueprint.resolve_card_abstraction_hash(config))
        tracker.mark_resumed()
    else:
        # Resolved BEFORE anything else is read or written: its failures name the fix.
        abstraction_hash = _resolved_abstraction(config, config_name)
        tag = experiment or ExperimentTag()
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
            kernel=kernel,
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    # Before training, so `created` is genuinely the log's first event.
    tracker.initialize()
    return OpenedRun(run_id, base_dir, run_dir, config, tracker, resuming, action_model)


def _resolved_abstraction(config: Config, config_name: str | None) -> str | None:
    """The abstraction hash a fresh run records, its two failures naming the fix.

    A bare "no such file" from deep inside the resolver does not tell a caller
    that the answer is `precompute`, and this is the first thing a fresh
    checkout hits.
    """
    try:
        return blueprint.resolve_card_abstraction_hash(config)
    except FileNotFoundError as e:
        name = config.card_abstraction.config
        raise FileNotFoundError(
            f"Card abstraction '{name}' for training config '{config_name}' is missing. "
            f"Precompute it before training. ({e})"
        ) from e
    except AbstractionHashMismatchError as e:
        name = config.card_abstraction.config
        raise AbstractionHashMismatchError(
            f"Card abstraction '{name}' for training config '{config_name}' is stale "
            f"(config hash mismatch). Recompute it. ({e})"
        ) from e


def train_recorded(
    tracker: RunTracker, train: Callable[[], StaticTrainingResult]
) -> StaticTrainingResult:
    """Run ``train`` and record how it ended on the run.

    ``cleanup_if_empty`` so a run that died before writing anything does not
    leave an empty directory for `describe_runs` to report as a real run.
    """
    started = time.time()
    try:
        result = train()
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
    return result


__all__ = ("LEGACY_KERNEL", "OpenedRun", "open_run", "train_recorded")
