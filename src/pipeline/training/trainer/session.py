"""Training session class for MCCFR solver orchestration."""

from __future__ import annotations

import gc
import multiprocessing as mp
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.core.actions.action_model import ActionModel
from src.engine.solver.storage.array_specs import ARRAY_SPECS
from src.engine.solver.storage.helpers import (
    get_missing_checkpoint_files,
    resolve_resume_iteration,
)
from src.pipeline.training import components
from src.pipeline.training.metrics import MetricsTracker
from src.pipeline.training.run_tracker import RunTracker
from src.shared.config import Config

from . import partitioned
from .checkpointing import CheckpointManager


class TrainingSession:
    """Orchestrates MCCFR training end to end."""

    def __init__(
        self,
        config: Config,
        run_id: str | None = None,
        run_tracker: RunTracker | None = None,
    ):
        self.config = config
        # Set by resume(): pre-allocate shared storage above the checkpoint's
        # capacity so the run never has to resize mid-flight.
        self.capacity_override: int | None = None

        self.run_tracker = run_tracker
        if self.run_tracker:
            self.run_dir = self.run_tracker.run_dir
        else:
            runs_base_dir = Path(self.config.training.runs_dir)
            if run_id is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Random suffix: second-resolution timestamps collide when runs
                # start simultaneously (e.g. parallel cloud containers writing to
                # a shared volume), silently interleaving their checkpoints.
                run_id = f"run-{timestamp}-{uuid.uuid4().hex[:6]}"
            self.run_dir = runs_base_dir / run_id

        self.checkpoints = CheckpointManager(self)

        try:
            self.action_model = ActionModel(config)
            self.card_abstraction = components.build_card_abstraction(config)
            action_config_hash = self.action_model.get_config_hash()
            card_abstraction_hash = components.resolve_card_abstraction_hash(config)

            if self.run_tracker is None:
                self.run_tracker = RunTracker(
                    run_dir=self.run_dir,
                    config_name=self.config.system.config_name,
                    config=config,
                    action_config_hash=action_config_hash,
                    card_abstraction_hash=card_abstraction_hash,
                )
            else:
                self.run_tracker.verify_action_config_hash(action_config_hash)
                self.run_tracker.verify_card_abstraction_hash(card_abstraction_hash)

            self.run_dir.mkdir(parents=True, exist_ok=True)
            self.storage = components.build_storage(
                config,
                run_dir=self.run_dir,
                run_metadata=self.run_tracker.metadata,
            )
            self.solver = components.build_solver(
                config, self.action_model, self.card_abstraction, self.storage
            )

            self.metrics = MetricsTracker()
        except Exception:
            if self.run_tracker is not None:
                self.run_tracker.mark_failed(cleanup_if_empty=True)
            raise

    @classmethod
    def resume(
        cls,
        run_dir: str | Path,
        capacity_override: int | None = None,
    ) -> TrainingSession:
        """Resume training from a checkpoint in an existing run directory."""
        run_path = Path(run_dir)
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_path}")

        run_tracker = RunTracker.load(run_path)
        metadata = run_tracker.metadata

        checkpoint_iter = resolve_resume_iteration(run_path, metadata.iterations)
        if checkpoint_iter is None:
            raise FileNotFoundError(f"No checkpoint found in {run_path}")

        missing_files = get_missing_checkpoint_files(run_path)
        if missing_files:
            raise ValueError(f"Checkpoint is incomplete. Missing files: {missing_files}")

        if metadata.config is None:
            raise ValueError(f"Missing config in run metadata: {run_tracker.metadata_path}")
        config = metadata.config

        session = cls(config, run_id=run_path.name, run_tracker=run_tracker)
        session.capacity_override = capacity_override

        if session.storage.num_infosets() == 0:
            raise ValueError(
                "Failed to load checkpoint data. Storage has 0 infosets.\n"
                f"Expected to load from: {run_path}"
            )

        session.solver.iteration = checkpoint_iter

        assert session.run_tracker is not None
        session.run_tracker.mark_resumed()

        print(f"✅ Resumed from checkpoint at iteration {checkpoint_iter}")

        return session

    def release_bootstrap_storage(self) -> None:
        """Free the session-level bootstrap storage before workers are forked.

        ``__init__`` builds a single-worker ``SharedArrayStorage`` so ``resume``
        can verify the checkpoint actually loads. Two distinct costs follow, and
        both are released here.

        Shared memory: training builds its own coordinator storage under the
        *same* ``session_id``, whose ``cleanup_stale_shm`` unlinks these segments
        regardless -- so releasing makes the handoff explicit and reclaims the
        allocation instead of orphaning it for the run's lifetime.

        Python heap: with ``num_workers=1`` this storage owns *every* key, so on
        resume ``owned_keys`` holds one entry per infoset -- measured at ~289
        bytes each, i.e. ~3 GB at 10.6M infosets and ~5 GB at 18M. Workers are
        forked, and CPython's refcounting touches inherited pages, so each child
        copies that heap: ~98 GB across 32 workers at 10.6M infosets, ~84 GB
        across 16 at 18M. That is the resume OOM -- SIGKILL right after "All N
        workers started" -- and it is why fresh runs are immune (their heap is
        near-empty at fork). The worker manager's own storage is built with
        ``load_checkpoint_on_init=False`` and its dicts are empty at fork, so
        this bootstrap copy is the whole cost.

        Idempotent: safe if training is entered more than once.
        """
        storage = getattr(self, "storage", None)
        if storage is None:
            return
        # Drop the key dictionaries before the fork, not just the shared memory:
        # these are the pages children would copy.
        state = storage.state
        state.owned_keys = {}
        # Shares InfoSetKey objects with owned_keys, so leaving it would make
        # clearing owned_keys free nothing.
        state.unshipped_keys = []
        state.remote_keys = {}
        state.legal_actions_cache = {}
        state.pending_id_requests = {}
        state.requested_id_keys = set()
        state.unanswered_id_requests = {}
        state.pending_late_responses = {}
        # Return the freed arenas to the allocator now; leaving collection to a
        # later automatic pass would let the fork copy pages we no longer need.
        gc.collect()
        # Drop the numpy views first: they export pointers into the buffers, and
        # SharedMemory.close() raises BufferError while any are still alive.
        for spec in ARRAY_SPECS:
            setattr(storage, spec.attr, np.empty(spec.shape(0, storage.max_actions), spec.dtype))
        storage.cleanup()

    def __del__(self):
        """Cleanup on deletion (guarded: __init__ may have failed before the manager)."""
        checkpoints = getattr(self, "checkpoints", None)
        if checkpoints is not None:
            checkpoints.shutdown()

    @property
    def verbose(self) -> bool:
        """Get verbose setting from config."""
        return self.config.training.verbose

    def train(
        self,
        num_iterations: int | None = None,
        num_workers: int | None = None,
        batch_size: int | None = None,
    ) -> dict[str, Any]:
        """Run parallel training using hash-partitioned infosets."""
        if num_iterations is None:
            num_iterations = self.config.training.num_iterations

        if num_workers is None:
            num_workers = mp.cpu_count()

        return partitioned.train_partitioned(self, num_iterations, num_workers, batch_size)

    def evaluate(
        self,
        num_samples: int = 10000,
        num_rollouts_per_infoset: int = 100,
        use_average_strategy: bool = True,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Evaluate current solver using exploitability estimation."""
        results = components.evaluate_solver_exploitability(
            self.solver,
            num_samples=num_samples,
            use_average_strategy=use_average_strategy,
            num_rollouts_per_infoset=num_rollouts_per_infoset,
            seed=seed,
        )

        return {
            "num_infosets": self.solver.num_infosets(),
            "exploitability_mbb": results["exploitability_mbb"],
            "std_error_mbb": results["std_error_mbb"],
            "confidence_95_mbb": results["confidence_95_mbb"],
        }

    def __str__(self) -> str:
        """String representation."""
        return f"TrainingSession(solver={self.solver}, config={self.config})"
