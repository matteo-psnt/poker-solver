"""Progress and summary rendering helpers for TrainingSession."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from tqdm import tqdm

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from src.pipeline.training.trainer.session import TrainingSession


def print_training_header(
    session: TrainingSession,
    num_workers: int,
    num_iterations: int,
    batch_size: int,
    initial_capacity: int,
    max_actions: int,
) -> None:
    if not session.verbose:
        return
    logger.info("\n🚀 Shared Array Parallel Training")
    logger.info(f"   Workers: {num_workers}")
    logger.info(f"   Iterations: {num_iterations}")
    logger.info(f"   Batch size: {batch_size}")
    logger.info(f"   Initial capacity: {initial_capacity:,}")
    logger.info(f"   Max actions: {max_actions}")
    logger.info("   Mode: Live shared memory arrays")


def update_progress_bar(
    session: TrainingSession,
    progress_bar: tqdm,
    iteration: int,
    total_infosets: int,
    capacity_usage: float,
) -> None:
    if not session.verbose or not isinstance(progress_bar, tqdm):
        return
    compact_summary = session.metrics.get_compact_summary()
    progress_bar.set_postfix_str(
        f"iter={iteration} infosets={total_infosets} cap={capacity_usage:.0%} | {compact_summary}"
    )


def print_final_summary(
    session: TrainingSession,
    total_iterations: int,
    total_infosets: int,
    elapsed_time: float,
    interrupted: bool,
) -> None:
    if not session.verbose:
        return
    if interrupted:
        logger.info("🟡 Training interrupted")
    else:
        logger.info("✅ Shared Array Training complete!")

    logger.info(f"   Iterations: {total_iterations}")
    logger.info(f"   Infosets: {total_infosets:,}")
    logger.info(f"   Time: {elapsed_time:.1f}s")

    if total_iterations > 0:
        logger.info(f"   Speed: {total_iterations / elapsed_time:.2f} iter/s")
