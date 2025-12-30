"""Progress and summary rendering helpers for TrainingSession."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tqdm import tqdm

if TYPE_CHECKING:
    from src.training.trainer.session import TrainingSession


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
    print("\n🚀 Shared Array Parallel Training")
    print(f"   Workers: {num_workers}")
    print(f"   Iterations: {num_iterations}")
    print(f"   Batch size: {batch_size}")
    print(f"   Initial capacity: {initial_capacity:,}")
    print(f"   Max actions: {max_actions}")
    print("   Mode: Live shared memory arrays")


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
    fallback_stats: dict[str, float] | None = None,
) -> None:
    if not session.verbose:
        return
    if interrupted:
        print("🟡 Training interrupted")
    else:
        print("✅ Shared Array Training complete!")

    print(f"   Iterations: {total_iterations}")
    print(f"   Infosets: {total_infosets:,}")
    print(f"   Time: {elapsed_time:.1f}s")

    if total_iterations > 0:
        print(f"   Speed: {total_iterations / elapsed_time:.2f} iter/s")

    if fallback_stats:
        total_lookups = int(fallback_stats.get("total_lookups", 0))
        fallback_count = int(fallback_stats.get("fallback_count", 0))
        if total_lookups > 0:
            fallback_rate = fallback_stats.get("fallback_rate", 0.0) * 100
            print(
                f"   Abstraction fallbacks: {fallback_count:,}/{total_lookups:,} "
                f"({fallback_rate:.2f}%)"
            )
