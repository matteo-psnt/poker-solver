#!/usr/bin/env python3
"""
Test real parallel training merge performance with actual game data.

This uses the real training pipeline to measure merge overhead accurately.
"""

import time

from src.training.trainer import TrainingSession
from src.utils.config import Config
from tests.test_helpers import DummyCardAbstraction


def test_real_merge_performance(num_iterations=500, num_workers=4):
    """
    Test merge performance with real training data.

    Args:
        num_iterations: Total iterations to run
        num_workers: Number of parallel workers
    """
    print(f"{'=' * 80}")
    print("REAL MERGE PERFORMANCE TEST")
    print(f"{'=' * 80}")
    print("Configuration:")
    print(f"  Iterations: {num_iterations}")
    print(f"  Workers: {num_workers}")
    print("  Storage: In-memory")
    print()

    # Create config with in-memory storage
    config = Config.default()
    config.set("storage.backend", "memory")
    config.set("training.verbose", False)

    # Build trainer with dummy card abstraction (fast initialization)
    print("Building trainer (using dummy card abstraction for speed)...")

    # Manually override card abstraction to avoid computation
    from src.training import components

    original_build = components.build_card_abstraction

    def mock_build_card_abstraction(config, prompt_user=False, auto_compute=False):
        return DummyCardAbstraction()

    components.build_card_abstraction = mock_build_card_abstraction

    try:
        trainer = TrainingSession(config)

        print("Trainer initialized")
        print("Starting parallel training...\n")

        # Run with detailed timing
        start_time = time.time()

        results = trainer.train(
            num_iterations=num_iterations,
            use_parallel=True,
            num_workers=num_workers,
        )

        total_time = time.time() - start_time

        # Print results
        print(f"\n{'=' * 80}")
        print("RESULTS:")
        print(f"{'=' * 80}")
        print(f"Total iterations: {results['total_iterations']}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Final infosets: {results['final_infosets']:,}")
        print(f"Average time per iteration: {total_time / num_iterations * 1000:.1f}ms")

        # Estimate merge overhead
        # The merge logging will show actual merge times in the output
        print("\nLook at the '[Master] Merge complete!' messages above to see actual merge times")
        print(f"{'=' * 80}\n")

        return results

    finally:
        # Restore original
        components.build_card_abstraction = original_build


def run_comparison():
    """Run comparison with different batch sizes."""
    print(f"\n{'#' * 80}")
    print("# BATCH SIZE COMPARISON (Real Training)")
    print(f"{'#' * 80}\n")

    configs = [
        {"iterations": 200, "workers": 4, "name": "Small (200 iters, 4 workers)"},
        {"iterations": 500, "workers": 4, "name": "Medium (500 iters, 4 workers)"},
        {"iterations": 1000, "workers": 4, "name": "Large (1000 iters, 4 workers)"},
    ]

    results = []
    for cfg in configs:
        print(f"\n{'=' * 80}")
        print(f"Testing: {cfg['name']}")
        print(f"{'=' * 80}")

        result = test_real_merge_performance(
            num_iterations=cfg["iterations"], num_workers=cfg["workers"]
        )

        result["config"] = cfg["name"]
        results.append(result)

        # Brief pause
        time.sleep(2)

    # Summary
    print(f"\n{'=' * 80}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Configuration':<45} {'Total Time':<15} {'Infosets':<15} {'Time/Iter':<15}")
    print(f"{'-' * 45} {'-' * 15} {'-' * 15} {'-' * 15}")

    for r in results:
        total_time = r.get("elapsed_time", 0)
        iters = r.get("total_iterations", 1)
        time_per_iter = total_time / iters if iters > 0 else 0

        print(
            f"{r['config']:<45} "
            f"{total_time:>13.2f}s "
            f"{r['final_infosets']:>13,} "
            f"{time_per_iter * 1000:>13.1f}ms"
        )

    print(f"{'=' * 80}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Test real merge performance")
    parser.add_argument(
        "--mode",
        choices=["single", "comparison"],
        default="single",
        help="Test mode (default: single)",
    )
    parser.add_argument(
        "--iterations", type=int, default=500, help="Number of iterations (default: 500)"
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of workers (default: 4)")

    args = parser.parse_args()

    if args.mode == "comparison":
        run_comparison()
    else:
        test_real_merge_performance(num_iterations=args.iterations, num_workers=args.workers)


if __name__ == "__main__":
    main()
