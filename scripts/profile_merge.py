#!/usr/bin/env python3
"""
Profiling script for parallel training merge performance.

Tests merge speed with different optimization strategies and provides detailed timing breakdowns.
"""

import cProfile
import io
import pstats
import time
from typing import Any, Dict, List, TypedDict, cast

import numpy as np

from src.bucketing.utils.infoset import InfoSetKey
from src.game.actions import Action, ActionType
from src.game.state import Street
from src.solver.storage import InMemoryStorage
from src.training.parallel import merge_worker_results


class WorkerResult(TypedDict):
    worker_id: int
    batch_id: int
    utilities: List[float]
    infoset_data: Dict[Any, Dict[str, Any]]
    num_infosets: int


def generate_mock_worker_results(
    num_workers: int, infosets_per_worker: int, overlap: float = 0.7
) -> List[WorkerResult]:
    """
    Generate realistic mock worker results for benchmarking.

    Args:
        num_workers: Number of workers to simulate
        infosets_per_worker: Number of infosets each worker discovers
        overlap: Fraction of infosets that overlap between workers (0-1)

    Returns:
        List of worker result dictionaries
    """
    print(f"Generating {num_workers} worker results with ~{infosets_per_worker} infosets each...")

    # Create a pool of shared infoset keys (for overlap)
    num_shared = int(infosets_per_worker * overlap)
    num_unique_per_worker = infosets_per_worker - num_shared

    shared_keys = []
    for i in range(num_shared):
        key = InfoSetKey(
            player_position=i % 2,
            street=Street.FLOP if i % 3 == 0 else Street.TURN,
            betting_sequence=f"shared-{i}",
            preflop_hand=None,
            postflop_bucket=i % 50,
            spr_bucket=1,
        )
        shared_keys.append(key)

    worker_results: List[WorkerResult] = []
    for worker_id in range(num_workers):
        infoset_data = {}

        # Add shared infosets (all workers see these)
        for key in shared_keys:
            actions = [
                Action(ActionType.FOLD),
                Action(ActionType.CALL),
                Action(ActionType.RAISE, 50),
            ]
            infoset_data[key] = {
                "regrets": np.random.randn(len(actions)).astype(np.float32),
                "strategy_sum": np.random.rand(len(actions)).astype(np.float32),
                "legal_actions": actions,
                "reach_count": np.random.randint(1, 100),
                "cumulative_utility": np.random.randn() * 10,
            }

        # Add unique infosets (only this worker sees)
        for i in range(num_unique_per_worker):
            key = InfoSetKey(
                player_position=i % 2,
                street=Street.RIVER if i % 2 == 0 else Street.TURN,
                betting_sequence=f"w{worker_id}-{i}",
                preflop_hand=None,
                postflop_bucket=i % 50,
                spr_bucket=1,
            )

            # Variable action counts (realistic)
            num_actions = np.random.choice([2, 3, 4, 5], p=[0.3, 0.4, 0.2, 0.1])
            actions = [Action(ActionType.FOLD), Action(ActionType.CALL)]
            if num_actions >= 3:
                actions.append(Action(ActionType.RAISE, 50))
            if num_actions >= 4:
                actions.append(Action(ActionType.RAISE, 100))
            if num_actions >= 5:
                actions.append(Action(ActionType.ALL_IN, 200))

            infoset_data[key] = {
                "regrets": np.random.randn(len(actions)).astype(np.float32),
                "strategy_sum": np.random.rand(len(actions)).astype(np.float32),
                "legal_actions": actions,
                "reach_count": np.random.randint(1, 100),
                "cumulative_utility": np.random.randn() * 10,
            }

        worker_results.append(
            {
                "worker_id": worker_id,
                "batch_id": 0,
                "utilities": [np.random.randn() for _ in range(50)],
                "infoset_data": infoset_data,
                "num_infosets": len(infoset_data),
            }
        )

    total_unique = sum(len(cast(Dict[Any, Any], r["infoset_data"])) for r in worker_results)
    all_keys: set[Any] = set()
    for r in worker_results:
        infoset_data = cast(Dict[Any, Any], r["infoset_data"])
        all_keys.update(infoset_data.keys())

    print(f"Generated {total_unique} total infoset entries")
    print(f"  → {len(all_keys)} unique infosets (after deduplication)")
    print(f"  → {overlap * 100:.1f}% overlap rate")

    return worker_results


def benchmark_merge(
    num_workers: int = 4, infosets_per_worker: int = 100000, profile: bool = False
) -> Dict[str, Any]:
    """
    Benchmark merge performance with timing and optional profiling.

    Args:
        num_workers: Number of workers to simulate
        infosets_per_worker: Infosets per worker
        profile: Whether to run cProfile analysis
    """
    print(f"\n{'=' * 80}")
    print("BENCHMARK: Merge Performance")
    print(f"{'=' * 80}")
    print("Configuration:")
    print(f"  Workers: {num_workers}")
    print(f"  Infosets per worker: {infosets_per_worker:,}")
    print(f"  Profile mode: {profile}")
    print()

    # Generate mock data
    start_gen = time.time()
    worker_results = generate_mock_worker_results(num_workers, infosets_per_worker)
    gen_time = time.time() - start_gen
    print(f"Data generation took {gen_time:.2f}s\n")

    # Create storage
    storage = InMemoryStorage()

    # Run merge with timing
    print("Starting merge...")
    start_merge = time.time()

    if profile:
        # Profile the merge
        profiler = cProfile.Profile()
        profiler.enable()
        merge_worker_results(storage, cast(List[Dict[str, Any]], worker_results))
        profiler.disable()

        merge_time = time.time() - start_merge

        # Print profiling results
        print(f"\nMerge completed in {merge_time:.2f}s")
        print("\nTop 20 time-consuming functions:")
        print("=" * 80)
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
        ps.print_stats(20)
        print(s.getvalue())
    else:
        merge_worker_results(storage, cast(List[Dict[str, Any]], worker_results))
        merge_time = time.time() - start_merge

    # Calculate throughput
    total_infosets = sum(len(cast(Dict[Any, Any], r["infoset_data"])) for r in worker_results)
    throughput = total_infosets / merge_time if merge_time > 0 else 0

    print(f"\n{'=' * 80}")
    print("RESULTS:")
    print(f"{'=' * 80}")
    print(f"Merge time: {merge_time:.2f}s")
    print(f"Throughput: {throughput:,.0f} infosets/sec")
    print(f"Final storage size: {storage.num_infosets():,} infosets")
    print(f"Average time per infoset: {merge_time / total_infosets * 1000:.3f}ms")
    print(f"{'=' * 80}\n")

    return {
        "merge_time": merge_time,
        "throughput": throughput,
        "total_infosets": total_infosets,
        "unique_infosets": storage.num_infosets(),
    }


def run_comparison_benchmarks():
    """Run a series of benchmarks with different scales to compare performance."""
    print(f"\n{'#' * 80}")
    print("# MERGE PERFORMANCE COMPARISON")
    print(f"{'#' * 80}\n")

    class RunConfig(TypedDict):
        workers: int
        infosets: int
        name: str

    configs: List[RunConfig] = [
        {"workers": 2, "infosets": 10000, "name": "Small (2 workers, 10K each)"},
        {"workers": 4, "infosets": 50000, "name": "Medium (4 workers, 50K each)"},
        {"workers": 4, "infosets": 100000, "name": "Large (4 workers, 100K each)"},
        {"workers": 8, "infosets": 100000, "name": "Very Large (8 workers, 100K each)"},
    ]

    results = []
    for config in configs:
        print(f"\n{'=' * 80}")
        print(f"Testing: {config['name']}")
        print(f"{'=' * 80}")

        result = benchmark_merge(
            num_workers=config["workers"],
            infosets_per_worker=config["infosets"],
            profile=False,
        )
        result["config"] = config["name"]
        results.append(result)

        # Brief pause between tests
        time.sleep(1)

    # Print comparison table
    print(f"\n{'=' * 80}")
    print("COMPARISON TABLE")
    print(f"{'=' * 80}")
    print(f"{'Configuration':<40} {'Time':<12} {'Throughput':<20} {'Total':<15}")
    print(f"{'-' * 40} {'-' * 12} {'-' * 20} {'-' * 15}")

    for r in results:
        print(
            f"{r['config']:<40} "
            f"{r['merge_time']:>10.2f}s "
            f"{r['throughput']:>18,.0f}/s "
            f"{r['total_infosets']:>13,}"
        )

    print(f"{'=' * 80}\n")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Profile merge performance")
    parser.add_argument(
        "--mode",
        choices=["single", "comparison", "profile"],
        default="single",
        help="Benchmark mode (default: single)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Number of workers (default: 4)")
    parser.add_argument(
        "--infosets", type=int, default=100000, help="Infosets per worker (default: 100000)"
    )

    args = parser.parse_args()

    if args.mode == "comparison":
        run_comparison_benchmarks()
    elif args.mode == "profile":
        benchmark_merge(
            num_workers=int(args.workers), infosets_per_worker=int(args.infosets), profile=True
        )
    else:
        benchmark_merge(
            num_workers=int(args.workers), infosets_per_worker=int(args.infosets), profile=False
        )


if __name__ == "__main__":
    main()
