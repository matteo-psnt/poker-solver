#!/usr/bin/env python3
"""
Analyze profiling results from training runs.

Usage:
    python scripts/analyze_profile.py <profile_file>

Example:
    python scripts/analyze_profile.py data/profiles/profile_run-20241225-120000.prof
"""

import pstats
import sys
from pathlib import Path


def main():
    """Analyze and display profiling results."""
    if len(sys.argv) != 2:
        print("Usage: python scripts/analyze_profile.py <profile_file>")
        print("\nExample:")
        print("  python scripts/analyze_profile.py data/profiles/profile_run-20241225-120000.prof")
        sys.exit(1)

    profile_file = Path(sys.argv[1])

    if not profile_file.exists():
        print(f"Error: Profile file not found: {profile_file}")
        sys.exit(1)

    # Load stats
    stats = pstats.Stats(str(profile_file))

    # Generate report
    print("=" * 80)
    print(f"Profile Analysis: {profile_file.name}")
    print("=" * 80)

    # Top functions by cumulative time
    print("\n" + "=" * 80)
    print("Top 30 functions by cumulative time:")
    print("=" * 80)
    stats.sort_stats("cumulative")
    stats.print_stats(30)

    # Top functions by internal time
    print("\n" + "=" * 80)
    print("Top 30 functions by internal time (tottime):")
    print("=" * 80)
    stats.sort_stats("tottime")
    stats.print_stats(30)

    # Callers of expensive functions
    print("\n" + "=" * 80)
    print("Callers of merge_worker_results:")
    print("=" * 80)
    stats.print_callers("merge_worker_results", 10)

    print("\n" + "=" * 80)
    print("Callers of train_iteration:")
    print("=" * 80)
    stats.print_callers("train_iteration", 10)

    print("\n" + "=" * 80)
    print("Analysis complete!")
    print("=" * 80)
    print(f"\nProfile file: {profile_file}")
    print(f"Interactive mode: python -m pstats {profile_file}")


if __name__ == "__main__":
    main()
