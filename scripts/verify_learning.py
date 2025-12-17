#!/usr/bin/env python3
"""
Quick script to verify that learning occurred by checking strategy_sum values.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.abstraction.infoset import InfoSetKey
from src.game.state import Street
from src.solver.storage import DiskBackedStorage
from src.training.checkpoint import CheckpointManager


def main():
    checkpoint_dir = Path("data/checkpoints")

    # List all runs
    runs = CheckpointManager.list_runs(checkpoint_dir)
    if not runs:
        print("No training runs found")
        return

    print("Available runs:")
    for i, run_id in enumerate(runs, 1):
        print(f"  {i}. {run_id}")

    # Check each run
    for run_id in runs:
        print(f"\n{'=' * 60}")
        print(f"Checking run: {run_id}")
        print(f"{'=' * 60}")

        # Load storage
        run_checkpoint_dir = checkpoint_dir / run_id
        storage = DiskBackedStorage(
            checkpoint_dir=run_checkpoint_dir,
            cache_size=100000,
            flush_frequency=1000,
        )

        num_infosets = storage.num_infosets()
        print(f"Loaded {num_infosets:,} infosets from disk")

        if num_infosets == 0:
            print("  -> No data saved in this run")
            continue

        # Check premium hands (bucket 5) - AA, KK, QQ, JJ, TT, AK all map to this bucket
        print("\nCHECKING PREMIUM HANDS (Bucket 5):")

        key = InfoSetKey(
            player_position=0,
            street=Street.PREFLOP,
            betting_sequence="",
            card_bucket=5,
            spr_bucket=2,
        )

        infoset = storage.get_infoset(key)
        if infoset:
            print(f"  InfoSet found: {key}")
            print(f"  Legal actions: {[str(a) for a in infoset.legal_actions]}")
            print(f"\n  reach_count (NOW PERSISTED): {infoset.reach_count}")
            print(f"  cumulative_utility: {infoset.cumulative_utility:.2f}")
            print(f"  average_utility: {infoset.get_average_utility():+.4f}")
            print(f"  strategy_sum (PERSISTED): {infoset.strategy_sum}")
            print(f"  Sum of strategy_sum: {infoset.strategy_sum.sum():.1f}")

            avg_strategy = infoset.get_average_strategy()
            print("\n  Average strategy (what the AI learned):")
            for action, prob in zip(infoset.legal_actions, avg_strategy):
                print(f"    {action}: {prob:.1%}")
        else:
            print("  No InfoSet found for premium hands")

        # Check medium hands
        print("\nCHECKING MEDIUM HANDS (Bucket 3):")
        key = InfoSetKey(
            player_position=0,
            street=Street.PREFLOP,
            betting_sequence="",
            card_bucket=3,
            spr_bucket=2,
        )

        infoset = storage.get_infoset(key)
        if infoset:
            avg_strategy = infoset.get_average_strategy()
            print("  Average strategy:")
            for action, prob in zip(infoset.legal_actions, avg_strategy):
                print(f"    {action}: {prob:.1%}")
        else:
            print("  No InfoSet found for medium hands")

    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("=" * 60)
    print("""
reach_count and cumulative_utility are NOW PERSISTED to disk!
- reach_count: Shows how many times each situation was encountered during training
- average_utility: Shows the expected value (EV) of each situation

The abstraction system means:
- AA, KK, QQ, JJ, TT, AKs all map to bucket 5 (premium)
- They all share the SAME InfoSet and learn together
- So the AI learned "how to play premium hands" not "how to play AA specifically"

This is the power of abstraction - it allows learning from similar situations!

Note: Old checkpoints won't have reach_count/utility data (will show 0).
      Only new training runs will persist these statistics.
""")


if __name__ == "__main__":
    main()
