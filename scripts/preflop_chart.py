#!/usr/bin/env python3
"""
Generate preflop strategy charts from trained AI.

Shows a grid of all 169 preflop hands with their optimal actions.
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.abstraction.action_abstraction import ActionAbstraction
from src.abstraction.card_abstraction import RankBasedBucketing
from src.abstraction.infoset import InfoSetKey
from src.game.actions import ActionType
from src.game.state import Street
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import DiskBackedStorage
from src.training.checkpoint import CheckpointManager


def get_hand_category(rank1: str, rank2: str, suited: bool) -> str:
    """Get display string for a hand (e.g., 'AKs', 'QJo', '77')."""
    ranks = "AKQJT98765432"

    # Normalize order (higher rank first)
    if ranks.index(rank1) > ranks.index(rank2):
        rank1, rank2 = rank2, rank1

    if rank1 == rank2:
        return f"{rank1}{rank2}"  # Pair
    elif suited:
        return f"{rank1}{rank2}s"
    else:
        return f"{rank1}{rank2}o"


def get_all_hands() -> list:
    """Get all 169 unique preflop hands."""
    ranks = "AKQJT98765432"
    hands = []

    # Pairs
    for rank in ranks:
        hands.append((rank, rank, False, True))  # (rank1, rank2, suited, is_pair)

    # Non-pairs
    for i, r1 in enumerate(ranks):
        for r2 in ranks[i + 1 :]:
            hands.append((r1, r2, True, False))  # Suited
            hands.append((r1, r2, False, False))  # Offsuit

    return hands


def get_hand_bucket(
    rank1: str, rank2: str, suited: bool, is_pair: bool, card_abs: RankBasedBucketing
) -> int:
    """Estimate which bucket a hand falls into."""
    ranks = "AKQJT98765432"
    r1_val = 14 - ranks.index(rank1)
    r2_val = 14 - ranks.index(rank2)

    # Pairs
    if is_pair:
        if r1_val >= 10:  # TT+
            return 5  # Premium
        elif r1_val >= 6:  # 66-99
            return 4  # Strong
        else:  # 22-55
            return 3  # Medium

    # Non-pairs
    high = max(r1_val, r2_val)
    low = min(r1_val, r2_val)
    gap = high - low

    # AK
    if high == 14 and low == 13:
        return 5  # Premium

    # AQ, AJ
    if high == 14 and low >= 11:
        return 4  # Strong

    # Broadway (KQ, KJ, QJ)
    if high >= 13 and low >= 11 and suited:
        return 4  # Strong

    # Suited connectors and one-gappers
    if suited and gap <= 2:
        return 3  # Medium

    # Ace-x
    if high == 14 and suited:
        return 3  # Medium

    # King-x suited
    if high == 13 and suited:
        return 2  # Weak

    # Everything else
    return 1  # Trash


def get_strategy_for_hand(
    solver: MCCFRSolver,
    rank1: str,
    rank2: str,
    suited: bool,
    is_pair: bool,
    position: int,
    betting_sequence: str,
) -> dict:
    """Get strategy for a specific hand."""
    card_abs = solver.card_abstraction
    bucket = get_hand_bucket(rank1, rank2, suited, is_pair, card_abs)

    # Try all SPR buckets (0=shallow, 1=medium, 2=deep)
    infoset = None
    for spr_bucket in [2, 1, 0]:  # Try deep first (200BB stacks)
        key = InfoSetKey(
            player_position=position,
            street=Street.PREFLOP,
            betting_sequence=betting_sequence,
            card_bucket=bucket,
            spr_bucket=spr_bucket,
        )

        infoset = solver.storage.get_infoset(key)
        if infoset is not None:
            break

    if infoset is None:
        return None

    strategy = infoset.get_average_strategy()
    actions = infoset.legal_actions

    # Categorize actions
    result = {
        "fold": 0.0,
        "call": 0.0,
        "raise": 0.0,
        "reach_count": infoset.reach_count,
        "avg_utility": infoset.get_average_utility(),
    }

    for action, prob in zip(actions, strategy):
        if action.type == ActionType.FOLD:
            result["fold"] += prob
        elif action.type in [ActionType.CALL, ActionType.CHECK]:
            result["call"] += prob
        elif action.type in [ActionType.RAISE, ActionType.BET, ActionType.ALL_IN]:
            result["raise"] += prob

    return result


def get_action_color(strategy: dict) -> str:
    """Get color code for action (for terminal display)."""
    if strategy is None:
        return "\033[90m"  # Gray (no data)

    fold = strategy["fold"]
    call = strategy["call"]
    raise_prob = strategy["raise"]

    # Determine dominant action
    if raise_prob > 0.6:
        return "\033[92m"  # Green (aggressive)
    elif raise_prob > 0.3:
        return "\033[93m"  # Yellow (mixed)
    elif call > 0.5:
        return "\033[96m"  # Cyan (passive)
    elif fold > 0.5:
        return "\033[91m"  # Red (fold)
    else:
        return "\033[37m"  # White (balanced)


def format_strategy(strategy: dict) -> str:
    """Format strategy as short string."""
    if strategy is None:
        return "???"

    fold = strategy["fold"]
    call = strategy["call"]
    raise_prob = strategy["raise"]

    # Show dominant action
    if raise_prob > 0.6:
        return f"R{raise_prob:.0%}"
    elif call > 0.5:
        return f"C{call:.0%}"
    elif fold > 0.5:
        return f"F{fold:.0%}"
    else:
        # Mixed strategy - show all
        return f"R{raise_prob:.0%}"


def generate_preflop_chart(solver: MCCFRSolver, position: int, betting_sequence: str):
    """Generate preflop chart for a position and situation."""
    ranks = "AKQJT98765432"

    position_name = "BTN (Button)" if position == 0 else "BB (Big Blind)"
    situation = betting_sequence if betting_sequence else "first to act"

    print(f"\n{'='*80}")
    print(f"PREFLOP CHART - {position_name} - {situation}")
    print(f"{'='*80}\n")
    print("Legend: R=Raise, C=Call, F=Fold, ???=No data")
    print("Colors: Green=Aggressive, Yellow=Mixed, Cyan=Passive, Red=Fold, Gray=No data\n")

    # Create 13x13 grid (pairs on diagonal, suited above, offsuit below)
    grid = []

    for i, r1 in enumerate(ranks):
        row = []
        for j, r2 in enumerate(ranks):
            if i == j:
                # Pair
                strategy = get_strategy_for_hand(
                    solver, r1, r2, False, True, position, betting_sequence
                )
                hand = f"{r1}{r2}"
            elif i < j:
                # Suited (above diagonal)
                strategy = get_strategy_for_hand(
                    solver, r1, r2, True, False, position, betting_sequence
                )
                hand = f"{r1}{r2}s"
            else:
                # Offsuit (below diagonal)
                strategy = get_strategy_for_hand(
                    solver, r2, r1, False, False, position, betting_sequence
                )
                hand = f"{r2}{r1}o"

            color = get_action_color(strategy)
            action_str = format_strategy(strategy)
            row.append((hand, strategy, color, action_str))
        grid.append(row)

    # Print header
    print("     ", end="")
    for rank in ranks:
        print(f"{rank:6s}", end="")
    print("\n")

    # Print grid
    for i, (rank, row) in enumerate(zip(ranks, grid)):
        print(f"{rank:3s}  ", end="")
        for hand, strategy, color, action_str in row:
            reset = "\033[0m"
            print(f"{color}{action_str:6s}{reset}", end="")
        print()

    print(f"\n{'='*80}\n")


def generate_detailed_stats(solver: MCCFRSolver, position: int, betting_sequence: str):
    """Show detailed statistics for hand ranges."""
    print(f"\nDETAILED STATISTICS\n{'='*80}\n")

    hands = get_all_hands()
    strategies = []

    for rank1, rank2, suited, is_pair in hands:
        strategy = get_strategy_for_hand(
            solver, rank1, rank2, suited, is_pair, position, betting_sequence
        )
        if strategy:
            hand_str = get_hand_category(rank1, rank2, suited)
            strategies.append((hand_str, strategy))

    if not strategies:
        print("No data available for this situation.")
        return

    # Calculate aggregate stats
    total_hands = len(strategies)
    avg_raise = np.mean([s[1]["raise"] for s in strategies])
    avg_call = np.mean([s[1]["call"] for s in strategies])
    avg_fold = np.mean([s[1]["fold"] for s in strategies])

    print(f"Hands with data: {total_hands}/169")
    print(f"Average raise: {avg_raise:.1%}")
    print(f"Average call: {avg_call:.1%}")
    print(f"Average fold: {avg_fold:.1%}")
    print()

    # Show top raising hands
    strategies.sort(key=lambda x: x[1]["raise"], reverse=True)
    print("Top 10 raising hands:")
    for i, (hand, strat) in enumerate(strategies[:10], 1):
        print(
            f"  {i:2d}. {hand:4s}  Raise: {strat['raise']:5.1%}  "
            f"Call: {strat['call']:5.1%}  Fold: {strat['fold']:5.1%}  "
            f"(seen {strat['reach_count']} times)"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate preflop strategy charts from trained AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show chart for button first to act
  python scripts/preflop_chart.py

  # Show chart for BB facing raise
  python scripts/preflop_chart.py --position BB --sequence r5

  # Use specific run
  python scripts/preflop_chart.py --run run_20251216_134249

  # Show detailed statistics
  python scripts/preflop_chart.py --detailed
        """,
    )

    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("data/checkpoints"),
        help="Checkpoint directory (default: data/checkpoints)",
    )

    parser.add_argument(
        "--run",
        type=str,
        help="Specific run ID to load (default: latest)",
    )

    parser.add_argument(
        "--position",
        choices=["BTN", "BB"],
        default="BTN",
        help="Position (default: BTN)",
    )

    parser.add_argument(
        "--sequence",
        default="",
        help="Betting sequence (default: empty = first to act)",
    )

    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Show detailed statistics",
    )

    args = parser.parse_args()

    # Load solver
    print("Loading trained AI...")

    # Find run
    run_id = args.run
    if run_id is None:
        runs = CheckpointManager.list_runs(args.checkpoint_dir)
        if not runs:
            print(f"Error: No training runs found in {args.checkpoint_dir}")
            return 1
        run_id = runs[-1]
        print(f"  Using latest run: {run_id}")

    # Load storage
    run_checkpoint_dir = args.checkpoint_dir / run_id
    storage = DiskBackedStorage(
        checkpoint_dir=run_checkpoint_dir,
        cache_size=100000,
        flush_frequency=1000,
    )

    print(f"  Loaded {storage.num_infosets():,} infosets from disk")

    # Create solver
    action_abs = ActionAbstraction()
    card_abs = RankBasedBucketing()
    solver = MCCFRSolver(
        action_abstraction=action_abs,
        card_abstraction=card_abs,
        storage=storage,
        config={"seed": 42, "starting_stack": 200},
    )

    # Generate chart
    position = 0 if args.position == "BTN" else 1
    generate_preflop_chart(solver, position, args.sequence)

    if args.detailed:
        generate_detailed_stats(solver, position, args.sequence)

    return 0


if __name__ == "__main__":
    sys.exit(main())
