#!/usr/bin/env python3
"""
Train poker AI and immediately show what it learned.

This script trains the AI and then displays strategies for interesting situations
while they're still in memory.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
from treys import Card as TreysCard

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.abstraction.action_abstraction import ActionAbstraction
from src.abstraction.card_abstraction import RankBasedBucketing
from src.abstraction.infoset import InfoSetKey
from src.game.actions import ActionType
from src.game.state import Card, Street
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import InMemoryStorage
from src.utils.config import Config


def parse_card(card_str: str) -> Card:
    """Parse card string like 'Ah' or 'Kd' into Card object."""
    rank_map = {
        "A": "A", "K": "K", "Q": "Q", "J": "J", "T": "T",
        "9": "9", "8": "8", "7": "7", "6": "6",
        "5": "5", "4": "4", "3": "3", "2": "2",
    }
    suit_map = {"h": "h", "d": "d", "c": "c", "s": "s"}

    if len(card_str) != 2:
        raise ValueError(f"Invalid card: {card_str}")

    rank = rank_map.get(card_str[0].upper())
    suit = suit_map.get(card_str[1].lower())

    if not rank or not suit:
        raise ValueError(f"Invalid card: {card_str}")

    treys_card = TreysCard.new(rank + suit)
    return Card(treys_card)


def format_action(action) -> str:
    """Format action for display."""
    if action.type == ActionType.FOLD:
        return "Fold"
    elif action.type == ActionType.CHECK:
        return "Check"
    elif action.type == ActionType.CALL:
        return "Call"
    elif action.type == ActionType.BET:
        return f"Bet {action.amount}"
    elif action.type == ActionType.RAISE:
        return f"Raise {action.amount}"
    elif action.type == ActionType.ALL_IN:
        return f"All-in ({action.amount})"
    else:
        return str(action)


def show_strategy_for_situation(
    solver: MCCFRSolver,
    name: str,
    hole_cards: tuple,
    street: Street,
    betting_sequence: str,
    position: int,
):
    """Show strategy for a specific situation."""
    print(f"\n{name}")
    print(f"  Cards: {hole_cards[0]} {hole_cards[1]}")
    print(f"  Position: {'BTN (Button)' if position == 0 else 'BB (Big Blind)'}")
    print(f"  Street: {street.name}")
    print(f"  Situation: {betting_sequence if betting_sequence else '(first to act)'}")
    print()

    # Get card bucket
    card_abs = solver.card_abstraction
    board = tuple()  # Empty for preflop

    bucket = card_abs.get_bucket(hole_cards, board, street)
    spr_bucket = 1  # Assume medium SPR

    # Create infoset key
    infoset_key = InfoSetKey(
        player_position=position,
        street=street,
        betting_sequence=betting_sequence,
        card_bucket=bucket,
        spr_bucket=spr_bucket,
    )

    # Get strategy
    infoset = solver.storage.get_infoset(infoset_key)

    if infoset is None:
        print(f"  ❌ No strategy found (situation not encountered during training)")
        return

    print(f"  ✅ Strategy learned (visited {infoset.reach_count} times):")
    print()

    # Get and display strategy
    strategy = infoset.get_average_strategy()
    actions = infoset.legal_actions

    # Sort by probability
    action_probs = list(zip(actions, strategy))
    action_probs.sort(key=lambda x: x[1], reverse=True)

    for action, prob in action_probs:
        if prob > 0.01:  # Only show >1% probability
            bar_length = int(prob * 40)
            bar = "█" * bar_length
            print(f"    {format_action(action):20s} {prob:6.1%}  {bar}")


def show_learned_strategies(solver: MCCFRSolver):
    """Display interesting strategies the AI actually learned."""
    print("\n" + "=" * 70)
    print("LEARNED STRATEGIES - Real Examples")
    print("=" * 70)
    print("\nShowing strategies by hand strength (from strong to weak):\n")

    storage = solver.storage

    # Group infosets by card bucket and street
    preflop_by_bucket = {}

    for key, infoset in storage.infosets.items():
        if key.street == Street.PREFLOP and infoset.reach_count >= 5:
            bucket = key.card_bucket
            if bucket not in preflop_by_bucket:
                preflop_by_bucket[bucket] = []
            preflop_by_bucket[bucket].append((key, infoset))

    # Show one example from each bucket (sorted by bucket = hand strength)
    shown_count = 0
    for bucket in sorted(preflop_by_bucket.keys(), reverse=True):
        if shown_count >= 10:  # Show max 10 examples
            break

        infosets = preflop_by_bucket[bucket]
        # Pick the one visited most
        infosets.sort(key=lambda x: x[1].reach_count, reverse=True)
        key, infoset = infosets[0]

        shown_count += 1

        # Describe hand strength
        total_buckets = 6  # From RankBasedBucketing
        strength_pct = (bucket / total_buckets) * 100

        if strength_pct >= 83:
            strength = "Premium (AA-TT, AK)"
        elif strength_pct >= 66:
            strength = "Strong (99-66, AQ-AJ)"
        elif strength_pct >= 50:
            strength = "Medium (55-22, KQ-KJ, suited connectors)"
        elif strength_pct >= 33:
            strength = "Weak (Ax, Kx suited)"
        else:
            strength = "Trash (offsuit junk)"

        print(f"{shown_count}. Hand Strength: {strength}")
        print(f"   Position: {'BTN' if key.player_position == 0 else 'BB'}")
        print(f"   Situation: {key.betting_sequence if key.betting_sequence else 'first to act'}")
        print(f"   Bucket: {bucket} (higher = stronger)")
        print(f"   Times seen: {infoset.reach_count}")
        print()

        # Show strategy
        strategy = infoset.get_average_strategy()
        actions = infoset.legal_actions

        action_probs = list(zip(actions, strategy))
        action_probs.sort(key=lambda x: x[1], reverse=True)

        for action, prob in action_probs[:3]:  # Top 3 actions
            if prob > 0.01:
                bar_length = int(prob * 40)
                bar = "█" * bar_length
                print(f"     {format_action(action):20s} {prob:6.1%}  {bar}")

        print()

    if shown_count == 0:
        print("  (Not enough training yet - try more iterations)")

    print("=" * 70)


def show_strategy_summary(solver: MCCFRSolver):
    """Show high-level summary of learned strategies."""
    storage = solver.storage
    total_infosets = storage.num_infosets()

    print("\n" + "=" * 70)
    print("STRATEGY SUMMARY")
    print("=" * 70)
    print(f"\nTotal situations learned: {total_infosets:,}")

    # Sample some infosets to show diversity
    print(f"\nSample of learned situations:")

    count = 0
    for key, infoset in storage.infosets.items():
        if count >= 5:
            break

        strategy = infoset.get_average_strategy()
        dominant_action_idx = np.argmax(strategy)
        dominant_prob = strategy[dominant_action_idx]
        dominant_action = infoset.legal_actions[dominant_action_idx]

        print(f"\n  Situation {count + 1}:")
        print(f"    Position: {key.player_position}")
        print(f"    Street: {key.street.name}")
        print(f"    Betting: {key.betting_sequence if key.betting_sequence else 'first to act'}")
        print(f"    Hand type: Bucket {key.card_bucket}")
        print(f"    Preferred action: {format_action(dominant_action)} ({dominant_prob:.1%})")
        print(f"    Times visited: {infoset.reach_count}")

        count += 1


def main():
    parser = argparse.ArgumentParser(
        description="Train poker AI and show what it learned",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--iterations",
        "-n",
        type=int,
        default=1000,
        help="Number of training iterations (default: 1000)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    parser.add_argument(
        "--no-examples",
        action="store_true",
        help="Skip showing example strategies (only show summary)",
    )

    args = parser.parse_args()

    print("=" * 70)
    print("POKER AI TRAINER")
    print("=" * 70)
    print(f"\nTraining for {args.iterations:,} iterations...")
    print("(This will take a moment - the AI is learning optimal poker strategy)\n")

    # Setup
    np.random.seed(args.seed)
    action_abs = ActionAbstraction()
    card_abs = RankBasedBucketing()
    storage = InMemoryStorage()

    # Create solver
    solver = MCCFRSolver(
        action_abs,
        card_abs,
        storage,
        config={"seed": args.seed, "starting_stack": 200},
    )

    # Train
    solver.train(num_iterations=args.iterations, verbose=True)

    # Show what it learned
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE!")
    print("=" * 70)
    print(f"\nThe AI learned {solver.num_infosets():,} different game situations.")
    print("Now let's see what strategies it discovered...\n")

    # Show summary
    show_strategy_summary(solver)

    # Show interesting examples
    if not args.no_examples:
        show_learned_strategies(solver)

    print("\n" + "=" * 70)
    print("What does this mean?")
    print("=" * 70)
    print("""
The AI has learned:
- Which hands to play aggressively vs passively
- When to fold weak hands vs defend
- Approximate Nash equilibrium strategy for heads-up poker

With more iterations (10K-1M), the strategy gets closer to perfect play.

Note: Strategies are currently lost after this script ends (in-memory only).
Phase 6 will add persistent storage to save and reload trained AIs.
    """)

    return 0


if __name__ == "__main__":
    sys.exit(main())
