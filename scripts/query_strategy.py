#!/usr/bin/env python3
"""
Query and display learned poker strategies.

Shows what the trained AI does in specific situations.
"""

import argparse
import sys
from pathlib import Path

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
from src.solver.storage import DiskBackedStorage
from src.training.checkpoint import CheckpointManager


def parse_card(card_str: str) -> Card:
    """Parse card string like 'Ah' or 'Kd' into Card object."""
    # Map to treys format
    rank_map = {
        "A": "A",
        "K": "K",
        "Q": "Q",
        "J": "J",
        "T": "T",
        "9": "9",
        "8": "8",
        "7": "7",
        "6": "6",
        "5": "5",
        "4": "4",
        "3": "3",
        "2": "2",
    }
    suit_map = {"h": "h", "d": "d", "c": "c", "s": "s"}

    if len(card_str) != 2:
        raise ValueError(f"Invalid card: {card_str}")

    rank = rank_map.get(card_str[0].upper())
    suit = suit_map.get(card_str[1].lower())

    if not rank or not suit:
        raise ValueError(f"Invalid card: {card_str}")

    # Create treys card and convert to our Card
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


def show_strategy(
    solver: MCCFRSolver,
    hole_cards: tuple,
    street: Street,
    betting_sequence: str,
    position: int,
):
    """Show strategy for a specific situation."""
    # Get card abstraction bucket
    card_abs = solver.card_abstraction
    board = tuple()  # Empty board for now (preflop)

    bucket = card_abs.get_bucket(hole_cards, board, street)

    # SPR bucket (simplified - assume medium SPR)
    spr_bucket = 1

    # Create infoset key
    infoset_key = InfoSetKey(
        player_position=position,
        street=street,
        betting_sequence=betting_sequence,
        card_bucket=bucket,
        spr_bucket=spr_bucket,
    )

    # Try to get strategy
    infoset = solver.storage.get_infoset(infoset_key)

    if infoset is None:
        print("  ❌ No strategy found for this situation")
        print("     (Situation never encountered during training)")
        return

    # Get average strategy (Nash equilibrium approximation)
    strategy = infoset.get_average_strategy()
    actions = infoset.legal_actions

    print(f"  ✅ Strategy found (visited {infoset.reach_count} times):")
    print()

    # Sort actions by probability (descending)
    action_probs = list(zip(actions, strategy))
    action_probs.sort(key=lambda x: x[1], reverse=True)

    for action, prob in action_probs:
        if prob > 0.01:  # Only show actions with >1% probability
            bar_length = int(prob * 40)
            bar = "█" * bar_length
            print(f"    {format_action(action):20s} {prob:6.1%}  {bar}")


def show_example_situations(solver: MCCFRSolver):
    """Show strategies for some interesting example situations."""
    print("=" * 70)
    print("EXAMPLE STRATEGIES FROM TRAINED AI")
    print("=" * 70)
    print()

    examples = [
        {
            "name": "Pocket Aces - Button - Facing limp",
            "cards": ("Ah", "As"),
            "street": Street.PREFLOP,
            "sequence": "c",  # Opponent limped
            "position": 0,
        },
        {
            "name": "Pocket Kings - Big Blind - Facing 3BB raise",
            "cards": ("Kh", "Kd"),
            "street": Street.PREFLOP,
            "sequence": "r5",  # Opponent raised to 5 (2.5BB)
            "position": 1,
        },
        {
            "name": "Ace-King suited - Button - First to act",
            "cards": ("Ah", "Kh"),
            "street": Street.PREFLOP,
            "sequence": "",  # No action yet
            "position": 0,
        },
        {
            "name": "7-2 offsuit - Big Blind - Facing 3BB raise",
            "cards": ("7h", "2d"),
            "street": Street.PREFLOP,
            "sequence": "r5",
            "position": 1,
        },
        {
            "name": "Pocket Queens - Button - First to act",
            "cards": ("Qh", "Qd"),
            "street": Street.PREFLOP,
            "sequence": "",
            "position": 0,
        },
    ]

    for i, ex in enumerate(examples, 1):
        print(f"{i}. {ex['name']}")
        print(f"   Cards: {ex['cards'][0]} {ex['cards'][1]}")
        print(f"   Position: {'BTN (Button)' if ex['position'] == 0 else 'BB (Big Blind)'}")
        print(f"   Street: {ex['street'].name}")
        print()

        try:
            hole_cards = (parse_card(ex["cards"][0]), parse_card(ex["cards"][1]))
            show_strategy(
                solver,
                hole_cards,
                ex["street"],
                ex["sequence"],
                ex["position"],
            )
        except Exception as e:
            print(f"  ❌ Error: {e}")

        print()


def query_custom(solver: MCCFRSolver, args):
    """Query a custom situation specified by user."""
    print("=" * 70)
    print("CUSTOM STRATEGY QUERY")
    print("=" * 70)
    print()

    # Parse cards
    try:
        card1 = parse_card(args.card1)
        card2 = parse_card(args.card2)
        hole_cards = (card1, card2)
    except ValueError as e:
        print(f"Error parsing cards: {e}")
        return

    # Parse street
    street_map = {
        "preflop": Street.PREFLOP,
        "flop": Street.FLOP,
        "turn": Street.TURN,
        "river": Street.RIVER,
    }
    street = street_map.get(args.street.lower(), Street.PREFLOP)

    # Parse position
    position = 0 if args.position.lower() in ["btn", "button", "0"] else 1

    # Betting sequence
    sequence = args.sequence or ""

    print(f"Cards: {args.card1} {args.card2}")
    print(f"Position: {'BTN' if position == 0 else 'BB'}")
    print(f"Street: {street.name}")
    print(f"Betting sequence: {sequence if sequence else '(first to act)'}")
    print()

    show_strategy(solver, hole_cards, street, sequence, position)


def load_solver(checkpoint_dir: Path, run_id: str = None) -> MCCFRSolver:
    """Load trained solver from checkpoint."""
    print("Loading trained AI...")

    # Find latest run if not specified
    if run_id is None:
        runs = CheckpointManager.list_runs(checkpoint_dir)
        if not runs:
            raise ValueError(f"No training runs found in {checkpoint_dir}")
        run_id = runs[-1]  # Latest run
        print(f"  Using latest run: {run_id}")

    # Load checkpoint metadata
    checkpoint_manager = CheckpointManager.from_run_id(checkpoint_dir, run_id)
    latest_checkpoint = checkpoint_manager.load_latest()

    if latest_checkpoint is None:
        raise ValueError(f"No checkpoints found for run {run_id}")

    print(f"  Checkpoint: iteration {latest_checkpoint.iteration}")
    print(f"  Infosets: {latest_checkpoint.num_infosets:,}")
    print()

    # Rebuild solver with disk-backed storage
    action_abs = ActionAbstraction()
    card_abs = RankBasedBucketing()

    # Create storage pointing to the run-specific directory
    run_checkpoint_dir = checkpoint_dir / run_id
    storage = DiskBackedStorage(
        checkpoint_dir=run_checkpoint_dir,
        cache_size=100000,
        flush_frequency=1000,
    )

    print(f"  Loaded {storage.num_infosets():,} infosets from disk")

    # Create solver with loaded storage
    solver = MCCFRSolver(
        action_abstraction=action_abs,
        card_abstraction=card_abs,
        storage=storage,
        config={"seed": 42, "starting_stack": 200},
    )

    return solver


def main():
    parser = argparse.ArgumentParser(
        description="Query learned poker strategies from trained AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Show example situations from latest training run
  python scripts/query_strategy.py

  # Query specific hand
  python scripts/query_strategy.py --custom --card1 Ah --card2 Kh --position BTN

  # Query from specific training run
  python scripts/query_strategy.py --run run_20251216_122944
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
        "--custom",
        action="store_true",
        help="Query custom situation (requires --card1, --card2, etc.)",
    )

    parser.add_argument("--card1", help="First hole card (e.g., Ah)")
    parser.add_argument("--card2", help="Second hole card (e.g., Kh)")
    parser.add_argument(
        "--position",
        default="BTN",
        help="Position: BTN (button) or BB (big blind)",
    )
    parser.add_argument(
        "--street",
        default="preflop",
        help="Street: preflop, flop, turn, river",
    )
    parser.add_argument(
        "--sequence",
        default="",
        help="Betting sequence (e.g., 'r5' for raise to 5)",
    )

    args = parser.parse_args()

    try:
        # Try to load solver
        solver = load_solver(args.checkpoint_dir, args.run)

        if args.custom:
            if not args.card1 or not args.card2:
                print("Error: --custom requires --card1 and --card2")
                return 1

            query_custom(solver, args)
        else:
            show_example_situations(solver)

    except NotImplementedError as e:
        print("=" * 70)
        print("STRATEGY VIEWER - NOT YET AVAILABLE")
        print("=" * 70)
        print()
        print(str(e))
        print()
        print("WORKAROUND:")
        print("  For now, you can inspect strategies by:")
        print("  1. Modifying train.py to print strategies after training")
        print("  2. Using Python interactively to query the solver object")
        print("  3. Implementing disk-backed storage (Phase 6 of the plan)")
        print()
        print("Example Python code to inspect current training:")
        print("""
  from src.training.trainer import Trainer
  from src.utils.config import Config

  config = Config.default()
  trainer = Trainer(config)
  trainer.train(num_iterations=100)

  # Now inspect strategies
  solver = trainer.solver
  storage = solver.storage

  # Show all infosets
  print(f"Total infosets: {storage.num_infosets()}")

  # Look at specific infoset (you'd need the key)
  for key, infoset in storage.infosets.items():
      print(f"{key}")
      print(f"  Strategy: {infoset.get_average_strategy()}")
      print(f"  Actions: {infoset.legal_actions}")
      break  # Just show first one as example
        """)
        return 1

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
