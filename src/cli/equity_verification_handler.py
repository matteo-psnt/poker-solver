"""Equity bucket verification and visualization handler."""

import numpy as np
import questionary

from src.abstraction.equity.equity_bucketing import EquityBucketing
from src.abstraction.equity.manager import EquityBucketManager
from src.game.state import Card, Street


def handle_verify_equity_buckets(style):
    """
    Handle viewing and verifying equity bucket contents.

    Args:
        style: Questionary style
    """
    print("\nVerify Equity Buckets")
    print("=" * 80)

    # Get available equity bucket sets
    manager = EquityBucketManager()
    abstractions = manager.list_abstractions()

    if not abstractions:
        print("\n[ERROR] No equity buckets found.")
        print("   Please precompute equity buckets first.")
        input("\nPress Enter to continue...")
        return

    # Let user select which bucket set to verify
    choices = [f"{name} ({metadata.created_at[:10]})" for name, _, metadata in abstractions]
    choices.append("Cancel")

    selected = questionary.select(
        "Select equity bucket set to verify:",
        choices=choices,
        style=style,
    ).ask()

    if selected == "Cancel" or selected is None:
        return

    # Extract the name from the selection
    selected_idx = choices.index(selected)
    name, path, metadata = abstractions[selected_idx]

    # Load the equity bucketing
    abstraction_file = path / "abstraction.pkl"
    if not abstraction_file.exists():
        # Try legacy name
        abstraction_file = path / "bucketing.pkl"
        if not abstraction_file.exists():
            print(f"\n[ERROR] Abstraction file not found in {path}")
            input("\nPress Enter to continue...")
            return

    print(f"\nLoading equity buckets from: {name}")

    try:
        bucketing = EquityBucketing.load(abstraction_file)
    except ModuleNotFoundError as e:
        print(f"\n[ERROR] Failed to load equity buckets: {e}")
        print("\n[!] This may be due to outdated pickle files from an older version.")
        print("    Please re-precompute the equity buckets using the current codebase.")
        input("\nPress Enter to continue...")
        return
    except Exception as e:
        print(f"\n[ERROR] Failed to load equity buckets: {e}")
        input("\nPress Enter to continue...")
        return

    # Show verification menu
    while True:
        action = questionary.select(
            "\nWhat would you like to view?",
            choices=[
                "Summary Statistics",
                "Bucket Distribution Analysis",
                "Sample Hand Buckets",
                "Board Cluster Analysis",
                "Specific Hand Lookup",
                "Configuration Details",
                "Back to Main Menu",
            ],
            style=style,
        ).ask()

        if action is None or "Back" in action:
            break

        if "Summary" in action:
            _show_summary_statistics(bucketing, metadata)
        elif "Distribution" in action:
            _show_bucket_distribution(bucketing, metadata)
        elif "Sample Hand" in action:
            _show_sample_hands(bucketing)
        elif "Board Cluster" in action:
            _show_board_clusters(bucketing)
        elif "Specific Hand" in action:
            _lookup_specific_hand(bucketing, style)
        elif "Configuration" in action:
            _show_configuration(metadata)

        input("\nPress Enter to continue...")


def _show_summary_statistics(bucketing: EquityBucketing, metadata):
    """Display summary statistics about the equity buckets."""
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)

    print(f"\nAbstraction: {metadata.name}")
    print(f"Created: {metadata.created_at}")
    if metadata.aliases:
        print(f"Aliases: {', '.join(metadata.aliases)}")

    print("\n" + "-" * 80)
    print("BUCKET CONFIGURATION")
    print("-" * 80)

    for street_name in ["FLOP", "TURN", "RIVER"]:
        street = Street[street_name]
        if street in bucketing.bucket_assignments:
            num_buckets = bucketing.num_buckets(street)
            assignments = bucketing.bucket_assignments[street]
            num_hands, num_board_clusters = assignments.shape

            print(f"\n{street_name}:")
            print(f"  • Buckets: {num_buckets}")
            print(f"  • Board Clusters: {num_board_clusters}")
            print(
                f"  • Total Cells: {num_hands} hands × {num_board_clusters} clusters = {num_hands * num_board_clusters:,}"
            )

            # Bucket usage statistics
            unique_buckets = len(np.unique(assignments))
            print(f"  • Unique Buckets Used: {unique_buckets}/{num_buckets}")

            # Distribution stats
            bucket_counts = np.bincount(assignments.flatten(), minlength=num_buckets)
            avg_per_bucket = bucket_counts.mean()
            std_per_bucket = bucket_counts.std()
            print(f"  • Avg cells per bucket: {avg_per_bucket:.1f} ± {std_per_bucket:.1f}")
            print(f"  • Min/Max cells: {bucket_counts.min()}/{bucket_counts.max()}")

    # Quality metrics if available
    if metadata.empty_clusters or metadata.conflict_defaults:
        print("\n" + "-" * 80)
        print("QUALITY METRICS")
        print("-" * 80)

        if metadata.empty_clusters:
            print("\nEmpty Clusters (boards with no valid samples):")
            for street_name, count in metadata.empty_clusters.items():
                if count > 0:
                    print(f"  {street_name}: {count}")

        if metadata.conflict_defaults:
            print("\nConflict Defaults (card conflicts resolved with 0.5 equity):")
            for street_name, count in metadata.conflict_defaults.items():
                if count > 0:
                    print(f"  {street_name}: {count}")


def _show_bucket_distribution(bucketing: EquityBucketing, metadata):
    """Show distribution of bucket assignments across streets."""
    print("\n" + "=" * 80)
    print("BUCKET DISTRIBUTION ANALYSIS")
    print("=" * 80)

    for street_name in ["FLOP", "TURN", "RIVER"]:
        street = Street[street_name]
        if street not in bucketing.bucket_assignments:
            continue

        print(f"\n{street_name}")
        print("-" * 80)

        assignments = bucketing.bucket_assignments[street]
        num_buckets = bucketing.num_buckets(street)

        # Count assignments per bucket
        bucket_counts = np.bincount(assignments.flatten(), minlength=num_buckets)

        # Show histogram
        print("\nBucket | Count   | Percentage | Bar")
        print("-" * 80)

        max_count = bucket_counts.max()
        for bucket_id in range(num_buckets):
            count = bucket_counts[bucket_id]
            percentage = (count / bucket_counts.sum()) * 100

            # Create bar chart (scaled to 40 chars)
            bar_length = int((count / max_count) * 40) if max_count > 0 else 0
            bar = "█" * bar_length

            print(f"{bucket_id:6d} | {count:7d} | {percentage:6.2f}%   | {bar}")

            # Only show first 20 buckets in detail, then summarize
            if bucket_id == 19 and num_buckets > 25:
                print(f"   ... ({num_buckets - 20} more buckets) ...")
                # Show last 5
                for bid in range(num_buckets - 5, num_buckets):
                    count = bucket_counts[bid]
                    percentage = (count / bucket_counts.sum()) * 100
                    bar_length = int((count / max_count) * 40) if max_count > 0 else 0
                    bar = "█" * bar_length
                    print(f"{bid:6d} | {count:7d} | {percentage:6.2f}%   | {bar}")
                break


def _show_sample_hands(bucketing: EquityBucketing):
    """Show bucket assignments for sample poker hands."""
    print("\n" + "=" * 80)
    print("SAMPLE HAND BUCKETS")
    print("=" * 80)

    # Sample hands to demonstrate
    sample_hands = [
        ("AA", "Pocket Aces"),
        ("KK", "Pocket Kings"),
        ("AKs", "Ace-King suited"),
        ("AKo", "Ace-King offsuit"),
        ("QQ", "Pocket Queens"),
        ("JTs", "Jack-Ten suited"),
        ("76s", "Seven-Six suited"),
        ("72o", "Seven-Two offsuit (worst hand)"),
    ]

    # Sample board for demonstration
    sample_board_flop = (Card.new("Ah"), Card.new("Kd"), Card.new("Qc"))
    sample_board_turn = sample_board_flop + (Card.new("7s"),)
    sample_board_river = sample_board_turn + (Card.new("2h"),)

    print("\nSample Board:")
    print(f"  Flop:  {_format_board(sample_board_flop)}")
    print(f"  Turn:  {_format_board(sample_board_turn)}")
    print(f"  River: {_format_board(sample_board_river)}")

    print("\n" + "-" * 80)
    print(f"{'Hand':<10} {'Description':<30} {'Flop':<8} {'Turn':<8} {'River':<8}")
    print("-" * 80)

    for hand_str, description in sample_hands:
        # Get example hole cards for this hand type
        hole_cards = bucketing._get_example_hand(hand_str)

        # Check for conflicts and get buckets
        buckets = {}
        for street_name, board in [
            ("FLOP", sample_board_flop),
            ("TURN", sample_board_turn),
            ("RIVER", sample_board_river),
        ]:
            street = Street[street_name]
            if street in bucketing.bucket_assignments:
                try:
                    # Check for card conflicts
                    if _has_card_conflict(hole_cards, board):
                        buckets[street_name] = "CONFLICT"
                    else:
                        bucket = bucketing.get_bucket(hole_cards, board, street)
                        buckets[street_name] = f"{bucket:3d}"
                except Exception:
                    buckets[street_name] = "ERROR"
            else:
                buckets[street_name] = "N/A"

        print(
            f"{hand_str:<10} {description:<30} {buckets.get('FLOP', 'N/A'):<8} "
            f"{buckets.get('TURN', 'N/A'):<8} {buckets.get('RIVER', 'N/A'):<8}"
        )


def _show_board_clusters(bucketing: EquityBucketing):
    """Show information about board clustering."""
    print("\n" + "=" * 80)
    print("BOARD CLUSTER ANALYSIS")
    print("=" * 80)

    if not hasattr(bucketing, "board_clusterer") or bucketing.board_clusterer is None:
        print("\n[INFO] Board cluster details not available in saved abstraction.")
        print("       Board clustering was performed during precomputation.")
        return

    for street_name in ["FLOP", "TURN", "RIVER"]:
        street = Street[street_name]
        if street not in bucketing.bucket_assignments:
            continue

        print(f"\n{street_name}")
        print("-" * 80)

        num_clusters = bucketing.bucket_assignments[street].shape[1]
        print(f"Number of board clusters: {num_clusters}")

        # Board features used for clustering
        print("\nBoard clustering groups similar board textures together based on:")
        print("  • Flush potential (suited boards)")
        print("  • Straight potential (connected boards)")
        print("  • Pair structure (paired boards)")
        print("  • High card distribution")

        print(f"\nThis reduces {_estimate_total_boards(street)} possible boards")
        print(f"down to {num_clusters} representative clusters for efficiency.")


def _lookup_specific_hand(bucketing: EquityBucketing, style):
    """Allow user to lookup a specific hand and board combination."""
    print("\n" + "=" * 80)
    print("SPECIFIC HAND LOOKUP")
    print("=" * 80)

    # Get hand input
    hand_str = questionary.text(
        "Enter preflop hand (e.g., AA, AKs, QJo, 72o):",
        style=style,
    ).ask()

    if hand_str is None:
        return

    # Get street
    street_name = questionary.select(
        "Select street:",
        choices=["FLOP", "TURN", "RIVER"],
        style=style,
    ).ask()

    if street_name is None:
        return

    street = Street[street_name]

    if street not in bucketing.bucket_assignments:
        print(f"\n[ERROR] {street_name} buckets not available in this abstraction.")
        return

    # Get board input
    print(f"\nEnter {street_name.lower()} board cards:")
    board_str = questionary.text(
        f"Board (e.g., AhKdQc{' 7s' if street != Street.FLOP else ''}{' 2h' if street == Street.RIVER else ''}):",
        style=style,
    ).ask()

    if board_str is None:
        return

    try:
        # Parse hand (get from bucketing or use a helper)
        hole_cards = _parse_hand_string(hand_str.strip())

        # Parse board
        board_cards = _parse_board(board_str.strip())

        # Validate board size
        expected_size = 3 if street == Street.FLOP else (4 if street == Street.TURN else 5)
        if len(board_cards) != expected_size:
            print(
                f"\n[ERROR] {street_name} requires {expected_size} board cards, got {len(board_cards)}"
            )
            return

        # Check conflicts
        if _has_card_conflict(hole_cards, board_cards):
            print("\n[ERROR] Card conflict detected! Hand and board share cards.")
            return

        # Get bucket
        bucket = bucketing.get_bucket(hole_cards, board_cards, street)

        print("\n" + "-" * 80)
        print("RESULT")
        print("-" * 80)
        print(f"Hand: {hand_str}")
        print(f"Board: {_format_board(board_cards)}")
        print(f"Street: {street_name}")
        print(f"Bucket: {bucket}")

        # Show bucket statistics
        assignments = bucketing.bucket_assignments[street]
        bucket_counts = np.bincount(assignments.flatten())
        total_cells = assignments.size
        bucket_size = bucket_counts[bucket]
        percentage = (bucket_size / total_cells) * 100

        print(
            f"\nBucket {bucket} contains {bucket_size:,} of {total_cells:,} total cells ({percentage:.2f}%)"
        )

    except Exception as e:
        print(f"\n[ERROR] {e}")


def _show_configuration(metadata):
    """Show detailed configuration of the equity buckets."""
    print("\n" + "=" * 80)
    print("CONFIGURATION DETAILS")
    print("=" * 80)

    print(f"\nName: {metadata.name}")
    print(f"Type: {metadata.abstraction_type}")
    print(f"Created: {metadata.created_at}")

    if metadata.aliases:
        print(f"Aliases: {', '.join(metadata.aliases)}")

    print("\n" + "-" * 80)
    print("PRECOMPUTATION PARAMETERS")
    print("-" * 80)

    print(f"\nEquity Samples: {metadata.num_equity_samples:,}")
    print("  (Monte Carlo samples per equity calculation)")

    print(f"\nSamples per Cluster: {metadata.num_samples_per_cluster}")
    print("  (Board samples per cluster for representative selection)")

    print(f"\nRandom Seed: {metadata.seed}")
    print("  (For reproducibility)")

    print("\n" + "-" * 80)
    print("ABSTRACTION SIZE")
    print("-" * 80)

    for street_name in ["FLOP", "TURN", "RIVER"]:
        buckets = metadata.num_buckets.get(street_name, 0)
        clusters = metadata.num_board_clusters.get(street_name, 0)

        if buckets > 0:
            total_cells = 169 * clusters
            storage_bytes = total_cells  # 1 byte per cell

            print(f"\n{street_name}:")
            print(f"  • Buckets: {buckets}")
            print(f"  • Board Clusters: {clusters}")
            print(f"  • Matrix Size: 169 × {clusters} = {total_cells:,} cells")
            print(f"  • Storage: ~{storage_bytes / 1024:.1f} KB")


def _format_board(board: tuple) -> str:
    """Format board cards as string."""
    return " ".join(str(card) for card in board)


def _parse_hand_string(hand_str: str) -> tuple:
    """
    Parse hand string to get example hole cards.

    Args:
        hand_str: Hand string like "AA", "AKs", "72o"

    Returns:
        Tuple of two Card objects
    """
    if len(hand_str) == 2:
        # Pair
        rank = hand_str[0]
        return (Card.new(f"{rank}h"), Card.new(f"{rank}d"))
    else:
        # Suited or offsuit
        high_rank = hand_str[0]
        low_rank = hand_str[1]
        suited = hand_str[2] == "s"

        if suited:
            return (Card.new(f"{high_rank}s"), Card.new(f"{low_rank}s"))
        else:
            return (Card.new(f"{high_rank}h"), Card.new(f"{low_rank}d"))


def _parse_board(board_str: str) -> tuple:
    """Parse board string into tuple of Card objects."""
    cards = []
    # Split by spaces or parse pairs of characters
    parts = board_str.strip().split()

    for part in parts:
        if len(part) >= 2:
            cards.append(Card.new(part[:2]))

    return tuple(cards)


def _has_card_conflict(hole_cards: tuple, board: tuple) -> bool:
    """Check if hole cards conflict with board cards."""
    hole_set = set(hole_cards)
    board_set = set(board)
    return len(hole_set & board_set) > 0


def _estimate_total_boards(street: Street) -> str:
    """Estimate total possible boards for a street."""
    if street == Street.FLOP:
        # C(50, 3) after removing 2 hole cards
        return "~19,600"
    elif street == Street.TURN:
        # C(50, 4)
        return "~230,300"
    elif street == Street.RIVER:
        # C(50, 5)
        return "~2,118,760"
    return "Unknown"
