"""Bucketing analysis flows for combo abstraction CLI."""

import random
from collections import Counter, defaultdict

from src.bucketing.postflop.precompute import PostflopPrecomputer
from src.cli.flows.combo_precompute.common import _select_abstraction
from src.cli.ui import prompts
from src.cli.ui.context import CliContext
from src.game.state import Card, Street


def handle_combo_analyze_bucketing(ctx: CliContext) -> None:
    """Analyze bucketing patterns using predefined and sampled scenarios."""
    print()
    print("=" * 60)
    print("  ANALYZE BUCKETING PATTERNS")
    print("=" * 60)

    abstraction_path, _metadata = _select_abstraction(ctx)
    if abstraction_path is None:
        return

    print(f"\nLoading abstraction from {abstraction_path.name}...")
    try:
        abstraction = PostflopPrecomputer.load(abstraction_path)
        print("✓ Loaded successfully\n")
    except Exception as exc:
        print(f"✗ Failed to load: {exc}")
        return

    street_choice = prompts.select(
        ctx,
        "Select street to analyze:",
        choices=["FLOP", "TURN", "RIVER", "Cancel"],
    )
    if street_choice is None or street_choice == "Cancel":
        return
    street = Street[street_choice]

    analysis_type = prompts.select(
        ctx,
        "Select analysis type:",
        choices=[
            "Premium vs Weak Hands (predefined scenarios)",
            "Random Sample (50 random hand/board combos)",
            "Hand Strength Correlation (various equities)",
            "Back",
        ],
    )
    if analysis_type is None or "Back" in analysis_type:
        return

    print("\n" + "=" * 60)
    print(f"BUCKETING ANALYSIS: {street.name}")
    print("=" * 60)

    if "Premium vs Weak" in analysis_type:
        _analyze_premium_vs_weak(abstraction, street)
    elif "Random Sample" in analysis_type:
        _analyze_random_sample(abstraction, street)
    elif "Hand Strength" in analysis_type:
        _analyze_hand_strength_correlation(abstraction, street)


def _analyze_premium_vs_weak(abstraction, street: Street) -> None:
    """Analyze bucketing for premium vs weak hands."""
    print("\nTesting predefined hand scenarios...")
    print("-" * 60)

    scenarios: list[tuple[tuple[str, str], tuple[str, ...], str, str]]
    if street == Street.FLOP:
        scenarios = [
            (("As", "Ah"), ("Ks", "Qh", "Jc"), "Top set", "STRONG"),
            (("Ac", "Kc"), ("Ad", "Kh", "2s"), "Top two pair", "STRONG"),
            (("Qs", "Qh"), ("Qd", "Jc", "Th"), "Middle set", "STRONG"),
            (("As", "Ks"), ("Ah", "Kh", "2h"), "Flush draw + pair", "MEDIUM"),
            (("Jc", "Tc"), ("Qs", "9h", "2d"), "Open-ended straight draw", "MEDIUM"),
            (("7d", "7c"), ("Ks", "Qh", "Jc"), "Low pocket pair", "WEAK"),
            (("9s", "8s"), ("Kh", "6d", "2c"), "Two overs", "WEAK"),
            (("3h", "2h"), ("Kd", "Qs", "Jc"), "Nothing", "WEAK"),
        ]
    elif street == Street.TURN:
        scenarios = [
            (("As", "Ah"), ("Ks", "Qh", "Jc", "Ac"), "Trips turning into quads area", "STRONG"),
            (("Ac", "Kc"), ("Ad", "Kh", "2s", "2d"), "Two pair improving", "STRONG"),
            (("As", "Ks"), ("Ah", "Kh", "2h", "3h"), "Made flush", "STRONG"),
            (("Jc", "Tc"), ("Qs", "9h", "2d", "8s"), "Made straight", "STRONG"),
            (("9s", "9h"), ("Kh", "6d", "2c", "9c"), "Turned set", "STRONG"),
            (("7d", "6d"), ("Kh", "5d", "2d", "Ac"), "Flush draw", "MEDIUM"),
            (("7d", "7c"), ("Ks", "Qh", "Jc", "Th"), "Pocket pair, no improvement", "WEAK"),
            (("3h", "2h"), ("Kd", "Qs", "Jc", "Tc"), "Nothing", "WEAK"),
        ]
    else:
        scenarios = [
            (("As", "Ah"), ("Ks", "Qh", "Jc", "Ac", "Ad"), "Quads", "STRONG"),
            (("Ac", "Kc"), ("Ad", "Kh", "As", "Ks", "2d"), "Full house (aces full)", "STRONG"),
            (("As", "Ks"), ("Ah", "Kh", "2h", "3h", "9h"), "Nut flush", "STRONG"),
            (("Jc", "Tc"), ("Qs", "9h", "8d", "7s", "2c"), "Straight", "STRONG"),
            (("Ac", "Kc"), ("Ad", "Kh", "2s", "3d", "7c"), "Top two pair", "MEDIUM"),
            (("Qs", "Qh"), ("Kd", "Qc", "Jc", "Th", "2s"), "Set", "MEDIUM"),
            (("7d", "7c"), ("Ks", "Qh", "Jc", "Th", "2d"), "Low pocket pair", "WEAK"),
            (("3h", "2h"), ("Kd", "Qs", "Jc", "Tc", "9s"), "High card", "WEAK"),
        ]

    results = defaultdict(list)
    for hole_str, board_str, description, category in scenarios:
        try:
            hole_cards = tuple([Card.new(c) for c in hole_str])
            board_cards = tuple([Card.new(c) for c in board_str])

            bucket = abstraction.get_bucket(hole_cards, board_cards, street)
            results[category].append((description, bucket))
        except Exception as exc:
            print(f"✗ Error testing '{description}': {exc}")

    for category in ["STRONG", "MEDIUM", "WEAK"]:
        if category in results:
            print(f"\n{category} HANDS:")
            for description, bucket in results[category]:
                print(f"  {description:35s} → Bucket {bucket:3d}")

    all_buckets = [bucket for scenarios in results.values() for _, bucket in scenarios]
    if all_buckets:
        print(f"\nBucket range: {min(all_buckets)} - {max(all_buckets)}")
        print(
            f"Unique buckets used: {len(set(all_buckets))} out of {abstraction.num_buckets(street)}"
        )


def _analyze_random_sample(abstraction, street: Street) -> None:
    """Analyze bucketing for random hand/board combinations."""
    print("\nGenerating 50 random hand/board combinations...")
    print("-" * 60)

    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["h", "d", "c", "s"]
    all_cards = [r + s for r in ranks for s in suits]

    num_board_cards = {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5}[street]
    bucket_counts: Counter[int] = Counter()
    results = []

    for i in range(50):
        sampled = random.sample(all_cards, 2 + num_board_cards)
        hole_str = sampled[:2]
        board_str = sampled[2:]

        try:
            hole_cards = tuple([Card.new(c) for c in hole_str])
            board_cards = tuple([Card.new(c) for c in board_str])

            bucket = abstraction.get_bucket(hole_cards, board_cards, street)
            bucket_counts[bucket] += 1
            results.append((hole_str, board_str, bucket))
        except Exception as exc:
            print(f"✗ Error in sample {i}: {exc}")

    print("\nSample results (first 20):")
    for i, (hole_str, board_str, bucket) in enumerate(results[:20], 1):
        hole_display = "".join(hole_str)
        board_display = " ".join(board_str)
        print(f"  {i:2d}. {hole_display:6s} on {board_display:20s} → Bucket {bucket:3d}")

    if len(results) > 20:
        print(f"\n  ... and {len(results) - 20} more\n")

    print("\nBucket distribution from sample:")
    print(f"  Unique buckets hit: {len(bucket_counts)} out of {abstraction.num_buckets(street)}")
    print("  Most common buckets:")
    for bucket, count in bucket_counts.most_common(10):
        bar = "█" * (count * 2)
        print(f"    Bucket {bucket:3d}: {bar} {count}")


def _analyze_hand_strength_correlation(abstraction, street: Street) -> None:
    """Analyze whether hand strength correlates with bucket assignment."""
    print("\nAnalyzing hand strength vs bucket assignment...")
    print("Testing various hand strengths on the same board texture...")
    print("-" * 60)

    if street == Street.FLOP:
        board_str = ["Kh", "9d", "3c"]
        test_hands = [
            (("Kc", "Ks"), "Set of Kings (NUTS)"),
            (("Ac", "Kd"), "Top pair, top kicker"),
            (("Kd", "Qh"), "Top pair, good kicker"),
            (("Kd", "Jh"), "Top pair, medium kicker"),
            (("9h", "9s"), "Middle set"),
            (("Ah", "As"), "Overpair (Aces)"),
            (("Qc", "Qd"), "Overpair (Queens)"),
            (("Jc", "Jd"), "Overpair (Jacks)"),
            (("Th", "Ts"), "Underpair (Tens)"),
            (("7h", "7s"), "Underpair (Sevens)"),
            (("Ah", "Qh"), "Ace-high (two overs)"),
            (("Jh", "Th"), "Jack-high (gutshot)"),
            (("5h", "4h"), "Low cards (nothing)"),
        ]
    elif street == Street.TURN:
        board_str = ["Kh", "9d", "3c", "2s"]
        test_hands = [
            (("Kc", "Ks"), "Set of Kings"),
            (("Ac", "Kd"), "Top pair, top kicker"),
            (("9h", "9s"), "Set of Nines"),
            (("Ah", "As"), "Overpair (Aces)"),
            (("Qc", "Qd"), "Overpair (Queens)"),
            (("Th", "Ts"), "Underpair (Tens)"),
            (("7h", "6h"), "Gutshot + backdoor flush"),
            (("5h", "4h"), "Gutshot"),
        ]
    else:
        board_str = ["Kh", "9d", "3c", "2s", "7h"]
        test_hands = [
            (("Kc", "Ks"), "Set of Kings"),
            (("Ac", "Kd"), "Top pair, top kicker"),
            (("Kd", "9h"), "Two pair (K9)"),
            (("9h", "9s"), "Set of Nines"),
            (("Ah", "As"), "Overpair (Aces)"),
            (("Qc", "Qd"), "Overpair (Queens)"),
            (("Th", "Ts"), "Underpair (Tens)"),
            (("Ah", "5h"), "Ace high"),
        ]

    results = []
    board_cards = tuple([Card.new(c) for c in board_str])

    for hole_str, description in test_hands:
        try:
            hole_cards = tuple([Card.new(c) for c in hole_str])
            bucket = abstraction.get_bucket(hole_cards, board_cards, street)
            results.append((description, bucket))
        except Exception as exc:
            print(f"✗ Error testing '{description}': {exc}")

    print(f"\nBoard: {' '.join(board_str)}\n")
    print(f"{'Hand Description':40s} Bucket")
    print("-" * 60)
    for description, bucket in results:
        print(f"{description:40s} {bucket:3d}")

    buckets = [bucket for _, bucket in results]
    if len(set(buckets)) == len(buckets):
        print("\n✓ Perfect differentiation: Each hand strength → unique bucket")
    else:
        print(
            f"\n⚠ Bucket overlap: {len(buckets)} hands mapped to {len(set(buckets))} unique buckets"
        )

    first_half_avg = sum(buckets[: len(buckets) // 2]) / (len(buckets) // 2)
    second_half_avg = sum(buckets[len(buckets) // 2 :]) / (len(buckets) - len(buckets) // 2)

    if first_half_avg > second_half_avg:
        print(
            f"✓ Strong hands ({first_half_avg:.1f}) tend to have higher buckets than weak hands "
            f"({second_half_avg:.1f})"
        )
    else:
        print(
            f"⚠ Unexpected pattern: Strong hands ({first_half_avg:.1f}) vs weak hands "
            f"({second_half_avg:.1f})"
        )
