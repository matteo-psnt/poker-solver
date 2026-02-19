"""Coverage analysis flow for combo abstraction CLI."""

import random

from questionary import Choice

from src.bucketing.postflop.precompute import PostflopPrecomputer
from src.cli.flows.combo_precompute.common import BOARD_CARDS_BY_STREET, _select_abstraction
from src.cli.ui import prompts
from src.cli.ui.context import CliContext
from src.game.state import Card, Street


def handle_combo_coverage(ctx: CliContext) -> None:
    """Analyze abstraction coverage and fallback rate."""
    print()
    print("=" * 60)
    print("  ANALYZE ABSTRACTION COVERAGE")
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

    street_choices = prompts.checkbox(
        ctx,
        "Select streets to analyze:",
        choices=[
            Choice("FLOP", checked=True),
            Choice("TURN", checked=True),
            Choice("RIVER", checked=True),
        ],
    )
    if not street_choices:
        return

    streets = [Street[s] for s in street_choices]

    num_samples = prompts.prompt_int(
        ctx,
        "Number of random samples per street:",
        default=1000,
        min_value=1,
    )
    if num_samples is None:
        return

    print("\n" + "=" * 60)
    print("COVERAGE ANALYSIS")
    print("=" * 60)

    ranks = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
    suits = ["h", "d", "c", "s"]
    all_cards = [r + s for r in ranks for s in suits]

    overall_stats = {}

    for street in streets:
        print(f"\nAnalyzing {street.name}...")
        print("-" * 60)

        num_board_cards = BOARD_CARDS_BY_STREET[street]
        abstraction.reset_stats()

        successful_lookups = 0

        for _ in range(num_samples):
            try:
                sampled = random.sample(all_cards, 2 + num_board_cards)
                hole_str = sampled[:2]
                board_str = sampled[2:]

                hole_cards = tuple([Card.new(c) for c in hole_str])
                board_cards = tuple([Card.new(c) for c in board_str])

                hole_cards_pair: tuple[Card, Card] = (hole_cards[0], hole_cards[1])
                _bucket = abstraction.get_bucket(hole_cards_pair, board_cards, street)
                successful_lookups += 1
            except Exception as exc:
                if successful_lookups == 0:
                    print(f"  [DEBUG] First error: {exc}")

        stats = abstraction.get_fallback_stats()
        total_lookups = stats["total_lookups"]
        fallback_count = stats["fallback_count"]
        fallback_rate = stats["fallback_rate"] * 100

        print(f"  Total lookups: {total_lookups:,}")
        print(f"  Direct hits:   {total_lookups - fallback_count:,} ({100 - fallback_rate:.1f}%)")
        print(f"  Fallback used: {fallback_count:,} ({fallback_rate:.1f}%)")

        overall_stats[street] = {
            "total_lookups": total_lookups,
            "fallback_count": fallback_count,
            "fallback_rate": fallback_rate,
        }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    total_all = sum(s["total_lookups"] for s in overall_stats.values())
    fallback_all = sum(s["fallback_count"] for s in overall_stats.values())
    fallback_rate_all = (fallback_all / total_all * 100) if total_all > 0 else 0

    print("\nOverall across all analyzed streets:")
    print(f"  Total lookups:  {total_all:,}")
    print(f"  Direct hits:    {total_all - fallback_all:,} ({100 - fallback_rate_all:.1f}%)")
    print(f"  Fallback used:  {fallback_all:,} ({fallback_rate_all:.1f}%)")

    print("\nInterpretation:")
    if fallback_rate_all < 5:
        print("  ✓ Excellent coverage! Fallback rarely needed.")
    elif fallback_rate_all < 15:
        print("  ✓ Good coverage. Some fallbacks occurring.")
        print("    Consider increasing representatives_per_cluster to 2-3 for better coverage.")
    elif fallback_rate_all < 30:
        print("  ⚠ Moderate coverage. Fallbacks fairly common.")
        print("    Recommend increasing representatives_per_cluster to 3-5.")
    else:
        print("  ✗ Poor coverage. Fallback used frequently.")
        print("    Strongly recommend increasing representatives_per_cluster to 5-10.")

    print("\n  Note: Fallback uses cluster median bucket, which is reasonable but less precise.")
    print("  Higher representatives_per_cluster = more hands precomputed = fewer fallbacks.")
    print()
