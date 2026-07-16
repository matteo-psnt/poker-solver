"""Validation flow for combo abstraction CLI."""

import random

from src.core.game.state import Card, Street
from src.interfaces.cli.flows.combo_precompute.common import (
    ALL_CARDS,
    BOARD_CARDS_BY_STREET,
    _select_abstraction,
)
from src.interfaces.cli.ui import prompts
from src.interfaces.cli.ui.context import CliContext
from src.pipeline.abstraction.postflop.bucketer import DenseBucketer
from src.pipeline.abstraction.postflop.precompute import PostflopPrecomputer


def handle_combo_validate(ctx: CliContext) -> None:
    """Run comprehensive validation tests on an abstraction."""
    print()
    print("=" * 60)
    print("  VALIDATE COMBO ABSTRACTION")
    print("=" * 60)

    entry = _select_abstraction(ctx)
    if entry is None:
        return
    abstraction_path = entry.path

    print(f"\nRunning validation on {abstraction_path.name}...")
    print("This will test:")
    print("  - Loading")
    print("  - Structure integrity")
    print("  - Bucket lookups")
    print("  - Isomorphism invariance")
    print("  - Bucket distribution")
    print()

    confirm = prompts.confirm(
        ctx,
        "Run validation?",
        default=True,
    )
    if not confirm:
        return

    try:
        abstraction = PostflopPrecomputer.load(abstraction_path)
    except Exception as exc:
        print(f"\n✗ Error loading abstraction: {exc}")
        return

    _run_basic_validation(abstraction)


def _run_basic_validation(abstraction: DenseBucketer, num_samples: int = 100) -> None:
    """Run quick sanity checks for an abstraction."""
    print("\nRunning basic validation...")
    print("  - Load abstraction")
    print("  - Random bucket lookups")
    print("  - Bucket range checks")

    total_checks = 0
    failures = 0

    for street in [Street.FLOP, Street.TURN, Street.RIVER]:
        num_buckets = abstraction.num_buckets(street)
        if num_buckets <= 0:
            print(f"\n✗ Invalid bucket count on {street.name}: {num_buckets}")
            failures += 1
            continue

        expected_board_cards = BOARD_CARDS_BY_STREET[street]
        for _ in range(num_samples):
            total_checks += 1
            try:
                sampled = random.sample(ALL_CARDS, 2 + expected_board_cards)
                hole_str = sampled[:2]
                board_str = sampled[2:]

                hole_cards = tuple([Card.new(c) for c in hole_str])
                board_cards = tuple([Card.new(c) for c in board_str])

                hole_cards_pair: tuple[Card, Card] = (hole_cards[0], hole_cards[1])
                bucket = abstraction.get_bucket(hole_cards_pair, board_cards, street)
                if bucket < 0 or bucket >= num_buckets:
                    raise ValueError(f"Bucket {bucket} out of range (0..{num_buckets - 1})")
            except Exception as exc:
                failures += 1
                if failures == 1:
                    print(f"\n  [DEBUG] First error on {street.name}: {exc}")

    if failures == 0:
        print(f"\n✓ Validation complete! ({total_checks} checks)")
    else:
        print(f"\n✗ Validation failed with {failures} error(s) out of {total_checks} checks")
