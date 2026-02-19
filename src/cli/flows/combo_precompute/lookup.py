"""Interactive lookup test flow for combo abstraction CLI."""

from src.bucketing.postflop.hand_bucketing import PostflopBucketer
from src.bucketing.postflop.precompute import PostflopPrecomputer
from src.cli.flows.combo_precompute.common import (
    BOARD_CARDS_BY_STREET,
    _parse_cards,
    _select_abstraction,
)
from src.cli.ui import prompts
from src.cli.ui.context import CliContext
from src.game.state import Street


def handle_combo_test_lookup(ctx: CliContext) -> None:
    """Interactively test bucket lookups for specific hands/boards."""
    print()
    print("=" * 60)
    print("  TEST COMBO ABSTRACTION LOOKUP")
    print("=" * 60)

    abstraction_path, _metadata = _select_abstraction(ctx)
    if abstraction_path is None:
        return

    print(f"\nLoading abstraction from {abstraction_path.name}...")
    try:
        abstraction: PostflopBucketer = PostflopPrecomputer.load(abstraction_path)
        print("✓ Loaded successfully")
    except Exception as exc:
        print(f"✗ Failed to load: {exc}")
        return

    while True:
        print("\n" + "-" * 60)

        street_choice = prompts.select(
            ctx,
            "Select street:",
            choices=["FLOP", "TURN", "RIVER", "Back"],
        )
        if street_choice is None or street_choice == "Back":
            break

        street = Street[street_choice]

        print("\nEnter hole cards (e.g., AsKh):")
        hole_input = prompts.text(
            ctx,
            "Hole cards:",
            default="AsKh",
        )
        if hole_input is None:
            continue

        if street == Street.FLOP:
            board_example = "QsJhTc"
        elif street == Street.TURN:
            board_example = "QsJhTc9d"
        else:
            board_example = "QsJhTc9d2h"

        print(f"\nEnter board (e.g., {board_example}):")
        board_input = prompts.text(
            ctx,
            "Board:",
            default=board_example,
        )
        if board_input is None:
            continue

        try:
            hole_cards = _parse_cards(hole_input, expected=2)
            board_cards = _parse_cards(board_input, expected=BOARD_CARDS_BY_STREET[street])

            bucket = abstraction.get_bucket(tuple(hole_cards), tuple(board_cards), street)

            print(f"\n✓ Bucket: {bucket}")
            print(f"  (out of {abstraction.num_buckets(street)} buckets on {street.name})")

            iso_board_input = prompts.text(
                ctx,
                "\nOptional: Enter isomorphic board to verify same bucket:",
                default="",
            )

            if iso_board_input:
                iso_board_cards = _parse_cards(
                    iso_board_input, expected=BOARD_CARDS_BY_STREET[street]
                )
                iso_bucket = abstraction.get_bucket(
                    tuple(hole_cards), tuple(iso_board_cards), street
                )

                if iso_bucket == bucket:
                    print(f"✓ Isomorphic board maps to same bucket: {iso_bucket}")
                else:
                    print(f"⚠ Different bucket: {iso_bucket} (expected {bucket})")

        except Exception as exc:
            print(f"\n✗ Error: {exc}")
