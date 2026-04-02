"""Interactive lookup test flow for combo abstraction CLI."""

from pathlib import Path

from src.core.game.state import Street
from src.interfaces.cli.flows.combo_precompute.common import (
    BOARD_CARDS_BY_STREET,
    _parse_cards,
    _select_abstraction,
)
from src.interfaces.cli.ui import prompts
from src.interfaces.cli.ui.context import CliContext
from src.pipeline.abstraction.postflop.bucketer import DenseBucketer
from src.pipeline.abstraction.postflop.precompute import PostflopPrecomputer

BOARD_EXAMPLES = {
    Street.FLOP: "QsJhTc",
    Street.TURN: "QsJhTc9d",
    Street.RIVER: "QsJhTc9d2h",
}


def handle_combo_test_lookup(ctx: CliContext) -> None:
    """Interactively test bucket lookups for specific hands/boards."""
    print()
    print("=" * 60)
    print("  TEST COMBO ABSTRACTION LOOKUP")
    print("=" * 60)

    entry = _select_abstraction(ctx)
    if entry is None:
        return

    abstraction = _load_abstraction(entry.path)
    if abstraction is None:
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
        _lookup_once(ctx, abstraction, Street[street_choice])


def _load_abstraction(abstraction_path: Path) -> DenseBucketer | None:
    """Load the selected abstraction, or report why it could not be loaded."""
    print(f"\nLoading abstraction from {abstraction_path.name}...")
    try:
        abstraction: DenseBucketer = PostflopPrecomputer.load(abstraction_path)
    except Exception as exc:
        print(f"✗ Failed to load: {exc}")
        return None
    print("✓ Loaded successfully")
    return abstraction


def _lookup_once(ctx: CliContext, abstraction: DenseBucketer, street: Street) -> None:
    """One hand/board lookup, plus the optional isomorphic-board check.

    Errors are printed rather than raised: this is a REPL over user-typed cards,
    where a typo should cost one line, not the session.
    """
    print("\nEnter hole cards (e.g., AsKh):")
    hole_input = prompts.text(ctx, "Hole cards:", default="AsKh")
    if hole_input is None:
        return

    board_example = BOARD_EXAMPLES[street]
    print(f"\nEnter board (e.g., {board_example}):")
    board_input = prompts.text(ctx, "Board:", default=board_example)
    if board_input is None:
        return

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
        if not iso_board_input:
            return

        iso_board_cards = _parse_cards(iso_board_input, expected=BOARD_CARDS_BY_STREET[street])
        iso_bucket = abstraction.get_bucket(tuple(hole_cards), tuple(iso_board_cards), street)
        if iso_bucket == bucket:
            print(f"✓ Isomorphic board maps to same bucket: {iso_bucket}")
        else:
            print(f"⚠ Different bucket: {iso_bucket} (expected {bucket})")
    except Exception as exc:
        print(f"\n✗ Error: {exc}")
