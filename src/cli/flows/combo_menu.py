"""Menu composition for combo abstraction CLI flows."""

from src.cli.flows.combo_precompute import (
    handle_combo_analyze_bucketing,
    handle_combo_coverage,
    handle_combo_info,
    handle_combo_precompute,
    handle_combo_test_lookup,
)
from src.cli.ui.context import CliContext
from src.cli.ui.menu import MenuItem, run_menu


def combo_menu(ctx: CliContext) -> None:
    """Show combo abstraction tools submenu."""
    items = [
        MenuItem("Precompute Abstraction", handle_combo_precompute),
        MenuItem("View Abstraction Info", handle_combo_info),
        MenuItem("Test Bucket Lookup", handle_combo_test_lookup),
        MenuItem("Analyze Bucketing Patterns", handle_combo_analyze_bucketing),
        MenuItem("Analyze Coverage (Fallback Rate)", handle_combo_coverage),
    ]

    run_menu(ctx, "Combo Abstraction Tools:", items, exit_label="Back")
