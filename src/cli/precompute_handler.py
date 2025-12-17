"""Precomputation handler."""

import questionary

from src.abstraction.precompute import (
    PrecomputeConfig,
    precompute_equity_bucketing,
    print_summary,
)


def handle_precompute(style):
    """Handle precomputation."""
    config_choice = questionary.select(
        "Configuration:",
        choices=["Fast (1-2 min)", "Production (10-15 min)", "Cancel"],
        style=style,
    ).ask()

    if not config_choice or "Cancel" in config_choice:
        return

    config = PrecomputeConfig.fast_test() if "Fast" in config_choice else PrecomputeConfig.default()

    import multiprocessing as mp

    config.num_workers = mp.cpu_count()

    print_summary(config)

    if questionary.confirm("Proceed?", default=True, style=style).ask():
        try:
            precompute_equity_bucketing(config)
            print("\n✅ Done!")
        except Exception as e:
            print(f"\n❌ Error: {e}")

    input("\nPress Enter...")
