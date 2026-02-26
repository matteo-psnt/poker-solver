"""Pruning section editor for CLI config."""

from src.interfaces.cli.ui.context import CliContext
from src.shared.config import Config


def edit_pruning(ctx: CliContext, config: Config) -> Config:
    print("Pruning")
    print("-" * 40)
    print(
        "Regret-based pruning is not implemented on the shared-array storage backend: "
        "the pruned mask lives on the per-visit InfoSet view and never persists, so it "
        "is a no-op. enable_pruning is rejected at config load and cannot be turned on "
        "here."
    )
    return config
