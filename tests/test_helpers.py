"""Test helpers for poker solver tests."""

from typing import Any

from src.pipeline.abstraction.base import BucketingStrategy
from src.shared.config import Config


def make_test_config(**overrides) -> Config:
    """
    Create a Config object for tests with optional overrides.

    Examples:
        make_test_config(seed=42)
        make_test_config(seed=42, sampling_method="outcome")
        make_test_config(starting_stack=100)
    """
    # Map common shorthand overrides to nested dict structure
    shorthand_map = {
        "seed": ("system", "seed"),
        "starting_stack": ("game", "starting_stack"),
        "small_blind": ("game", "small_blind"),
        "big_blind": ("game", "big_blind"),
        "sampling_method": ("solver", "sampling_method"),
        "cfr_plus": ("solver", "cfr_plus"),
        "iteration_weighting": ("solver", "iteration_weighting"),
        # DCFR parameters
        "dcfr_alpha": ("solver", "dcfr_alpha"),
        "dcfr_beta": ("solver", "dcfr_beta"),
        "dcfr_gamma": ("solver", "dcfr_gamma"),
        # Pruning parameters
        "enable_pruning": ("solver", "enable_pruning"),
        "pruning_threshold": ("solver", "pruning_threshold"),
        "prune_start_iteration": ("solver", "prune_start_iteration"),
        "prune_reactivate_frequency": ("solver", "prune_reactivate_frequency"),
    }

    # Build nested dict from overrides
    nested: dict[str, dict[str, Any]] = {}
    for key, value in overrides.items():
        if key in shorthand_map:
            section, field = shorthand_map[key]
            if section not in nested:
                nested[section] = {}
            nested[section][field] = value
        else:
            # Assume it's already a section.field format or top-level
            parts = key.split(".")
            if len(parts) == 2:
                section, field = parts
                if section not in nested:
                    nested[section] = {}
                nested[section][field] = value
            else:
                nested[key] = value

    return Config.default().merge(nested) if nested else Config.default()


class DummyCardAbstraction(BucketingStrategy):
    """
    Minimal card abstraction for testing.

    All hands map to bucket 0 (single bucket per street).
    Used when card abstraction logic isn't being tested.
    """

    def get_bucket(self, hole_cards, board, street):
        """All hands map to bucket 0."""
        return 0

    def num_buckets(self, street):
        """Single bucket per street."""
        return 1

    def get_fallback_stats(self) -> dict:
        return {"total_lookups": 0, "fallback_count": 0, "fallback_rate": 0.0}
