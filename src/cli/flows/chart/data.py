"""Compatibility wrapper for chart data builders.

Chart data generation lives in :mod:`src.chart.data` so API and CLI can share
it without cross-layer imports. Keep this module as a stable import path for
legacy callers and tests.
"""

from src.chart.data import (
    _ranks_to_hand_string,
    build_chart_metadata,
    build_preflop_chart_data,
)

__all__ = [
    "build_chart_metadata",
    "build_preflop_chart_data",
    "_ranks_to_hand_string",
]
