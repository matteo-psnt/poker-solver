"""Compatibility facade for combo abstraction CLI flows."""

from src.cli.flows.combo_analysis import (
    _analyze_hand_strength_correlation,
    _analyze_premium_vs_weak,
    _analyze_random_sample,
    handle_combo_analyze_bucketing,
    handle_combo_coverage,
)
from src.cli.flows.combo_inspect import (
    _parse_cards,
    _run_basic_validation,
    _select_abstraction,
    _show_detailed_info,
    handle_combo_info,
    handle_combo_test_lookup,
    handle_combo_validate,
)
from src.cli.flows.combo_menu import combo_abstraction_menu, combo_menu
from src.cli.flows.combo_precompute import (
    _estimate_time,
    _get_config_choice,
    _get_config_name_from_metadata,
    _get_output_path,
    _list_available_configs,
    handle_combo_precompute,
)

__all__ = [
    "combo_menu",
    "combo_abstraction_menu",
    "handle_combo_precompute",
    "handle_combo_info",
    "handle_combo_test_lookup",
    "handle_combo_validate",
    "handle_combo_coverage",
    "handle_combo_analyze_bucketing",
    "_get_config_name_from_metadata",
    "_list_available_configs",
    "_get_config_choice",
    "_estimate_time",
    "_get_output_path",
    "_show_detailed_info",
    "_select_abstraction",
    "_parse_cards",
    "_run_basic_validation",
    "_analyze_premium_vs_weak",
    "_analyze_random_sample",
    "_analyze_hand_strength_correlation",
]
