"""Preflop action-size token vocabulary (e.g. ``"3x_open"``, ``"2.5x_last"``, ``"jam"``).

Single source of truth for the token format: config validation
(``src/shared/config.py``) and runtime parsing
(``src/core/actions/action_model.py``) both build on these helpers so the
accepted vocabulary cannot drift between the two.
"""

from __future__ import annotations

# Tokens that never contribute a raise size.
PASSIVE_TOKENS = frozenset({"fold", "call", "check", "limp"})
# Tokens that mean "raise all-in".
JAM_TOKENS = frozenset({"jam", "allin", "all_in"})
# Suffixes carrying a size multiplier; both scale the amount faced.
MULTIPLIER_SUFFIXES = ("x_open", "x_last")


def parse_multiplier_token(token: str) -> float | None:
    """Multiplier from a ``<number>x_open`` / ``<number>x_last`` token.

    Returns None when the token carries no multiplier suffix. Raises
    ValueError when it has the suffix but the prefix is not a number.
    """
    normalized = token.strip().lower()
    for suffix in MULTIPLIER_SUFFIXES:
        if normalized.endswith(suffix):
            return float(normalized[: -len(suffix)])
    return None
