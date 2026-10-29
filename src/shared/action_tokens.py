"""Preflop action-size token vocabulary (e.g. ``"3x_to"``, ``"2.5x_last"``, ``"jam"``).

Single source of truth for the token format: config validation
(``src/shared/config/schema.py``) and runtime parsing
(``src/core/actions/action_model.py``) both build on these helpers so the
accepted vocabulary cannot drift between the two.
"""

from __future__ import annotations

# Tokens that never contribute a raise size.
PASSIVE_TOKENS = frozenset({"fold", "call", "check", "limp"})
# Tokens that mean "raise all-in".
JAM_TOKENS = frozenset({"jam", "allin", "all_in"})
# Suffixes carrying a size multiplier. `x_open`/`x_last` scale the CALL owed, so
# 3.5x_open over a 2 bb open is a 3-bet to 4.5 bb; every fielded tree is built on
# that. `x_to` scales the wager faced: 3.5x_to over a 2 bb open is to 7 bb.
TO_SUFFIX = "x_to"
MULTIPLIER_SUFFIXES = ("x_open", "x_last", TO_SUFFIX)


def parse_multiplier_token(token: str) -> float | None:
    """Multiplier from a ``<number>x_open`` / ``x_last`` / ``x_to`` token.

    Returns None when the token carries no multiplier suffix. Raises
    ValueError when it has the suffix but the prefix is not a number.
    """
    normalized = token.strip().lower()
    for suffix in MULTIPLIER_SUFFIXES:
        if normalized.endswith(suffix):
            return float(normalized[: -len(suffix)])
    return None
