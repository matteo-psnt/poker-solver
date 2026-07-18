"""JSON serialization helpers shared across layers."""

from __future__ import annotations

from typing import Any


def json_default(obj: Any) -> Any:
    """Coerce non-JSON-native values (e.g. numpy scalars) to plain types."""
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)
