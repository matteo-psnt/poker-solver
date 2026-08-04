"""Turning a leg's facts into the few words that identify it.

Two callers, on opposite sides of a layering boundary, must agree exactly:
``src.interfaces.cloud.spec`` builds the task id a leg is SUBMITTED under, and
``src.shared.leg_log`` describes the same leg when the record is READ back. If
those disagreed, one leg would be called two things and the id would stop being
a way to find it.

Stdlib only, and 3.10-compatible: ``leg_log`` imports this and runs on the
node's system python before ``uv sync``. See
``tests/shared/node/test_node_interpreter.py``, which enforces both.
"""

from __future__ import annotations

from collections.abc import Sequence


def compact_count(value: int) -> str:
    """``150000000`` -> ``150M``. An iteration target is the point of a leg."""
    for scale, suffix in ((1_000_000_000, "B"), (1_000_000, "M"), (1_000, "k")):
        if value >= scale:
            return f"{value / scale:g}{suffix}"
    return str(value)


def flag_value(flags: Sequence[str], name: str) -> str:
    """The value of ``--name v`` or ``--name=v`` in a passthrough flag list."""
    for index, item in enumerate(flags):
        if item == name:
            return flags[index + 1] if index + 1 < len(flags) else ""
        if item.startswith(f"{name}="):
            return item.split("=", 1)[1]
    return ""
