"""JSON serialization helpers shared across layers."""

from __future__ import annotations

import json
from typing import Any


def json_default(obj: Any) -> Any:
    """Coerce non-JSON-native values (e.g. numpy scalars) to plain types.

    A payload MODEL is handled first and by duck typing rather than by importing
    pydantic. Both are deliberate. First, because the float coercion below
    swallows anything -- a model would fall through to ``str(obj)`` and reach the
    browser as a repr, which parses as a string and fails nowhere. And duck
    typed, because this module is inside the node's fail-closed import closure:
    it runs before ``uv sync``, where pydantic does not exist yet.
    ``model_dump`` recurses, so one hop is enough
    however deeply models are nested.
    """
    dump = getattr(obj, "model_dump", None)
    if callable(dump):
        return dump()
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def dumps(payload: Any, *, indent: int | None = None, allow_nan: bool = True) -> str:
    """Serialise a command payload the same way for every surface.

    Both doors out of the command layer serialise -- ``headless --json`` prints the
    payload, the console returns it over HTTP -- and with different encoders a value
    JSON has no native form for (a numpy scalar, a ``Path``) printed fine on one and
    crashed the other with a 500, past the ladder that lets a panel fail politely.
    The encoder is shared rather than the discipline restated.

    ``allow_nan`` is the one thing the two are allowed to differ on: ``NaN`` is not
    valid JSON, so the browser's ``JSON.parse`` rejects it and the console wants a
    loud failure, while a human reading a terminal can see what it means.
    """
    return json.dumps(payload, indent=indent, default=json_default, allow_nan=allow_nan)
