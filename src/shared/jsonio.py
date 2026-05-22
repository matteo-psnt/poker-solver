"""JSON serialization helpers shared across layers."""

from __future__ import annotations

import json
from typing import Any


def json_default(obj: Any) -> Any:
    """Coerce non-JSON-native values (e.g. numpy scalars) to plain types."""
    try:
        return float(obj)
    except (TypeError, ValueError):
        return str(obj)


def dumps(payload: Any, *, indent: int | None = None, allow_nan: bool = True) -> str:
    """Serialise a command payload the same way for every surface.

    Both doors out of the command layer serialise: ``headless --json`` prints
    the payload, and the console returns it over HTTP. They used to do it with
    different encoders -- the CLI passing ``default=json_default``, the console
    handing the dict to Starlette's ``JSONResponse``, which has no such hook.
    A payload carrying anything JSON has no native form for (a numpy scalar, a
    ``Path``) therefore printed fine and crashed the console with a 500, past
    the refusal/unavailable ladder that exists so a panel can fail politely.

    Nothing currently reaches that: every console-reachable payload is derived
    from JSONL on the share, and the numpy-adjacent values are coerced at their
    source (``StaticArrayStorage.coverage`` and friends return built-ins). But
    that is provenance, not a guarantee -- the payload fixture cannot catch it,
    being hand-written JSON-native literals -- so the encoder is shared instead
    of the discipline being restated.

    ``allow_nan`` is the one thing the two surfaces are still allowed to differ
    on, and deliberately: ``NaN`` is not valid JSON, so the browser's
    ``JSON.parse`` rejects it and the console asks for a loud failure, while the
    command line has always printed it and a human reading a terminal can see
    what it means.
    """
    return json.dumps(payload, indent=indent, default=json_default, allow_nan=allow_nan)
