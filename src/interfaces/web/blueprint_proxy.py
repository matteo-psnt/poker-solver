"""Forwarding `/api/blueprint/*` to whichever process is holding a run.

This module is the reason the console can show solver strategy without breaking
its own rule. `web_reads_through_the_command_layer` forbids
:mod:`src.interfaces.web` importing pipeline, engine or core -- and nothing here
does. The engine is on the far end of a socket, so the console stays a client of
something else's answers, which is the property that stopped the previous browser
UI from growing a second read path.

Where the blueprint server is
-----------------------------
``POKER_SOLVER_BLUEPRINT_URL``. Unset means the feature is off, and every
endpoint says so in a sentence rather than failing as a connection error -- "not
configured" and "configured but unreachable" are different problems with
different fixes, and a bare `ConnectionRefused` conflates them.
"""

from __future__ import annotations

import os

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

BLUEPRINT_URL_ENV = "POKER_SOLVER_BLUEPRINT_URL"

# Generous: a node read crosses a tunnel, and the grid is a real computation on
# the far side. Short enough that a dead server is reported rather than hung on.
TIMEOUT_SECONDS = 30.0


def blueprint_url() -> str | None:
    """The configured server, or ``None`` when the feature is not turned on."""
    url = os.environ.get(BLUEPRINT_URL_ENV, "").strip()
    return url.rstrip("/") or None


def forward(path: str, params: dict[str, str | bool]) -> JSONResponse:
    """One GET against the blueprint server, with its refusals preserved.

    A 422 from the far side is a refusal the analysis layer already phrased for a
    person -- passing it through unchanged is the whole point, since rewriting it
    here would mean maintaining a second vocabulary for the same failures.
    """
    base = blueprint_url()
    if base is None:
        return JSONResponse(
            {
                "error": "No blueprint server is configured. Point "
                f"{BLUEPRINT_URL_ENV} at one to browse a run's strategy."
            },
            status_code=503,
        )
    try:
        response = httpx.get(f"{base}{path}", params=params, timeout=TIMEOUT_SECONDS)
    except httpx.HTTPError as error:
        return JSONResponse(
            {"error": f"The blueprint server at {base} did not answer: {error}"},
            status_code=503,
        )
    if response.status_code >= 500:
        return JSONResponse(
            {"error": f"The blueprint server failed: {response.text[:200]}"},
            status_code=503,
        )
    return JSONResponse(response.json(), status_code=response.status_code)


def mount(app: FastAPI) -> None:
    """Add the proxied endpoints.

    ``def``, not ``async def``, for the same reason as every other endpoint in
    this package: the call below is synchronous, and a coroutine would hold the
    event loop for the whole round trip and serialise every other panel behind
    it.
    """

    @app.get("/api/blueprint/run")
    def _run() -> JSONResponse:
        return forward("/api/run", {})

    @app.get("/api/blueprint/combos")
    def _combos() -> JSONResponse:
        return forward("/api/combos", {})

    @app.get("/api/blueprint/node")
    def _node(path: str = "", board: str = "", average: bool = True) -> JSONResponse:
        return forward("/api/node", {"path": path, "board": board, "average": average})
