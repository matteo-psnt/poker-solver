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
from typing import Any

import httpx
from fastapi import FastAPI
from fastapi.responses import JSONResponse

BLUEPRINT_URL_ENV = "POKER_SOLVER_BLUEPRINT_URL"
BLUEPRINT_TOKEN_ENV = "POKER_SOLVER_BLUEPRINT_TOKEN"

# Generous: a node read crosses a tunnel, and the grid is a real computation on
# the far side. Short enough that a dead server is reported rather than hung on.
TIMEOUT_SECONDS = 30.0


def blueprint_url() -> str | None:
    """The configured server, or ``None`` when the feature is not turned on."""
    url = os.environ.get(BLUEPRINT_URL_ENV, "").strip()
    return url.rstrip("/") or None


def _headers() -> dict[str, str]:
    """The bearer token, when there is one.

    Absent on a laptop pointing at a local server, which is why this is optional
    rather than required: the token belongs to the hosted box, where Caddy checks
    it, and demanding one everywhere would make the plain case unrunnable.
    """
    token = os.environ.get(BLUEPRINT_TOKEN_ENV, "").strip()
    return {"authorization": f"Bearer {token}"} if token else {}


def forward(
    path: str,
    params: dict[str, str | bool] | None = None,
    *,
    method: str = "GET",
    json: Any = None,
) -> JSONResponse:
    """One request against the blueprint server, with its refusals preserved.

    A 422 from the far side is a refusal the analysis layer already phrased for a
    person -- passing it through unchanged is the whole point, since rewriting it
    here would mean maintaining a second vocabulary for the same failures. A 404
    likewise: only the far side knows whether a session still exists.
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
        response = httpx.request(
            method,
            f"{base}{path}",
            params=params,
            json=json,
            headers=_headers(),
            timeout=TIMEOUT_SECONDS,
        )
    except httpx.HTTPError as error:
        return JSONResponse(
            {"error": f"The blueprint server at {base} did not answer: {error}"},
            status_code=503,
        )
    if response.status_code == 404 and not response.headers.get("content-type", "").startswith(
        "application/json"
    ):
        # Caddy answers an unauthorized request with a bare 404 rather than a
        # 401, so a scanner learns nothing. That makes a wrong token look exactly
        # like a wrong address from here -- say both, since the fix differs.
        return JSONResponse(
            {
                "error": f"{base} did not recognise this request. Check "
                f"{BLUEPRINT_TOKEN_ENV} matches the box's token."
            },
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

    # The one WRITE here, and the reason the console can change which run it is
    # charting at all. It returns immediately with a 202 and the far side loads
    # on its own thread, so nothing here needs a longer timeout than any other
    # call: the client watches `/api/blueprint/run` for the swap to land.
    @app.post("/api/blueprint/load")
    def _load(body: dict[str, Any]) -> JSONResponse:
        return forward("/api/load", method="POST", json=body)

    @app.get("/api/blueprint/node")
    def _node(path: str = "", board: str = "", average: bool = True) -> JSONResponse:
        return forward("/api/node", {"path": path, "board": board, "average": average})

    # Play is stateful, so these carry a body and a session id. The proxy still
    # holds no state of its own: the session lives where the blueprint does, and
    # a console restart therefore does not lose a hand in progress.
    @app.post("/api/blueprint/play")
    def _start(body: dict[str, Any]) -> JSONResponse:
        return forward("/api/play", method="POST", json=body)

    @app.get("/api/blueprint/play/{session_id}")
    def _hand(session_id: str) -> JSONResponse:
        return forward(f"/api/play/{session_id}")

    @app.post("/api/blueprint/play/{session_id}/action")
    def _act(session_id: str, body: dict[str, Any]) -> JSONResponse:
        return forward(f"/api/play/{session_id}/action", method="POST", json=body)

    @app.delete("/api/blueprint/play/{session_id}")
    def _leave(session_id: str) -> JSONResponse:
        return forward(f"/api/play/{session_id}", method="DELETE")
