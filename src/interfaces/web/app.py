"""The console's HTTP layer: one endpoint per command, and nothing else.

**No new read path.** Every endpoint body is a single ``Command.invoke`` and a
memo. That is the whole design, and it is the property the previous browser UI
lacked: `fbcf9a8` carried `api/chart_service.py`, `api/play_service.py` and
`chart/data.py` -- a second way to ask questions the CLI already answered, which
drifted from it and then rotted. `tests/interfaces/web/` fails if anything here
grows one.

**Endpoints are ``def``, not ``async def``.** Every Azure client in
:mod:`src.interfaces.cloud` is synchronous, so a coroutine here would block the
event loop for the whole 2-4s of a cloud read and serialise every other request
behind it. FastAPI runs sync handlers in a threadpool, which is exactly right --
and is the single detail about this file most likely to be "tidied" into a bug.

One endpoint per command rather than one aggregate: a page fetches only what it
shows, `/legs` does not pay for `ledger`, and each panel fails alone. The client
owns cadence (TanStack Query's ``refetchInterval``), so there is no poller here
and nothing depends on a background thread staying alive.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from azure.core.exceptions import ClientAuthenticationError, HttpResponseError
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.interfaces.cli.commands import (
    Command,
    curve,
    jobs,
    ledger,
    legs,
    logs,
    pool_status,
    progress,
    runinfo,
    runs,
)
from src.interfaces.errors import CommandError
from src.interfaces.web.cache import TtlCache

# Long enough that several open tabs (or a remount) share one cloud sweep,
# short enough that a manual refresh feels live. A sweep is 2-4s per panel.
CACHE_TTL_SECONDS = 15.0

# Anchored to the repo, not to the working directory: `serve` is run from
# wherever the operator happens to be, and a CWD-relative path would report a
# perfectly good build as missing -- indistinguishable from not having built it.
CONSOLE_DIST = Path(__file__).resolve().parents[3] / "console" / "dist"


def answer(cache: TtlCache, command: Command, **kwargs: Any) -> JSONResponse:
    """Run one command, memoised, and map its failures onto status codes.

    The cache is passed in rather than reached for: :func:`create_app` owns one
    per application, so two apps in one process (a test and its subject, most
    of all) cannot serve each other's answers.

    A refusal is data, not a crash: the client renders one panel as unavailable
    and keeps the rest. That is the same contract `status` relies on, so the
    translation lives here rather than in every endpoint.

    Failures are deliberately not cached. A repeated 503 costs a repeated cloud
    read, which is the right trade: the alternative keeps serving "Azure is
    down" for the whole TTL after `az login` has already fixed it.
    """
    key = (command.name, tuple(sorted(kwargs.items())))
    try:
        return JSONResponse(cache.get(key, lambda: command.invoke(**kwargs)))
    except CommandError as error:
        # 422: the request was understood and the answer is "no" -- an
        # unpublished run, a run with no checkpoint history. Not a server fault.
        return JSONResponse({"error": str(error)}, status_code=422)
    except ClientAuthenticationError:
        return JSONResponse(
            {"error": "Azure rejected the credential — try `az login`."}, status_code=503
        )
    except HttpResponseError as error:
        return JSONResponse({"error": f"Azure did not answer: {error}"}, status_code=503)


def create_app() -> FastAPI:
    """Build the application, with a memo of its own.

    A function rather than a module-level app, so a test gets a genuinely fresh
    one: the cache is built here and closed over, so nothing survives between
    two applications in the same process.
    """
    app = FastAPI(title="poker-solver console", docs_url="/api/docs")
    cache = TtlCache(CACHE_TTL_SECONDS)

    @app.get("/api/pool")
    def _pool() -> JSONResponse:
        return answer(cache, pool_status.COMMAND)

    @app.get("/api/jobs")
    def _jobs(limit: int = 20, all: bool = False) -> JSONResponse:  # noqa: A002
        return answer(cache, jobs.COMMAND, limit=limit, all=all)

    @app.get("/api/legs")
    def _legs(limit: int = 0, skip_reconcile: bool = False) -> JSONResponse:
        return answer(
            cache, legs.COMMAND, limit=limit, skip_reconcile=skip_reconcile, legs_dir=None
        )

    @app.get("/api/runs")
    def _runs(limit: int = 0) -> JSONResponse:
        return answer(cache, runs.COMMAND, limit=limit, loadable_only=False)

    @app.get("/api/runs/{run_id}")
    def _run(run_id: str) -> JSONResponse:
        return answer(cache, runinfo.COMMAND, run=run_id)

    @app.get("/api/runs/{run_id}/progress")
    def _progress(run_id: str, last: int = 0) -> JSONResponse:
        return answer(cache, progress.COMMAND, run=run_id, last=last)

    @app.get("/api/runs/{run_id}/curve")
    def _curve(run_id: str) -> JSONResponse:
        return answer(cache, curve.COMMAND, run=run_id)

    @app.get("/api/evals")
    def _evals(limit: int = 50) -> JSONResponse:
        return answer(
            cache, ledger.COMMAND, limit=limit, run=None, experiment=None, method=None, since=None
        )

    @app.get("/api/logs/{task_id}")
    def _log(task_id: str, lines: int = 200) -> JSONResponse:
        return answer(cache, logs.COMMAND, task=task_id, lines=lines)

    _mount_console(app)
    return app


def _mount_console(app: FastAPI) -> None:
    """Serve the built console, with SPA fallback.

    Client-side routing means `/legs` and `/runs/abc` are not files: the server
    must return `index.html` and let the router decide. Anything under `/api`
    has already matched above, so the fallback cannot swallow a real endpoint.

    A missing build is reported rather than served as a blank page -- the
    failure otherwise looks like a broken app instead of a skipped build step.
    """
    index = CONSOLE_DIST / "index.html"
    if not index.is_file():

        @app.get("/{_path:path}")
        def _unbuilt(_path: str) -> JSONResponse:
            return JSONResponse(
                {
                    "error": "The console is not built. Run `just console-build` "
                    f"(expected {index})."
                },
                status_code=503,
            )

        return

    # Conditional because `StaticFiles` raises at mount time on a missing
    # directory, which would take the whole server down at startup rather than
    # degrade one route -- and a build small enough to inline every asset emits
    # no `assets/` at all.
    assets = CONSOLE_DIST / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")

    @app.get("/{_path:path}")
    def _spa(_path: str) -> FileResponse:
        return FileResponse(index)
