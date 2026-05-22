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
shows, `/tasks` does not pay for `ledger`, and each panel fails alone. The client
owns cadence (TanStack Query's ``refetchInterval``), so there is no poller here
and nothing depends on a background thread staying alive.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from src.interfaces.cloud.store import workspace
from src.interfaces.commands import (
    Command,
    cancel,
    cost,
    curve,
    jobs,
    ledger,
    logs,
    pool_status,
    progress,
    runinfo,
    runs,
    serve_box,
    tasks,
)
from src.interfaces.errors import attempt
from src.interfaces.web import blueprint_proxy
from src.interfaces.web.cache import TtlCache
from src.shared import jsonio, repo

# Long enough that several open tabs (or a remount) share one cloud sweep,
# short enough that a manual refresh feels live. A sweep is 2-4s per panel.
CACHE_TTL_SECONDS = 15.0

"""Why the server materialises the record once, and this file knows about it
--------------------------------------------------------------------------
The memo above is per (command, arguments), so five endpoints asking five
different questions about the SAME record each paid for their own copy of it:
`/api/runs` and `/api/evals` pulled the whole thing (12.4s each) and a run's
three detail panels pulled that run three times over. The work is ~120 network
round trips for 0.23 MB -- latency, not bytes, and duplicated per endpoint.

Held for longer than `CACHE_TTL_SECONDS` because it is a different thing being
cached: the payloads are what a panel shows and should look live, the tree is
the substrate they are all derived from. The honest bound is that a panel's age
badge -- client fetch time -- can understate the data's age by up to this.
"""
RECORD_TREE_TTL_SECONDS = 45.0


# Anchored to the repo, not to the working directory: `serve` is run from
# wherever the operator happens to be, and a CWD-relative path would report a
# perfectly good build as missing -- indistinguishable from not having built it.
CONSOLE_DIST = repo.ROOT / "console" / "dist"


class PayloadResponse(JSONResponse):
    """A command payload, encoded exactly as ``--json`` would encode it.

    Starlette's ``JSONResponse.render`` calls ``json.dumps`` with no ``default``
    hook, so a payload carrying a numpy scalar or a ``Path`` -- which the CLI
    prints without complaint -- raised a ``TypeError`` from inside the response,
    past :func:`answer`'s ladder, and reached the browser as a 500. Sharing the
    encoder with :mod:`headless` is what makes "the payload is the interface"
    true of the bytes and not just the dict.

    ``allow_nan=False`` is kept from Starlette's default: ``NaN`` is not valid
    JSON and ``JSON.parse`` rejects it, so a payload containing one should fail
    here rather than reach a panel as a parse error with no origin.
    """

    def render(self, content: Any) -> bytes:
        return jsonio.dumps(content, allow_nan=False).encode()


# A refusal is understood-and-the-answer-is-no (an unpublished run, a run with
# no checkpoint history), so it is the client's business, not a server fault;
# unavailable means Azure did not answer, which is transient and worth retrying.
_STATUS = {"refusal": 422, "unavailable": 503}


def answer(cache: TtlCache, command: Command, **kwargs: Any) -> JSONResponse:
    """Run one command, memoised, and map its failures onto status codes.

    The cache is passed in rather than reached for: :func:`create_app` owns one
    per application, so two apps in one process (a test and its subject, most
    of all) cannot serve each other's answers.

    A refusal is data, not a crash: the client renders one panel as unavailable
    and keeps the rest. That is the same contract `status` relies on -- which is
    why *which* failures are survivable is decided once, in
    :func:`~src.interfaces.errors.attempt`, and only the rendering of them is
    decided here.

    Failures are deliberately not cached. A repeated 503 costs a repeated cloud
    read, which is the right trade: the alternative keeps serving "Azure is
    down" for the whole TTL after `az login` has already fixed it.
    """
    key = (command.name, tuple(sorted(kwargs.items())))
    payload, failure = attempt(lambda: cache.get(key, lambda: command.invoke(**kwargs)))
    if failure is not None:
        return PayloadResponse({"error": failure.message}, status_code=_STATUS[failure.kind])
    return PayloadResponse(payload)


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Share one materialised record between every reader, while serving.

    Scoped to the server rather than switched on globally, because the command
    line wants the opposite: it is one-shot, so it would gain nothing and would
    lose the guarantee its readers are built on -- that an answer is against the
    record as it is NOW. A run published thirty seconds ago must not be
    invisible to `promote`.
    """
    with workspace.shared_record_cache(RECORD_TREE_TTL_SECONDS):
        yield


def create_app() -> FastAPI:
    """Build the application, with a memo of its own.

    A function rather than a module-level app, so a test gets a genuinely fresh
    one: the cache is built here and closed over, so nothing survives between
    two applications in the same process.
    """
    app = FastAPI(title="poker-solver console", docs_url="/api/docs", lifespan=_lifespan)
    cache = TtlCache(CACHE_TTL_SECONDS)

    @app.get("/api/pool")
    def _pool() -> JSONResponse:
        return answer(cache, pool_status.COMMAND)

    @app.get("/api/jobs")
    def _jobs(limit: int = 20, all: bool = False) -> JSONResponse:  # noqa: A002
        return answer(cache, jobs.COMMAND, limit=limit, all=all)

    @app.get("/api/tasks")
    def _tasks(limit: int = 0, skip_reconcile: bool = False) -> JSONResponse:
        return answer(
            cache, tasks.COMMAND, limit=limit, skip_reconcile=skip_reconcile, tasks_dir=None
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

    @app.get("/api/cost")
    def _cost(hours: float = 0.0) -> JSONResponse:
        return answer(cache, cost.COMMAND, hours=hours, rate="")

    @app.get("/api/evals")
    def _evals(limit: int = 50) -> JSONResponse:
        return answer(
            cache, ledger.COMMAND, limit=limit, run=None, experiment=None, method=None, since=None
        )

    @app.get("/api/logs/{task_id}")
    def _log(task_id: str, lines: int = 200) -> JSONResponse:
        return answer(cache, logs.COMMAND, task=task_id, lines=lines)

    # THE FIRST WRITE THAT IS NOT ABOUT THE CONSOLE'S OWN BOX.
    #
    # Still one `Command.invoke`, and that rule matters MORE for a write than a
    # read: the moment a button does something `poker-solver` cannot, the console
    # has behaviour that is neither scriptable nor reproducible, which is exactly
    # what the previous browser UI died of. So there is no "cancel all", no
    # retry-then-cancel -- if a composite is wanted, it becomes a command first
    # and a button second.
    @app.post("/api/tasks/{job_id}/{task_id}/cancel")
    def _cancel(job_id: str, task_id: str) -> JSONResponse:
        return answer(TtlCache(0.0), cancel.COMMAND, job=job_id, task=task_id)

    # These three ARE commands, so they go through `answer` like the rest -- the
    # button and `poker-solver serve-box` are then the same code path, which is
    # the property that stops a second control surface existing. Not cached:
    # asking whether the box is up must not be answered from 15 seconds ago while
    # someone watches it boot.
    @app.get("/api/box")
    def _box() -> JSONResponse:
        return answer(
            TtlCache(0.0),
            serve_box.COMMAND,
            action="status",
            wait=False,
            resource_group=serve_box.DEFAULT_RESOURCE_GROUP,
            vm=serve_box.DEFAULT_VM,
            subscription=serve_box.DEFAULT_SUBSCRIPTION,
        )

    @app.post("/api/box/start")
    def _box_start() -> JSONResponse:
        return answer(
            TtlCache(0.0),
            serve_box.COMMAND,
            action="start",
            wait=False,
            resource_group=serve_box.DEFAULT_RESOURCE_GROUP,
            vm=serve_box.DEFAULT_VM,
            subscription=serve_box.DEFAULT_SUBSCRIPTION,
        )

    @app.post("/api/box/stop")
    def _box_stop() -> JSONResponse:
        return answer(
            TtlCache(0.0),
            serve_box.COMMAND,
            action="stop",
            wait=False,
            resource_group=serve_box.DEFAULT_RESOURCE_GROUP,
            vm=serve_box.DEFAULT_VM,
            subscription=serve_box.DEFAULT_SUBSCRIPTION,
        )

    # Not `answer(...)`: these are not commands and there is nothing to memoise
    # here. The blueprint server owns one loaded run and answers in
    # milliseconds, so a TTL cache would only serve a stale grid after a caller
    # walked to a different node.
    blueprint_proxy.mount(app)

    _mount_console(app)
    return app


def _warn_if_stale(index: Path) -> None:
    """Say so on stderr when the built console is older than its sources.

    Compared against the newest file under `console/src`, which is what a build
    consumes. Not fatal: an old build still serves every page it does have, and
    refusing to start would be a worse trade than a line of warning.
    """
    if not index.is_file():
        return
    sources = CONSOLE_DIST.parent / "src"
    if not sources.is_dir():
        return
    newest = max((f.stat().st_mtime for f in sources.rglob("*") if f.is_file()), default=0.0)
    if newest > index.stat().st_mtime:
        print(
            "WARNING: console/dist is older than console/src — pages added since "
            "the last build will be missing. Run `npm --prefix console run build`.",
            file=sys.stderr,
        )


def _mount_console(app: FastAPI) -> None:
    """Serve the built console, with SPA fallback.

    Client-side routing means `/tasks` and `/runs/abc` are not files: the server
    must return `index.html` and let the router decide. Anything under `/api`
    has already matched above, so the fallback cannot swallow a real endpoint.

    A missing build is reported rather than served as a blank page -- the
    failure otherwise looks like a broken app instead of a skipped build step.

    A STALE build is reported too, and that is the one that actually costs time.
    A missing page announces itself; a page built before the route you are
    looking for existed just silently lacks it, and the obvious conclusion is
    that the feature is broken rather than unbuilt. `serve` does not build, so
    this is reachable by simply forgetting.
    """
    index = CONSOLE_DIST / "index.html"
    _warn_if_stale(index)
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
