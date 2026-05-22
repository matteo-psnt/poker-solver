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
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.interfaces.cloud.store import workspace
from src.interfaces.commands import (
    Command,
    autoscale_check,
    cancel,
    compact_legs,
    compare,
    configs,
    cost,
    curve,
    jobs,
    ledger,
    logs,
    pool_status,
    progress,
    promote,
    push_code,
    push_data,
    report,
    runinfo,
    runs,
    score,
    serve_box,
    submit,
    submit_precompute,
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


"""Request bodies, and why the writes take one at all
---------------------------------------------------
A read's arguments are few and short, so they ride in the query string. A
dispatch's are neither: `submit` alone carries nine, one of them a repeatable
list of config overrides, and a repeated query parameter is the shape most
likely to arrive as a bare string rather than a list of one.

These models are NOT a second declaration of what a command accepts. An
optional field means *omitted*, spelled `None`, and :func:`given` drops those
before `invoke` so the command's OWN parser supplies every default. Writing the
defaults again here is the failure mode being avoided: a default that disagreed
with its flag would make the console and the command line do different things
from the same input, and nothing would report it.

That is also what keeps the destructive halves safe by construction rather than
by care. `compact-legs --delete` is the irreversible one -- its own help says so
-- and an omitted `delete` reaches argparse's `store_true` default of False,
which is the dry run.
"""


def given(body: BaseModel) -> dict[str, Any]:
    """The fields a caller actually supplied, ready to splat into ``invoke``."""
    return {key: value for key, value in body.model_dump().items() if value is not None}


class SubmitBody(BaseModel):
    to: int
    config: str | None = None
    run: str | None = None
    experiment: str | None = None
    arm: str | None = None
    parent: str | None = None
    sets: list[str] | None = None
    workers: int | None = None
    checkpoint_every: int | None = None
    timeout: str | None = None


class ScoreBody(BaseModel):
    run: str
    method: str | None = None
    at: str | None = None
    timeout: str | None = None
    # `score`'s REMAINDER passthrough, already split. The `--` separator is a
    # command-line artefact -- argparse's way of being told the rest is not its
    # business -- so it has no meaning here and `_passthrough` tolerates its
    # absence.
    flags: list[str] | None = None


class PrecomputeBody(BaseModel):
    config: str
    workers: int | None = None
    timeout: str | None = None
    force: bool | None = None


class PushCodeBody(BaseModel):
    root: str | None = None


class PushDataBody(BaseModel):
    source: str | None = None
    name: str | None = None


class CompactBody(BaseModel):
    apply: bool | None = None
    delete: bool | None = None
    backup: str | None = None
    label: str | None = None


class PromoteBody(BaseModel):
    run: str
    rationale: str


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

    # A local directory read, and the only endpoint here that touches neither
    # Azure nor the share. It is what makes the dispatch form offerable at all:
    # `submit --config` names a stem, and a surface that cannot enumerate them
    # has to make the operator type one from memory.
    @app.get("/api/configs")
    def _configs() -> JSONResponse:
        return answer(cache, configs.COMMAND, kind="")

    @app.get("/api/autoscale")
    def _autoscale() -> JSONResponse:
        return answer(cache, autoscale_check.COMMAND)

    @app.get("/api/experiments/{experiment_id}")
    def _report(experiment_id: str) -> JSONResponse:
        return answer(cache, report.COMMAND, experiment=experiment_id)

    @app.get("/api/compare")
    def _compare(
        a: str, b: str, a_at: int | None = None, b_at: int | None = None, force: bool = False
    ) -> JSONResponse:
        # `None` is `--a-at`'s own default and means "take the newest row for
        # this run" -- a real state, since most runs have exactly one eval. An
        # absent query parameter has to arrive as that same sentinel rather than
        # as 0, which the ledger would read as a rung and never match.
        return answer(cache, compare.COMMAND, a=a, b=b, a_at=a_at, b_at=b_at, force=force)

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

    """The dispatching writes, and the one property they all share
    ------------------------------------------------------------
    `TtlCache(0.0)`, every one of them. The memo above keys on (command,
    arguments), which is exactly right for a read and exactly wrong here: two
    identical `submit` bodies fifteen seconds apart are two runs someone wants,
    and serving the second from the first's payload would report a job id for
    work that was never queued. A dispatch is not idempotent and must not be
    made to look like it.

    Sealing a code snapshot and queueing a task is ~10-20s, which is longer than
    a person waits before clicking again -- so the CLIENT disables the control
    while one is in flight. That is a nicety, not the guarantee; nothing here
    can tell a double-click from a deliberate second submission, and it should
    not try.
    """

    @app.post("/api/submit")
    def _submit(body: SubmitBody) -> JSONResponse:
        return answer(TtlCache(0.0), submit.COMMAND, **given(body))

    @app.post("/api/score")
    def _score(body: ScoreBody) -> JSONResponse:
        return answer(TtlCache(0.0), score.COMMAND, **given(body))

    @app.post("/api/precompute")
    def _precompute(body: PrecomputeBody) -> JSONResponse:
        return answer(TtlCache(0.0), submit_precompute.COMMAND, **given(body))

    # `push-code` and `push-data` read a tree on the machine RUNNING THIS
    # SERVER, which is the one fact about them a browser hides. `--root` and
    # `--source` default to this checkout, so a console served from a different
    # worktree publishes that worktree -- and the payload names what it sealed,
    # which is what the page shows back.
    @app.post("/api/push-code")
    def _push_code(body: PushCodeBody) -> JSONResponse:
        return answer(TtlCache(0.0), push_code.COMMAND, **given(body))

    @app.post("/api/push-data")
    def _push_data(body: PushDataBody) -> JSONResponse:
        return answer(TtlCache(0.0), push_data.COMMAND, **given(body))

    @app.post("/api/compact-legs")
    def _compact_legs(body: CompactBody) -> JSONResponse:
        return answer(TtlCache(0.0), compact_legs.COMMAND, **given(body))

    @app.post("/api/promote")
    def _promote(body: PromoteBody) -> JSONResponse:
        return answer(TtlCache(0.0), promote.COMMAND, **given(body))

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
            if _path.startswith("api/"):
                return JSONResponse({"error": f"No such endpoint: /{_path}"}, status_code=404)
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
    def _spa(_path: str) -> Response:
        # An `/api` path that reached here matched no endpoint, and answering it
        # with the shell is the wrong answer twice: a mistyped URL comes back
        # 200 with HTML, and a WRITE endpoint fetched with GET looks like it
        # worked. `client.get("/api/submit")` returning the console is the
        # single most confusing thing this fallback can do.
        if _path.startswith("api/"):
            return JSONResponse({"error": f"No such endpoint: /{_path}"}, status_code=404)
        return FileResponse(index)
