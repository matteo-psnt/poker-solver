"""The console's HTTP layer: commands, and composed views of commands.

**No new read path.** Every endpoint body reaches a ``Command.invoke`` and a
memo -- directly through :func:`answer`, or through :func:`view` for a screen
that is several commands at once. That is the whole design, and it is the
property the previous browser UI lacked: `fbcf9a8` carried
`api/chart_service.py`, `api/play_service.py` and `chart/data.py` -- a second way
to ask questions the CLI already answered, which drifted from it and then
rotted. `tests/interfaces/web/` fails if anything here grows one.

**Endpoints are ``def``, not ``async def``.** Every Azure client in
:mod:`src.interfaces.cloud` is synchronous, so a coroutine here would block the
event loop for the whole 2-4s of a cloud read and serialise every other request
behind it. FastAPI runs sync handlers in a threadpool, which is exactly right --
and is the single detail about this file most likely to be "tidied" into a bug.

**Two shapes of endpoint, and the reason both exist.** One per command is the
right grain for an ad-hoc question: a page fetches only what it shows, `/tasks`
does not pay for `ledger`, and each panel fails alone. It is the wrong grain for
a SCREEN -- four panels became four requests, each with its own cache slot and
its own poll cadence, so nothing on the Overview was ever the same age as
anything else, and `RunDetail` downloaded the entire task log to find one run's
rows. So `/api/view/*` composes: several commands fanned out concurrently,
joined once, served as one answer. See :mod:`src.interfaces.web.views` for the
rule that keeps composing from becoming computing.

The client still owns cadence (TanStack Query's ``refetchInterval``), so there
is no poller here and nothing depends on a background thread staying alive.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from typing import TYPE_CHECKING, Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.interfaces import telemetry
from src.interfaces.cloud.store import workspace
from src.interfaces.commands import (
    Command,
    activity,
    autoscale_check,
    cancel,
    compact_legs,
    configs,
    cost,
    curve,
    jobs,
    ledger,
    logs,
    pool_status,
    progress,
    push_code,
    push_data,
    runinfo,
    runs,
    score,
    serve_box,
    submit,
    submit_coupling,
    submit_precompute,
    submit_vector,
    tasks,
)
from src.interfaces.errors import attempt
from src.interfaces.web import blueprint_proxy, contract, views
from src.interfaces.web.cache import TtlCache
from src.shared import jsonio, repo

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path

# Long enough that several open tabs (or a remount) share one cloud sweep,
# short enough that a manual refresh feels live. A sweep is 2-4s per panel.
CACHE_TTL_SECONDS = 15.0

# The payload memo is per (command, arguments), so five endpoints asking five
# questions about the SAME record each paid ~120 round trips for their own copy.
# Held longer than `CACHE_TTL_SECONDS` because it is a different thing being
# cached: the payloads are what a panel shows, the tree is what they derive from.
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

# OpenAPI only -- FastAPI skips validation for a handler returning a `Response`,
# and every handler here returns :class:`PayloadResponse` to keep `jsonio.dumps`.
# The models are enforced by `tests/interfaces/web/test_contract.py` instead.
ERRORS: dict[int | str, dict[str, Any]] = {
    422: {"model": contract.ApiError, "description": "Understood, and the answer is no."},
    503: {"model": contract.ApiError, "description": "Azure did not answer."},
}


# A dispatch's arguments are many and repeatable, which a query string handles
# badly. These are NOT a second declaration of what a command accepts: an
# optional field means *omitted*, and :func:`given` drops those so the command's
# own parser supplies every default.


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
    # `--kernel` and the two groups it gates. Absent here for as long as they
    # existed on the command, so the console could queue only scalar runs and
    # only cold ones -- a gap that reads as a missing feature rather than a
    # missing field. `test_endpoint_arguments` now pins every body to its flags.
    kernel: str | None = None
    universe_boards: int | None = None
    universe_seed: int | None = None
    dtype: str | None = None
    warm_start_from: str | None = None
    warm_start_weight: int | None = None
    warm_start_at: int | None = None


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


class SubmitVectorBody(BaseModel):
    # `abstractions` is required and repeatable: one arm per abstraction per
    # kernel, and the command has no default for it because which abstractions
    # to compare IS the experiment.
    abstractions: list[str]
    kernels: list[str] | None = None
    derive_boards: list[int] | None = None
    train_boards: int | None = None
    score_boards: int | None = None
    checkpoints: str | None = None
    config: str | None = None
    stack: int | None = None
    timeout: str | None = None


class SubmitCouplingBody(BaseModel):
    # Repeatable and required for the same reason as the sweep's: WHICH
    # abstractions to price against each other is the measurement.
    abstractions: list[str]
    boards: int | None = None
    classes: str | None = None
    seed: int | None = None
    timeout: str | None = None


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


def _served(cache: TtlCache, key: tuple[Any, ...], produce: Callable[[], Any]) -> JSONResponse:
    """One memoised answer, with its failures mapped onto status codes.

    The surface goes on around the memo, not inside it: a request served from the
    cache did not run the command, and recording it as if it had would report a
    `tasks` that takes 5 seconds as one that mostly takes microseconds.

    Failures are deliberately NOT cached: a repeated 503 costs a repeated cloud
    read, and the alternative keeps serving "Azure is down" for the whole TTL
    after `az login` has fixed it.
    """
    with telemetry.surface("console"):
        payload, failure = attempt(lambda: cache.get(key, produce))
    if failure is not None:
        return PayloadResponse({"error": failure.message}, status_code=_STATUS[failure.kind])
    return PayloadResponse(payload)


def answer(cache: TtlCache, command: Command, /, **kwargs: Any) -> JSONResponse:
    """Run one command, memoised per (command, arguments).

    ``cache`` and ``command`` are POSITIONAL-ONLY and the ``/`` is load-bearing:
    everything after it is a command's own flags, and a command is free to have one
    called `command` -- `activity --command tasks` does. Without the slash that
    argument binds here instead, several frames from anything the reader was
    thinking about.

    The cache is passed in rather than reached for, so two apps in one process (a
    test and its subject) cannot serve each other's answers.
    """
    key = (command.name, tuple(sorted(kwargs.items())))
    return _served(cache, key, lambda: command.invoke(**kwargs))


@asynccontextmanager
async def _lifespan(_app: FastAPI) -> AsyncIterator[None]:
    """Share one materialised record between every reader, while serving.

    Scoped to the server rather than switched on globally, because the command
    line wants the opposite: it is one-shot, so it would gain nothing and would
    lose the guarantee its readers are built on -- that an answer is against the
    record as it is NOW. A run published thirty seconds ago must not be
    invisible to the next reader.
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

    @app.get("/api/pool", response_model=contract.Pool, responses=ERRORS)
    def _pool() -> JSONResponse:
        return answer(cache, pool_status.COMMAND)

    @app.get("/api/jobs", response_model=contract.Jobs, responses=ERRORS)
    def _jobs(limit: int = 20, all: bool = False) -> JSONResponse:  # noqa: A002
        return answer(cache, jobs.COMMAND, limit=limit, all=all)

    @app.get("/api/tasks", response_model=contract.Tasks, responses=ERRORS)
    def _tasks(limit: int = 0, skip_reconcile: bool = False) -> JSONResponse:
        return answer(cache, tasks.COMMAND, limit=limit, skip_reconcile=skip_reconcile)

    @app.get("/api/runs", response_model=contract.Runs, responses=ERRORS)
    def _runs(limit: int = 0) -> JSONResponse:
        return answer(cache, runs.COMMAND, limit=limit)

    @app.get("/api/runs/{run_id}", response_model=contract.RunInfo, responses=ERRORS)
    def _run(run_id: str) -> JSONResponse:
        return answer(cache, runinfo.COMMAND, run=run_id)

    @app.get("/api/runs/{run_id}/progress", response_model=contract.Progress, responses=ERRORS)
    def _progress(run_id: str, last: int = 0) -> JSONResponse:
        return answer(cache, progress.COMMAND, run=run_id, last=last)

    @app.get("/api/runs/{run_id}/curve", response_model=contract.Curve, responses=ERRORS)
    def _curve(run_id: str) -> JSONResponse:
        return answer(cache, curve.COMMAND, run=run_id)

    @app.get("/api/cost", response_model=contract.Cost, responses=ERRORS)
    def _cost(hours: float = 0.0) -> JSONResponse:
        return answer(cache, cost.COMMAND, hours=hours)

    @app.get("/api/evals", response_model=contract.Ledger, responses=ERRORS)
    def _evals(limit: int = 50) -> JSONResponse:
        return answer(cache, ledger.COMMAND, limit=limit)

    @app.get("/api/logs/{task_id}", response_model=contract.LogLines, responses=ERRORS)
    def _log(task_id: str, lines: int = 200) -> JSONResponse:
        return answer(cache, logs.COMMAND, task=task_id, lines=lines)

    # A local directory read, and the only endpoint here that touches neither
    # Azure nor the share. It is what makes the dispatch form offerable at all:
    # `submit --config` names a stem, and a surface that cannot enumerate them
    # has to make the operator type one from memory.
    @app.get("/api/configs", response_model=contract.Configs, responses=ERRORS)
    def _configs() -> JSONResponse:
        return answer(cache, configs.COMMAND)

    # Local, so it costs nothing and is memoised only to keep a shared tab from
    # re-reading the log every poll. It is also the one endpoint whose answer
    # this server's own requests keep changing.
    @app.get("/api/activity", response_model=contract.Activity, responses=ERRORS)
    def _activity(days: float = 7.0, limit: int = 20) -> JSONResponse:
        return answer(cache, activity.COMMAND, days=days, limit=limit)

    @app.get("/api/autoscale", response_model=contract.Autoscale, responses=ERRORS)
    def _autoscale() -> JSONResponse:
        return answer(cache, autoscale_check.COMMAND)

    # THE FIRST WRITE THAT IS NOT ABOUT THE CONSOLE'S OWN BOX.
    #
    # Still one `Command.invoke`, and that rule matters MORE for a write than a
    # read: the moment a button does something `poker-solver` cannot, the console
    # has behaviour that is neither scriptable nor reproducible, which is exactly
    # what the previous browser UI died of. So there is no "cancel all", no
    # retry-then-cancel -- if a composite is wanted, it becomes a command first
    # and a button second.
    @app.post(
        "/api/tasks/{job_id}/{task_id}/cancel", response_model=contract.Cancelled, responses=ERRORS
    )
    def _cancel(job_id: str, task_id: str) -> JSONResponse:
        return answer(TtlCache(0.0), cancel.COMMAND, job=job_id, task=task_id)

    # `TtlCache(0.0)`, every one: two identical `submit` bodies fifteen seconds
    # apart are two runs someone wants, and a dispatch must not be made to look
    # idempotent.

    @app.post("/api/submit", response_model=contract.SubmitPayload, responses=ERRORS)
    def _submit(body: SubmitBody) -> JSONResponse:
        return answer(TtlCache(0.0), submit.COMMAND, **given(body))

    @app.post("/api/score", response_model=contract.ScorePayload, responses=ERRORS)
    def _score(body: ScoreBody) -> JSONResponse:
        return answer(TtlCache(0.0), score.COMMAND, **given(body))

    @app.post(
        "/api/precompute", response_model=contract.PrecomputeDispatchPayload, responses=ERRORS
    )
    def _precompute(body: PrecomputeBody) -> JSONResponse:
        return answer(TtlCache(0.0), submit_precompute.COMMAND, **given(body))

    @app.post("/api/submit-vector", response_model=contract.SubmitVectorPayload, responses=ERRORS)
    def _submit_vector(body: SubmitVectorBody) -> JSONResponse:
        return answer(TtlCache(0.0), submit_vector.COMMAND, **given(body))

    @app.post(
        "/api/submit-coupling", response_model=contract.SubmitCouplingPayload, responses=ERRORS
    )
    def _submit_coupling(body: SubmitCouplingBody) -> JSONResponse:
        return answer(TtlCache(0.0), submit_coupling.COMMAND, **given(body))

    # `push-code` and `push-data` read a tree on the machine RUNNING THIS
    # SERVER, which is the one fact about them a browser hides. `--root` and
    # `--source` default to this checkout, so a console served from a different
    # worktree publishes that worktree -- and the payload names what it sealed,
    # which is what the page shows back.
    @app.post("/api/push-code", response_model=contract.PushedCode, responses=ERRORS)
    def _push_code(body: PushCodeBody) -> JSONResponse:
        return answer(TtlCache(0.0), push_code.COMMAND, **given(body))

    @app.post("/api/push-data", response_model=contract.PushedData, responses=ERRORS)
    def _push_data(body: PushDataBody) -> JSONResponse:
        return answer(TtlCache(0.0), push_data.COMMAND, **given(body))

    @app.post("/api/compact-legs", response_model=contract.Compacted, responses=ERRORS)
    def _compact_legs(body: CompactBody) -> JSONResponse:
        return answer(TtlCache(0.0), compact_legs.COMMAND, **given(body))

    # These three ARE commands, so they go through `answer` like the rest -- the
    # button and `poker-solver serve-box` are then the same code path, which is
    # the property that stops a second control surface existing. Not cached:
    # asking whether the box is up must not be answered from 15 seconds ago while
    # someone watches it boot.
    #
    # WHICH box is the command's own answer, and these three used to ask it to
    # repeat itself: `--resource-group`, `--vm` and `--subscription` were passed
    # back at their parser defaults, so the VM's name lived here as well as in
    # `serve_box.py`. `action` stays spelled out -- these differ in exactly one
    # argument, and leaving it implicit on the one whose value is the default
    # makes the set read as though it were doing something else.
    @app.get("/api/box", response_model=contract.Box, responses=ERRORS)
    def _box() -> JSONResponse:
        return answer(TtlCache(0.0), serve_box.COMMAND, action="status")

    @app.post("/api/box/start", response_model=contract.Box, responses=ERRORS)
    def _box_start() -> JSONResponse:
        return answer(TtlCache(0.0), serve_box.COMMAND, action="start")

    @app.post("/api/box/stop", response_model=contract.Box, responses=ERRORS)
    def _box_stop() -> JSONResponse:
        return answer(TtlCache(0.0), serve_box.COMMAND, action="stop")

    # Several commands fanned out concurrently and joined -- served through the
    # same memo and the same failure ladder as a single command, so a screen is
    # classified and recorded exactly like the panels it is made of.
    # `build.__name__` stands in for a command name in the cache key.

    def view(build: Any, *key: str) -> JSONResponse:
        return _served(cache, (build.__name__, key), lambda: build(*key))

    @app.get("/api/view/now", response_model=contract.NowView, responses=ERRORS)
    def _view_now() -> JSONResponse:
        return view(views.now)

    @app.get("/api/view/runs", response_model=contract.RunsView, responses=ERRORS)
    def _view_runs() -> JSONResponse:
        return view(views.runs)

    @app.get("/api/view/run/{run_id}", response_model=contract.RunView, responses=ERRORS)
    def _view_run(run_id: str) -> JSONResponse:
        return view(views.run, run_id)

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
