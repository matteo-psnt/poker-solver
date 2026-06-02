"""The API's vocabulary: what each endpoint answers, NAMED rather than declared.

``response_model`` puts these in the OpenAPI schema and
`console/src/api/types.gen.ts` is generated from it -- the TypeScript is not
written by anyone, and never should be.

**Every command payload here is IMPORTED.** The shape is declared once, in the
command that constructs it, and this file gives it the name the API uses. That
is the whole design, and it is new: these models used to be a second, parallel
declaration of every payload, hand-written and hand-maintained. Measured on
2026-08-13 -- renaming a REQUIRED field in `jobs.py` passed 1061 tests, because
the model here and the `PAYLOADS` fixture were both hand-written and drifted
together, away from what `run()` actually returned. `ty` makes that a
pre-commit failure now.

What is still DECLARED in this file, and why
--------------------------------------------
Two kinds, and neither is a command's payload:

- **The composed views** (`Part`, `View`, `NowView` and friends). These are the
  console's own shapes -- `web/views.py` builds them and no command owns one --
  so this is where they belong.
- **The blueprint server's shapes.** That is a SEPARATE PROCESS holding a loaded
  run, reached over HTTP through `/api/blueprint/*` precisely so the console
  never imports the engine. They are declared here as a client describing a
  server it cannot import from, which is the one place a second declaration is
  the honest answer rather than a smell. They remain hand-maintained; if they
  start to drift, the fix is the same move -- declare them in
  `src/interfaces/blueprint/app.py` and import them here.

Why the view models stay lenient
--------------------------------
``extra="allow"`` on :class:`Payload`, which only the classes below inherit. An
imported command payload does not need it: the model IS the payload, so there is
nothing extra to tolerate.

Why this does not validate at request time
------------------------------------------
It cannot. FastAPI skips validation and serialization for a handler that returns
a ``Response``, and every endpoint here returns :class:`PayloadResponse` -- which
exists because `jsonio.dumps` handles the numpy scalars and ``Path`` objects
these payloads carry, and that plain ``json.dumps`` turns into a 500 the CLI
never sees. Measured: a handler declaring ``response_model`` and returning a
wrong-shaped ``JSONResponse`` answers 200 with the wrong shape, while
`/openapi.json` still carries the right ``$ref``.

So the request-time guarantee is the CONSTRUCTOR, checked statically, plus
`tests/interfaces/web/test_contract.py` round-tripping each model against the
`PAYLOADS` examples -- which are constructor calls now, not literals.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.interfaces.cloud.cost.billing import BilledPayload as Billed
from src.interfaces.cloud.cost.billing import ServiceCharge, StandingCharge
from src.interfaces.cloud.cost.node_time import ConcurrencyPoint
from src.interfaces.cloud.tasks.batch import BatchTask, Job, ResizeError
from src.interfaces.cloud.tasks.dispatch import Dispatched
from src.interfaces.commands.activity import ActivityPayload as Activity
from src.interfaces.commands.activity import CommandActivity, Failure
from src.interfaces.commands.autoscale_check import AutoscalePayload as Autoscale
from src.interfaces.commands.cancel import CancelledPayload as Cancelled
from src.interfaces.commands.compact_legs import CompactedPayload as Compacted
from src.interfaces.commands.compare import ComparePayload as Comparison
from src.interfaces.commands.compare import PairedComparison
from src.interfaces.commands.configs import ConfigKind
from src.interfaces.commands.configs import ConfigsPayload as Configs
from src.interfaces.commands.cost import CostPayload as Cost
from src.interfaces.commands.curve import CurvePayload as Curve
from src.interfaces.commands.curve import CurvePoint
from src.interfaces.commands.jobs import JobsPayload as Jobs
from src.interfaces.commands.ledger import LedgerPayload as Ledger
from src.interfaces.commands.logs import LogsPayload as LogLines
from src.interfaces.commands.pool_status import PoolPayload as Pool
from src.interfaces.commands.progress import ProgressPayload as Progress
from src.interfaces.commands.promote import PromotedPayload as Promoted
from src.interfaces.commands.push_code import PushedCodePayload as PushedCode
from src.interfaces.commands.push_data import PushedDataPayload as PushedData
from src.interfaces.commands.report import ArmResult as Arm
from src.interfaces.commands.report import ReportPayload as Report
from src.interfaces.commands.runinfo import RunInfoPayload as RunInfo
from src.interfaces.commands.runs import RunsPayload as Runs
from src.interfaces.commands.runs import RunSummary
from src.interfaces.commands.score import ScorePayload
from src.interfaces.commands.serve_box import BoxPayload as Box
from src.interfaces.commands.submit import SubmitPayload
from src.interfaces.commands.submit_precompute import PrecomputeDispatchPayload
from src.interfaces.commands.submit_vector import SubmitVectorPayload, VectorArm
from src.interfaces.commands.tasks import TasksPayload as Tasks
from src.interfaces.commands.tasks import TasksSummary
from src.shared.task_history import TaskProgress, TaskRow

"""What is IMPORTED here, and why that is the whole point
-------------------------------------------------------
A payload model that a command CONSTRUCTS is imported, never restated. Below,
the shapes still declared in this file are the ones whose command has not been
typed yet -- each is a second declaration of something, and the list is meant to
shrink to nothing.

The re-exports are not ceremony: `Jobs`, `Job` and `BatchTask` are the names the
generated TypeScript uses, `response_model` needs the class, and a console
importing `JobsPayload` would be reaching into the command layer for a name it
has no business knowing. Aliasing here keeps the API's vocabulary the API's.
"""
__all__ = [
    "Activity",
    "Arm",
    "Autoscale",
    "BatchTask",
    "Billed",
    "Box",
    "Cancelled",
    "CommandActivity",
    "Compacted",
    "Comparison",
    "ConcurrencyPoint",
    "ConfigKind",
    "Configs",
    "Cost",
    "Curve",
    "CurvePoint",
    "Dispatched",
    "Failure",
    "Job",
    "Jobs",
    "Ledger",
    "LogLines",
    "PairedComparison",
    "Pool",
    "PrecomputeDispatchPayload",
    "Progress",
    "Promoted",
    "PushedCode",
    "PushedData",
    "Report",
    "ResizeError",
    "RunInfo",
    "RunSummary",
    "Runs",
    "ScorePayload",
    "ServiceCharge",
    "StandingCharge",
    "SubmitPayload",
    "SubmitVectorPayload",
    "TaskProgress",
    "TaskRow",
    "Tasks",
    "TasksSummary",
    "VectorArm",
]


class Payload(BaseModel):
    """Base for every payload model: describes what is read, tolerates the rest."""

    model_config = ConfigDict(extra="allow")


"""The pool, and what Batch is running
------------------------------------"""


"""The durable task account -- the only thing that can say why a task DIED
------------------------------------------------------------------------
The run log cannot record a death: the container is gone first. The wrapper
writes its own account to the share, and `tasks` reconciles the ones whose exit
record never landed against Batch's view.
"""


"""`TaskRow`, `TaskProgress`, `Tasks` and `TasksSummary` are all IMPORTED.

`TaskRow` comes from `task_history`, which assembles it, rather than from the
command that ships it: the shape is the reader's, and `report` joins the same
rows into a run digest without going near an endpoint.

`Tasks` and `TasksSummary` are two models on purpose -- see `TasksSummary` for
what one model describing both cost.
"""


"""The record: runs, their training history, and what they scored
---------------------------------------------------------------"""


"""What it all cost
-----------------"""


"""Experiments, and the comparisons that decide them
--------------------------------------------------"""


"""This tool's own behaviour, and the local reads
-----------------------------------------------"""


"""The writes
-----------"""


"""The blueprint server's shapes, proxied through `/api/blueprint/*`
------------------------------------------------------------------
Unlike everything above, these do NOT correspond to a command: the blueprint
server is a separate process holding a loaded run, and the console reaches it
over HTTP precisely so it never imports the engine. They are declared here
anyway, because a generated client should describe every endpoint the console
calls -- and its refusals arrive as the same `{error}` body either way.
"""


class BlueprintRun(Payload):
    run: str
    starting_stack: int
    small_blind: int
    big_blind: int
    combos: int
    """The run being swapped in, while one is. Null when nothing is loading."""
    loading: str | None = None
    """False on a server handed a blueprint directly -- a laptop, a test."""
    can_switch: bool = False


class BlueprintLoad(Payload):
    """The 202 from asking for a swap. The work outlives the request."""

    run: str
    loading: bool


class Combos(Payload):
    combos: list[str] = []


class LeftSession(Payload):
    """Confirmation that a play session was dropped on the far side.

    The session lives where the blueprint does, so leaving is a request rather
    than a local forget -- which is also why the proxy holds no state and a
    console restart does not lose a hand in progress.
    """

    session: str
    dropped: bool


class Bucket(Payload):
    """`strategy` is null exactly when `trained` is false -- the server refuses to
    emit the uniform an allocated-but-unvisited row would otherwise read as, and
    this keeps that distinction rather than defaulting it away."""

    trained: bool
    strategy: list[float] | None
    reach_count: int


class Edge(Payload):
    token: str
    type: str
    amount: float


class NodeGrid(Payload):
    street: str
    board: list[str] = []
    actor: int
    actions: list[str] = []
    """-1 where the board blocks the combo. Kept in place rather than filtered,
    so index i is always ALL_COMBOS[i]."""
    combo_buckets: list[int] = []
    blocked: int
    trained_buckets: int
    buckets: dict[str, Bucket] = {}


class SolverNode(Payload):
    path: str
    terminal: bool
    board: list[str] = []
    grid: NodeGrid | None
    children: list[Edge] = []


class HandEvent(Payload):
    seat: int
    actor: str
    action: str
    amount: float
    street: str
    untrained: bool
    """Null until the hand is over -- see :class:`Hand`."""
    mix: list[tuple[str, float]] | None


class Hand(Payload):
    """A hand in progress.

    `bot_hole_cards` and every `mix` are null until `over`: the server withholds
    them, and this says so, because a client that received them could show them
    and a sit-down where you see the opponent's hand measures nothing.
    """

    session: str
    over: bool
    street: str
    board: list[str] = []
    pot: float
    stacks: list[float] = []
    human_seat: int
    button: int
    to_act: int | None
    hole_cards: list[str] = []
    bot_hole_cards: list[str] | None
    legal: list[Edge] = []
    payoff: float | None
    showdown: bool
    bot_decisions: int
    bot_untrained_decisions: int
    log: list[HandEvent] = []


"""The composed views
-------------------
A view is several commands answered at once. Its envelope is the same for all
three; what differs is which parts it carries and what it joined.

`payload` is typed per view rather than as a bare dict, so the generated
TypeScript knows that `parts.pool.payload` is a Pool -- which is the whole
reason to declare an envelope rather than return `dict[str, Any]` and lose the
type at the boundary.
"""


class Part[T](Payload):
    """One command's contribution: its payload, or why there is not one.

    Exactly one of these is set. A part that failed keeps its place in the view
    so the UI greys one panel and keeps the rest -- which is the property the
    whole fan-out exists for, and it disappears if a failed part is simply
    omitted.
    """

    payload: T | None = None
    error: str | None = None


class View(Payload):
    """What every composed view carries, whatever it is a view OF."""

    op: str
    at: str
    """The wall clock of the whole fan-out. The number that says whether the
    concurrency is working: a composed screen whose elapsed time equals the sum
    of its parts has silently become serial, and nothing else would show it."""
    elapsed_seconds: float


class NowParts(Payload):
    pool: Part[Pool]
    jobs: Part[Jobs]
    tasks: Part[Tasks]
    autoscale: Part[Autoscale]
    cost: Part[Cost]


class NowView(View):
    """What is happening right now, and did anything die."""

    parts: NowParts


class RunsParts(Payload):
    runs: Part[Runs]
    """Fetched with a LARGER job limit than the live screen uses, because a run
    outlives the daily job its tasks land in -- see `views.RUN_LIST_JOB_LIMIT`."""
    jobs: Part[Jobs]
    """Answered, then trimmed: what the client needs from it is the `task_runs`
    projection below, not the rows -- so the type has none."""
    tasks: Part[TasksSummary]


class RunsView(View):
    """Every published run, plus what is needed to check its claimed status."""

    parts: RunsParts
    """`task_id -> run_id`, projected from the task log.

    A run's `status` is a claim written by a living process, so it cannot record
    how an attempt died -- a run whose task was OOM-killed claims `running`
    forever. Cross-checking needs Batch (which TASKS are live) joined to the task
    log (which RUN each task was for).

    Only the join is here. Which Batch states count as live, and whether a run
    with no live task reads as "abandoned" or "abandoned?", is the client's --
    that is wording and semantics, not data, and it is unit-tested there.
    """
    task_runs: dict[str, str] = {}


class RunParts(Payload):
    run: Part[RunInfo]
    progress: Part[Progress]
    curve: Part[Curve]
    evals: Part[Ledger]
    """Answered, then trimmed: the rows themselves reach the client as
    `run_tasks` below, filtered to this run -- so the type has none."""
    tasks: Part[TasksSummary]


class RunView(View):
    """Everything about one run, and the tasks that built it."""

    parts: RunParts
    """The task log filtered to this run, joined server-side.

    Empty when the `tasks` part FAILED, which is not the same as this run having
    no tasks -- the part carries its own error, and that is what the UI must
    render. A join cannot signal failure and must not try.
    """
    run_tasks: list[TaskRow] = []


class ExperimentParts(Payload):
    report: Part[Report]
    runs: Part[Runs]


class ExperimentView(View):
    """One experiment's arms, each pinned to the run record behind it."""

    parts: ExperimentParts
    arm_runs: list[RunSummary] = []


class ApiError(Payload):
    """A refusal (422) or an outage (503). The only non-payload body served."""

    error: str
