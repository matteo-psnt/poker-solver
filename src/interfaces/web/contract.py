"""The API's vocabulary: what each endpoint answers, NAMED rather than declared.

``response_model`` puts these in the OpenAPI schema and
`console/src/api/types.gen.ts` is generated from it -- the TypeScript is not
written by anyone, and never should be.

**Every command payload here is IMPORTED.** The shape is declared once, in the
command that constructs it, and this file gives it the name the API uses. What
is still DECLARED below is neither a command's payload nor meant to stay: the
composed views, which are the console's own shapes and belong to no command, and
the blueprint server's, which is a separate process reached over HTTP precisely
so the console never imports the engine. If those start to drift, declare them
in `src/interfaces/blueprint/app.py` and import them here.

``extra="allow"`` on :class:`Payload` covers only the classes below. An imported
command payload IS the payload, so there is nothing extra to tolerate.

This does not validate at request time and cannot: FastAPI skips validation and
serialization for a handler returning a ``Response``, and every endpoint returns
:class:`PayloadResponse` to keep `jsonio.dumps`, which handles the numpy scalars
and ``Path`` objects these payloads carry. The guarantee is the CONSTRUCTOR,
checked statically, plus `tests/interfaces/web/test_contract.py`.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from src.interfaces.blueprint.app import (
    BlueprintLoad,
    BlueprintRun,
    Bucket,
    Combos,
    Edge,
    Hand,
    HandEvent,
    LeftSession,
    NodeGrid,
    SolverNode,
)
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

# Aliased so the API's vocabulary stays the API's: these are the names the
# generated TypeScript uses. What is still DECLARED below is a command not yet
# typed, and that list is meant to shrink to nothing.
__all__ = [
    "Activity",
    "Arm",
    "Autoscale",
    "BatchTask",
    "Billed",
    "BlueprintLoad",
    "BlueprintRun",
    "Box",
    "Bucket",
    "Cancelled",
    "Combos",
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
    "Edge",
    "Failure",
    "Hand",
    "HandEvent",
    "Job",
    "Jobs",
    "Ledger",
    "LeftSession",
    "LogLines",
    "NodeGrid",
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
    "SolverNode",
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


# The blueprint server is a separate process, reached over HTTP so the console
# never imports the engine. Its shapes are declared here rather than imported
# because they belong to no command.
# A view is several commands answered at once. `payload` is typed per view
# rather than a bare dict, so the generated TypeScript keeps the part's type.
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
    elapsed_seconds: float
    """Wall clock of the whole fan-out. Equal to the sum of its parts means the
    concurrency has silently become serial."""


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
    jobs: Part[Jobs]
    """A larger job limit than the live screen -- see `views.RUN_LIST_JOB_LIMIT`."""
    tasks: Part[TasksSummary]
    """Answered, then trimmed to the `task_runs` projection -- so the type has no rows."""


class RunsView(View):
    """Every published run, plus what is needed to check its claimed status."""

    parts: RunsParts
    task_runs: dict[str, str] = {}
    """`task_id -> run_id`. Only the join; which Batch states count as live is
    the client's call."""


class RunParts(Payload):
    run: Part[RunInfo]
    progress: Part[Progress]
    curve: Part[Curve]
    evals: Part[Ledger]
    tasks: Part[TasksSummary]
    """Answered, then trimmed: the rows reach the client as `run_tasks`."""


class RunView(View):
    """Everything about one run, and the tasks that built it."""

    parts: RunParts
    run_tasks: list[TaskRow] = []
    """Empty when the `tasks` part FAILED, which is not the same as no tasks --
    the part carries its own error, and that is what the UI renders."""


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
