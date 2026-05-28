"""The payload contract, declared ONCE, in Python, beside what produces it.

What this replaces
------------------
The shapes used to be written twice. A handler returned an untyped dict; a
hand-written `PAYLOADS` example in a Python test pinned it; a generated
`payloads.fixture.json` exported that example; and `console/src/api/schemas.ts`
declared all of it AGAIN as 682 lines of hand-written Zod, checked against the
fixture. Adding one field meant editing Python, a Python test and TypeScript,
and the only thing that caught you forgetting the third was a test that failed
by name after the fixture regenerated. `schemas.ts` opened by claiming it was
"ONE declaration rather than two"; it was the second declaration of every
payload in the system.

Now: these models are the declaration, `response_model` puts them in the OpenAPI
schema, and `console/src/api/types.gen.ts` is GENERATED from that schema. The
TypeScript is no longer written by anyone.

Why the models are lenient
--------------------------
``extra="allow"`` throughout, which is the Zod ``.passthrough()`` these came
from. A payload gaining a key is not a reason to break the console, and these
models deliberately describe only what a surface READS -- they are a contract,
not a mirror. The commands remain free to return more.

Why this does not validate at request time
------------------------------------------
It cannot, and the attempt would be worse than the absence. FastAPI skips
validation and serialization for a handler that returns a ``Response``, and
every endpoint here returns :class:`PayloadResponse` -- which exists because
`jsonio.dumps` handles the numpy scalars and ``Path`` objects that reach these
payloads (`runinfo` is `dataclasses.asdict` of a digest) and that plain
``json.dumps`` turns into a 500 the CLI never sees. Measured: a handler
declaring ``response_model`` and returning a wrong-shaped ``JSONResponse``
answers 200 with the wrong shape, while `/openapi.json` still carries the right
``$ref``.

So the enforcement is a TEST -- `tests/interfaces/web/test_contract.py` -- which
round-trips every model against the `PAYLOADS` examples the renderer tests
already pin. That closes the loop with one hand-written declaration instead of
two: change a payload, `PAYLOADS` must change, and the model then fails against
it.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict


class Payload(BaseModel):
    """Base for every payload model: describes what is read, tolerates the rest."""

    model_config = ConfigDict(extra="allow")


"""The pool, and what Batch is running
------------------------------------"""


class ResizeError(Payload):
    code: str | None = None
    message: str | None = None
    values: dict[str, str | None] = {}


class Pool(Payload):
    op: Literal["pool-status"]
    pool_id: str
    allocation_state: str | None
    current_dedicated_nodes: int | None
    target_dedicated_nodes: int | None
    vm_size: str | None
    hourly_cost: str | None = None
    resize_errors: list[ResizeError] = []


class BatchTask(Payload):
    task: str
    state: str | None
    exit_code: int | None
    node: str | None = None
    """Orders the queue: a task that has not started has no start_time."""
    created: str | None = None
    start_time: str | None = None
    end_time: str | None = None


class Job(Payload):
    job: str
    state: str | None
    tasks: list[BatchTask] = []


class Jobs(Payload):
    op: Literal["jobs"]
    jobs: list[Job] = []
    total_jobs: int
    hidden_jobs: int


"""The durable task account -- the only thing that can say why a task DIED
------------------------------------------------------------------------
The run log cannot record a death: the container is gone first. The wrapper
writes its own account to the share, and `tasks` reconciles the ones whose exit
record never landed against Batch's view.
"""


class TaskProgress(Payload):
    done: int
    total: int
    unit: str


class TaskRow(Payload):
    task_id: str
    attempt: int | None
    op: str | None
    run_id: str | None
    cause: str | None
    exit_code: int | None
    ended_at: str | None
    started_at: str | None = None
    job_id: str | None = None
    node_id: str | None = None
    """Derived server-side so the terminal and the console word it identically."""
    what: str | None = None
    """Seconds left, derived server-side so both surfaces agree."""
    eta_seconds: float | None = None
    """What a FINISHED task achieved, and how wide it ran -- the two things an
    estimate for the next one is built from."""
    units: float | None = None
    workers: int | None = None
    """WHICH CODE RAN. Three answers that do not replace one another: the commit
    names a history, the branch names the line of work (several worktrees share a
    commit and differ only in what is uncommitted), and the snapshot names the
    exact bytes. Absent on every task recorded before this existed."""
    code_snapshot: str | None = None
    git_commit: str | None = None
    git_dirty: str | None = None
    git_branch: str | None = None
    """Present only while a task is running, and only if its kind can say."""
    progress: TaskProgress | None = None


class Tasks(Payload):
    op: Literal["tasks"]
    rows: list[TaskRow] = []
    reconciled: int | None
    hidden_rows: int | None = None
    """How many rows the log held before a view trimmed them to a join.

    Set by `views._summarised` and absent everywhere else. A run page showing two
    tasks out of a log of four hundred has to be able to say which number is
    which -- a join that quietly returns few rows out of many is
    indistinguishable from a join that is broken.
    """
    source_rows: int | None = None


"""The record: runs, their training history, and what they scored
---------------------------------------------------------------"""


class RunSummary(Payload):
    name: str
    commits_ago: int | None
    git_dirty: bool | None
    loadable: bool
    blocker: str | None
    iterations: int | None
    num_infosets: int | None
    config_name: str | None
    status: str | None
    """Lineage. The run listing is the only place the SET of experiments exists
    -- `report` takes an id and `runinfo` reports one run's -- so the experiment
    picker is built from these. Absent on every legacy run."""
    experiment_id: str | None = None
    arm: str | None = None


class Runs(Payload):
    op: Literal["runs"]
    runs: list[RunSummary] = []


class ProgressRow(Payload):
    iteration: int
    coverage: float | None
    """THE convergence diagnostic. Compare against the 1e3-1e4 regret updates per
    infoset CFR needs before an average carries signal; coverage saturates early
    and says nothing about it."""
    mean_visits_per_touched: float | None = None
    iters_per_sec: float | None = None


class RunInfo(Payload):
    op: Literal["runinfo"]
    run_id: str
    config_name: str | None
    status: str | None
    iterations: int | None
    runtime_seconds: float | None
    """The run's own count, which legitimately differs from what the task log
    observed."""
    training_tasks: int | None
    git_commit: str | None
    card_abstraction_hash: str | None
    """TRUNCATED to `--last` (default eight). `progress` with `last=0` is the
    full series -- the console's chart drew 8 of 112 checkpoints from this field
    and looked like a complete history."""
    progress: list[ProgressRow] = []


class Progress(Payload):
    op: Literal["progress"]
    run_id: str
    total_rows: int
    coverage_plateau_iteration: int | None
    rows: list[ProgressRow] = []


class CurvePoint(Payload):
    iteration: int
    exploitability_mbb: float | None
    std_error_mbb: float | None


class Curve(Payload):
    op: Literal["curve"]
    run_id: str
    tier: str | None
    points: list[CurvePoint] = []
    """Rungs with no score. A sparse line and a complete one look identical once
    drawn, so a chart that silently omits most of its points reads as a finished
    measurement."""
    missing_iterations: list[int] = []


class LedgerRow(Payload):
    run_id: str | None = None
    eval_git_commit: str | None = None
    knobs: dict[str, Any] = {}
    results: dict[str, Any] = {}


class Ledger(Payload):
    op: Literal["ledger"]
    ledger: str
    rows: list[LedgerRow] = []


class LogLines(Payload):
    op: Literal["logs"]
    task: str | None
    lines: list[str] | None


"""What it all cost
-----------------"""


class StandingCharge(Payload):
    resource_group: str
    hours: float
    cost: float


class ServiceCharge(Payload):
    service: str
    cost: float


class Billed(Payload):
    """What Azure actually charged.

    Nullable as a whole on :class:`Cost`: the billing API is asked independently
    of the task log and is allowed to fail on its own, so the page has to render
    node time with this absent.
    """

    total: float
    other: float
    currency: str
    """Batch pool nodes -- the ONLY figure node time may be compared against."""
    pool_cost: float
    pool_node_hours: float
    """Machines left ON outside the pool. No task log will ever explain these."""
    standing_cost: float
    standing_hours: float
    standing: list[StandingCharge] = []
    since: str
    """The earliest day carrying a charge, which is not the query floor."""
    first_at: str | None
    """The latest day with data. Cost lags hours, so the last day reads low."""
    as_of: str | None
    by_service: list[ServiceCharge] = []


class ConcurrencyPoint(Payload):
    at: str
    running: int


class Cost(Payload):
    op: Literal["cost"]
    hours: float
    task_hours: float
    tasks: int
    """Attempts that started and never recorded an end: unknown, not zero."""
    unended: int
    peak_concurrency: int
    first_at: str | None
    last_at: str | None
    rate_per_node_hour: float | None
    dollars: float | None
    billed: Billed | None
    """Why there is no billed figure. Throttling is not an auth problem."""
    billed_reason: str | None
    series: list[ConcurrencyPoint] = []


"""Experiments, and the comparisons that decide them
--------------------------------------------------"""


class Arm(Payload):
    arm: str
    run_id: str
    checkpoint_iteration: int | None
    exploitability_mbb: float | None
    std_error_mbb: float | None
    git_branch: str | None = None
    vs_control_mbb: float | None
    vs_control_p_value: float | None
    """The field that matters and the one easiest to drop: an arm that could not
    be paired has no `vs_control_mbb`, and rendering that as a dash claims "no
    difference" for a comparison that never happened."""
    vs_control_blocked: list[str] = []


class Report(Payload):
    op: Literal["report"]
    experiment_id: str
    control_run_id: str | None
    baseline_run_id: str | None
    notes: list[str] = []
    arms: list[Arm] = []


class PairedComparison(Payload):
    mean_a: float
    mean_b: float
    mean_diff: float
    se_diff: float
    ci_lower: float
    ci_upper: float
    p_value: float
    is_significant: bool
    correlation: float | None
    se_unpaired: float | None


class Comparison(Payload):
    op: Literal["compare"]
    run_a: str
    run_b: str
    tier_warnings: list[str] = []
    """Null because the command REFUSES rather than guesses: two runs with no
    shared seed have no paired statistic, and `tier_warnings` says why. Both
    halves are needed -- a refusal with no reason is indistinguishable from a
    bug."""
    comparison: PairedComparison | None


"""This tool's own behaviour, and the local reads
-----------------------------------------------"""


class CommandActivity(Payload):
    command: str
    calls: int
    """p50 is what it normally costs, p95 what it costs when it does not. A mean
    would be wrong about both."""
    p50_seconds: float
    p95_seconds: float
    max_seconds: float
    """What says whether a command is worth optimising: 0.4s run 3,000 times
    outranks 9s run twice."""
    total_seconds: float
    refusals: int
    errors: int


class Failure(Payload):
    at: str | None = None
    command: str | None = None
    surface: str | None = None
    outcome: str | None = None
    error_type: str | None = None
    error: str | None = None
    asked: dict[str, Any] = {}


class Activity(Payload):
    """The local activity log, summarised.

    The only payload here that describes THIS TOOL rather than the solver or the
    cloud. `exists`/`enabled` are two different empty states with different fixes
    -- nothing has run yet, versus recording is switched off -- and collapsing
    them would send someone hunting for a bug in the writer.
    """

    op: Literal["activity"]
    log: str
    exists: bool
    enabled: bool
    days: float
    rows: int
    total_rows: int
    commands: list[CommandActivity] = []
    failures: list[Failure] = []
    """Before `--limit` truncates `failures`. The list is a display cap and the
    count is the fact; reporting the capped length said "20 failures" when there
    were 100."""
    total_failures: int = 0
    by_surface: dict[str, int] = {}


class ConfigKind(Payload):
    kind: str
    """The flag these feed, verbatim -- so a picker can say what it sets."""
    flag: str
    names: list[str] = []


class Configs(Payload):
    op: Literal["configs"]
    root: str
    kinds: list[ConfigKind] = []


class Autoscale(Payload):
    """The deployed autoscale formula, evaluated against the live pool.

    `error` is a FIELD, not a failed request: Batch evaluates the formula and
    reports that it did not compute, which is the answer to "why is the pool not
    growing" and must reach the screen rather than blanking the panel.
    """

    op: Literal["autoscale-check"]
    pool_id: str
    variables: list[str] = []
    error: str | None


"""The writes
-----------"""


class Dispatched(Payload):
    """What a dispatch reports back.

    One shape for all three, because `dispatch.stage_and_queue` supplies the same
    three keys to each: the snapshot it sealed, the job it queued into, and the
    tasks it created. The op differs and is deliberately not narrowed -- the page
    already knows which button it pressed, and a literal per command would be
    three near-identical models whose only job is to disagree eventually.
    """

    op: str
    code_snapshot: str
    job_id: str
    tasks: list[str] = []


class VectorArm(Payload):
    abstraction: str
    kernel: str
    derive_boards: int


class DispatchedVector(Dispatched):
    """`submit-vector`, which is a dispatch plus WHICH ARMS it queued.

    It had no schema at all in the console -- not a wrong one, an absent one --
    while its endpoint had been served the whole time. The hand-maintained list
    that was supposed to catch that never named it either. Which abstractions to
    compare IS the experiment, so the arms are the part of this payload worth
    reading back.
    """

    arms: list[VectorArm] = []


class PushedCode(Payload):
    op: Literal["push-code"]
    code_snapshot: str


class PushedData(Payload):
    op: Literal["push-data"]
    """Abstraction name to files uploaded. Empty means everything was current."""
    uploaded: dict[str, int] = {}


class Compacted(Payload):
    """`compact-legs`, whose payload describes BOTH halves of the operation.

    `applied`/`verified`/`deleted` separate a dry run from the irreversible one,
    and the page renders the dry run's numbers as a preview before offering it --
    so one shape carries "what would move" and "what did".
    """

    op: Literal["compact-legs"]
    bundle: str | None
    files_before: int
    files_after: int
    movable: int
    """What an existing bundle at this label already held, and this round carries
    forward rather than replaces. Reporting only `movable` would read as though
    the earlier documents had been dropped -- which is what used to happen before
    a second round absorbed the first."""
    carried: int = 0
    attempts: int | None = None
    applied: bool
    verified: bool
    deleted: int
    backup: str | None


class Promoted(Payload):
    op: Literal["promote"]
    run_id: str
    rationale: str
    promoted_at: str | None
    checkpoint_iteration: int | None


class Cancelled(Payload):
    """What `cancel` reports back. Loose: the UI only needs to know it landed.

    `job_id`/`task_id`, which is what `cancel.run` actually returns. The
    hand-written Zod this replaced declared `job`/`task`, so the console's cancel
    button terminated the task and then threw a parse error reporting it had
    failed -- the worst shape of bug this seam can produce, because the operator
    retries an action that already worked. It survived because the fixture check
    listing was hand-maintained too and simply never named `cancel`.
    """

    op: Literal["cancel"]
    job_id: str
    task_id: str


class Box(Payload):
    """The blueprint host's power state.

    `power` carries transitional values ("starting", "stopping") as well as the
    two stable ones, because a UI that only knew "running" and "deallocated"
    would show "stopped" for the whole two minutes a box takes to wake -- which
    reads as the button having done nothing.
    """

    op: Literal["serve-box"]
    action: str
    vm: str
    resource_group: str
    power: str
    usable: bool
    location: str


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


class RunParts(Payload):
    run: Part[RunInfo]
    progress: Part[Progress]
    curve: Part[Curve]
    evals: Part[Ledger]
    """Answered, then trimmed to `source_rows`: the rows themselves reach the
    client as `run_tasks` below, filtered to this run."""
    tasks: Part[Tasks]


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
