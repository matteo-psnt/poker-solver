/**
 * The payload contract, as runtime-checked schemas.
 *
 * Python emits dicts and TypeScript expects shapes; nothing else connects them.
 * Parsing at the fetch boundary means a payload change fails HERE, loudly, with
 * a readable path — instead of surfacing as `undefined` three components deep.
 *
 * Types are inferred (`z.infer`), so this is ONE declaration rather than two.
 * The reference when writing a schema is `PAYLOADS` in
 * `tests/interfaces/cli/test_command_renderers.py`: the shapes are already
 * pinned there by a passing Python test.
 *
 * Schemas are deliberately loose about fields the UI does not read
 * (`.passthrough()`): a Python payload gaining a key is not a reason to break
 * the console.
 */
import { z } from "zod";

export const poolSchema = z
  .object({
    op: z.literal("pool-status"),
    pool_id: z.string(),
    allocation_state: z.string().nullable(),
    current_dedicated_nodes: z.number().nullable(),
    target_dedicated_nodes: z.number().nullable(),
    vm_size: z.string().nullable(),
    hourly_cost: z.string().optional(),
    resize_errors: z
      .array(
        z.object({
          code: z.string().nullable(),
          message: z.string().nullable(),
          values: z.record(z.string().nullable()),
        }),
      )
      .default([]),
  })
  .passthrough();

const taskSchema = z
  .object({
    task: z.string(),
    state: z.string().nullable(),
    exit_code: z.number().nullable(),
    node: z.string().nullable().optional(),
    /** Orders the queue: a task that has not started has no start_time. */
    created: z.string().nullable().optional(),
    start_time: z.string().nullable().optional(),
    end_time: z.string().nullable().optional(),
  })
  .passthrough();

export const jobsSchema = z
  .object({
    op: z.literal("jobs"),
    jobs: z.array(
      z.object({
        job: z.string(),
        state: z.string().nullable(),
        tasks: z.array(taskSchema),
      }),
    ),
    total_jobs: z.number(),
    hidden_jobs: z.number(),
  })
  .passthrough();

export const taskRowSchema = z
  .object({
    task_id: z.string(),
    attempt: z.number().nullable(),
    op: z.string().nullable(),
    run_id: z.string().nullable(),
    cause: z.string().nullable(),
    exit_code: z.number().nullable(),
    ended_at: z.string().nullable(),
    started_at: z.string().nullable().optional(),
    job_id: z.string().nullable().optional(),
    node_id: z.string().nullable().optional(),
    /** Derived server-side so the terminal and the console word it identically. */
    what: z.string().nullable().optional(),
    /** Seconds left, derived server-side so both surfaces agree. */
    eta_seconds: z.number().nullable().optional(),
    /** What a FINISHED task achieved, and how wide it ran — the two things an
        estimate for the next one is built from. */
    units: z.number().nullable().optional(),
    workers: z.number().nullable().optional(),
    /** WHICH CODE RAN. Three answers that do not replace one another: the
        commit names a history, the branch names the line of work (several
        worktrees share a commit and differ only in what is uncommitted), and
        the snapshot names the exact bytes. Absent on every task recorded before
        this existed. */
    code_snapshot: z.string().nullable().optional(),
    git_commit: z.string().nullable().optional(),
    git_dirty: z.string().nullable().optional(),
    git_branch: z.string().nullable().optional(),
    /** Present only while a task is running and only if its kind can say. */
    progress: z
      .object({ done: z.number(), total: z.number(), unit: z.string() })
      .nullable()
      .optional(),
  })
  .passthrough();

export const tasksSchema = z
  .object({
    op: z.literal("tasks"),
    rows: z.array(taskRowSchema),
    reconciled: z.number().nullable(),
    hidden_rows: z.number().optional(),
  })
  .passthrough();

export const runSummarySchema = z
  .object({
    name: z.string(),
    commits_ago: z.number().nullable(),
    git_dirty: z.boolean().nullable(),
    loadable: z.boolean(),
    blocker: z.string().nullable(),
    iterations: z.number().nullable(),
    num_infosets: z.number().nullable(),
    config_name: z.string().nullable(),
    status: z.string().nullable(),
    /** Lineage. The run listing is the only place the SET of experiments
        exists — `report` takes an id and `runinfo` reports one run's — so the
        experiment picker is built from these. Absent on every legacy run. */
    experiment_id: z.string().nullable().optional(),
    arm: z.string().nullable().optional(),
  })
  .passthrough();

export const runsSchema = z
  .object({ op: z.literal("runs"), runs: z.array(runSummarySchema) })
  .passthrough();

export const runinfoSchema = z
  .object({
    op: z.literal("runinfo"),
    run_id: z.string(),
    config_name: z.string().nullable(),
    status: z.string().nullable(),
    iterations: z.number().nullable(),
    runtime_seconds: z.number().nullable(),
    /** The run's own count, which legitimately differs from what the task log observed. */
    training_tasks: z.number().nullable(),
    git_commit: z.string().nullable(),
    card_abstraction_hash: z.string().nullable(),
    progress: z
      .array(
        z
          .object({
            iteration: z.number(),
            coverage: z.number().nullable(),
            mean_visits_per_touched: z.number().nullable(),
            iters_per_sec: z.number().nullable(),
          })
          .passthrough(),
      )
      .default([]),
  })
  .passthrough();

export const curveSchema = z
  .object({
    op: z.literal("curve"),
    run_id: z.string(),
    tier: z.string().nullable(),
    points: z.array(
      z
        .object({
          iteration: z.number(),
          exploitability_mbb: z.number().nullable(),
          std_error_mbb: z.number().nullable(),
        })
        .passthrough(),
    ),
    missing_iterations: z.array(z.number()).default([]),
  })
  .passthrough();

export const progressSchema = z
  .object({
    op: z.literal("progress"),
    run_id: z.string(),
    total_rows: z.number(),
    coverage_plateau_iteration: z.number().nullable(),
    rows: z.array(
      z
        .object({
          iteration: z.number(),
          coverage: z.number().nullable(),
          // THE convergence diagnostic. Compare against the 1e3-1e4 CFR needs
          // before a regret average carries signal; coverage saturates early
          // and says nothing about it.
          mean_visits_per_touched: z.number().nullable().optional(),
          iters_per_sec: z.number().nullable().optional(),
        })
        .passthrough(),
    ),
  })
  .passthrough();

export const logSchema = z
  .object({
    op: z.literal("logs"),
    task: z.string().nullable(),
    lines: z.array(z.string()).nullable(),
  })
  .passthrough();

export const ledgerSchema = z
  .object({
    op: z.literal("ledger"),
    ledger: z.string(),
    rows: z.array(
      z
        .object({
          run_id: z.string().optional(),
          eval_git_commit: z.string().nullable().optional(),
          knobs: z.record(z.unknown()).default({}),
          results: z.record(z.unknown()).default({}),
        })
        .passthrough(),
    ),
  })
  .passthrough();

/**
 * What Azure actually charged. Nullable as a whole: the billing API is asked
 * independently of the task log and is allowed to fail on its own, so the page
 * has to render node time with this absent.
 */
export const billedSchema = z
  .object({
    total: z.number(),
    other: z.number(),
    currency: z.string(),
    /** Batch pool nodes — the ONLY figure node time may be compared against. */
    pool_cost: z.number(),
    pool_node_hours: z.number(),
    /** Machines left ON outside the pool. No task log will ever explain these. */
    standing_cost: z.number(),
    standing_hours: z.number(),
    standing: z.array(
      z.object({ resource_group: z.string(), hours: z.number(), cost: z.number() }),
    ),
    since: z.string(),
    /** The earliest day carrying a charge, which is not the query floor. */
    first_at: z.string().nullable(),
    /** The latest day with data. Cost lags hours, so the last day reads low. */
    as_of: z.string().nullable(),
    by_service: z.array(z.object({ service: z.string(), cost: z.number() })),
  })
  .passthrough();

export const costSchema = z
  .object({
    op: z.literal("cost"),
    hours: z.number(),
    task_hours: z.number(),
    tasks: z.number(),
    /** Attempts that started and never recorded an end: unknown, not zero. */
    unended: z.number(),
    peak_concurrency: z.number(),
    first_at: z.string().nullable(),
    last_at: z.string().nullable(),
    rate_per_node_hour: z.number().nullable(),
    dollars: z.number().nullable(),
    billed: billedSchema.nullable(),
    /** Why there is no billed figure. Throttling is not an auth problem. */
    billed_reason: z.string().nullable(),
    series: z.array(z.object({ at: z.string(), running: z.number() })),
  })
  .passthrough();

export type Pool = z.infer<typeof poolSchema>;
export type Jobs = z.infer<typeof jobsSchema>;
export type Tasks = z.infer<typeof tasksSchema>;
export type TaskRow = z.infer<typeof taskRowSchema>;
export type Runs = z.infer<typeof runsSchema>;
export type RunSummary = z.infer<typeof runSummarySchema>;
export type RunInfo = z.infer<typeof runinfoSchema>;
export type Curve = z.infer<typeof curveSchema>;
export type Ledger = z.infer<typeof ledgerSchema>;
export type Cost = z.infer<typeof costSchema>;
export type Billed = z.infer<typeof billedSchema>;
export type Progress = z.infer<typeof progressSchema>;
export type LogLines = z.infer<typeof logSchema>;

/**
 * The blueprint server's shapes, proxied through `/api/blueprint/*`.
 *
 * Unlike every other schema here these do NOT correspond to a command: the
 * blueprint server is a separate process holding a loaded run, and the console
 * reaches it over HTTP precisely so it never imports the engine. Its refusals
 * arrive as the same `{error}` body, so `client.get` needs no special case.
 */
export const blueprintRunSchema = z
  .object({
    run: z.string(),
    starting_stack: z.number(),
    small_blind: z.number(),
    big_blind: z.number(),
    combos: z.number(),
  })
  .passthrough();

export const combosSchema = z.object({ combos: z.array(z.string()) }).passthrough();

/**
 * `strategy` is null exactly when `trained` is false — the server refuses to
 * emit the uniform an allocated-but-unvisited row would otherwise read as, and
 * the schema keeps that distinction rather than defaulting it away.
 */
const bucketSchema = z
  .object({
    trained: z.boolean(),
    strategy: z.array(z.number()).nullable(),
    reach_count: z.number(),
  })
  .passthrough();

export const nodeSchema = z
  .object({
    path: z.string(),
    terminal: z.boolean(),
    board: z.array(z.string()),
    grid: z
      .object({
        street: z.string(),
        board: z.array(z.string()),
        actor: z.number(),
        actions: z.array(z.string()),
        // -1 where the board blocks the combo. Kept in place rather than
        // filtered, so index i is always ALL_COMBOS[i].
        combo_buckets: z.array(z.number()),
        blocked: z.number(),
        trained_buckets: z.number(),
        buckets: z.record(bucketSchema),
      })
      .passthrough()
      .nullable(),
    children: z.array(
      z.object({ token: z.string(), type: z.string(), amount: z.number() }).passthrough(),
    ),
  })
  .passthrough();

export type BlueprintRun = z.infer<typeof blueprintRunSchema>;
export type Combos = z.infer<typeof combosSchema>;
export type SolverNode = z.infer<typeof nodeSchema>;
export type NodeGrid = NonNullable<SolverNode["grid"]>;

/**
 * A hand in progress.
 *
 * `bot_hole_cards` and every `mix` are null until `over` — the server withholds
 * them, and the schema says so, because a client that received them could show
 * them and a sit-down where you see the opponent's hand measures nothing.
 */
const handEventSchema = z
  .object({
    seat: z.number(),
    actor: z.string(),
    action: z.string(),
    amount: z.number(),
    street: z.string(),
    untrained: z.boolean(),
    mix: z.array(z.tuple([z.string(), z.number()])).nullable(),
  })
  .passthrough();

export const handSchema = z
  .object({
    session: z.string(),
    over: z.boolean(),
    street: z.string(),
    board: z.array(z.string()),
    pot: z.number(),
    stacks: z.array(z.number()),
    human_seat: z.number(),
    button: z.number(),
    to_act: z.number().nullable(),
    hole_cards: z.array(z.string()),
    bot_hole_cards: z.array(z.string()).nullable(),
    legal: z.array(
      z.object({ token: z.string(), type: z.string(), amount: z.number() }).passthrough(),
    ),
    payoff: z.number().nullable(),
    showdown: z.boolean(),
    bot_decisions: z.number(),
    bot_untrained_decisions: z.number(),
    log: z.array(handEventSchema),
  })
  .passthrough();

export type Hand = z.infer<typeof handSchema>;
export type HandEvent = z.infer<typeof handEventSchema>;

/**
 * The blueprint host's power state.
 *
 * `power` carries transitional values ("starting", "stopping") as well as the
 * two stable ones, because a UI that only knew "running" and "deallocated"
 * would show "stopped" for the whole two minutes a box takes to wake — which
 * reads as the button having done nothing.
 */
export const boxSchema = z
  .object({
    op: z.literal("serve-box"),
    action: z.string(),
    vm: z.string(),
    resource_group: z.string(),
    power: z.string(),
    usable: z.boolean(),
    location: z.string(),
  })
  .passthrough();

export type Box = z.infer<typeof boxSchema>;

/** What `cancel` reports back. Loose: the UI only needs to know it landed. */
export const cancelSchema = z
  .object({
    op: z.literal("cancel"),
    job: z.string(),
    task: z.string(),
  })
  .passthrough();

export type Cancelled = z.infer<typeof cancelSchema>;

/**
 * The commands the console gained when coverage became a maintained property.
 *
 * `tests/interfaces/web/test_command_coverage.py` is the other half: it fails if
 * a command has neither an endpoint nor a written-down reason to lack one. These
 * schemas are what makes an endpoint usable rather than merely present.
 */

export const configsSchema = z
  .object({
    op: z.literal("configs"),
    root: z.string(),
    kinds: z.array(
      z
        .object({
          kind: z.string(),
          /** The flag these feed, verbatim — so a picker can say what it sets. */
          flag: z.string(),
          names: z.array(z.string()),
        })
        .passthrough(),
    ),
  })
  .passthrough();

/**
 * The deployed autoscale formula, evaluated against the live pool.
 *
 * `error` is a FIELD, not a failed request: Batch evaluates the formula and
 * reports that it did not compute, which is the answer to "why is the pool not
 * growing" and must reach the screen rather than blanking the panel.
 */
export const autoscaleSchema = z
  .object({
    op: z.literal("autoscale-check"),
    pool_id: z.string(),
    variables: z.array(z.string()).default([]),
    error: z.string().nullable(),
  })
  .passthrough();

/**
 * One experiment, every arm attributed against its control.
 *
 * `vs_control_blocked` is the field that matters and the one easiest to drop: an
 * arm that could not be paired has no `vs_control_mbb`, and rendering that as a
 * dash claims "no difference" for a comparison that never happened.
 */
export const reportSchema = z
  .object({
    op: z.literal("report"),
    experiment_id: z.string(),
    control_run_id: z.string().nullable(),
    baseline_run_id: z.string().nullable(),
    notes: z.array(z.string()).default([]),
    arms: z.array(
      z
        .object({
          arm: z.string(),
          run_id: z.string(),
          checkpoint_iteration: z.number().nullable(),
          exploitability_mbb: z.number().nullable(),
          std_error_mbb: z.number().nullable(),
          git_branch: z.string().nullable().optional(),
          vs_control_mbb: z.number().nullable(),
          vs_control_p_value: z.number().nullable(),
          vs_control_blocked: z.array(z.string()).default([]),
        })
        .passthrough(),
    ),
  })
  .passthrough();

/**
 * A paired comparison of two runs.
 *
 * `comparison` is nullable because the command refuses rather than guesses: two
 * runs with no shared seed have no paired statistic, and `tier_warnings` says
 * why. Both halves are needed — a refusal with no reason is indistinguishable
 * from a bug.
 */
export const compareSchema = z
  .object({
    op: z.literal("compare"),
    run_a: z.string(),
    run_b: z.string(),
    tier_warnings: z.array(z.string()).default([]),
    comparison: z
      .object({
        mean_a: z.number(),
        mean_b: z.number(),
        mean_diff: z.number(),
        se_diff: z.number(),
        ci_lower: z.number(),
        ci_upper: z.number(),
        p_value: z.number(),
        is_significant: z.boolean(),
        correlation: z.number().nullable(),
        se_unpaired: z.number().nullable(),
      })
      .passthrough()
      .nullable(),
  })
  .passthrough();

/**
 * What a dispatch reports back. One shape for all three, because
 * `dispatch.stage_and_queue` supplies the same three keys to each: the snapshot
 * it sealed, the job it queued into, and the tasks it created.
 *
 * The op differs, so it is not narrowed here — the page already knows which
 * button it pressed, and a literal per command would be three near-identical
 * schemas whose only job is to disagree with each other eventually.
 */
export const dispatchedSchema = z
  .object({
    op: z.string(),
    code_snapshot: z.string(),
    job_id: z.string(),
    tasks: z.array(z.string()).default([]),
  })
  .passthrough();

export const pushCodeSchema = z
  .object({ op: z.literal("push-code"), code_snapshot: z.string() })
  .passthrough();

export const pushDataSchema = z
  .object({
    op: z.literal("push-data"),
    /** abstraction name → files uploaded. Empty means everything was current. */
    uploaded: z.record(z.number()).default({}),
  })
  .passthrough();

/**
 * `compact-legs`, whose payload describes BOTH halves of the operation.
 *
 * `applied`/`verified`/`deleted` are what separate a dry run from the
 * irreversible one, and the page renders the dry run's numbers as a preview
 * before offering it — so the same shape has to carry "what would move" and
 * "what did".
 */
export const compactSchema = z
  .object({
    op: z.literal("compact-legs"),
    bundle: z.string().nullable(),
    files_before: z.number(),
    files_after: z.number(),
    movable: z.number(),
    attempts: z.number().optional(),
    applied: z.boolean(),
    verified: z.boolean(),
    deleted: z.number(),
    backup: z.string().nullable(),
  })
  .passthrough();

export const promoteSchema = z
  .object({
    op: z.literal("promote"),
    run_id: z.string(),
    rationale: z.string(),
    promoted_at: z.string().nullable(),
    checkpoint_iteration: z.number().nullable(),
  })
  .passthrough();

export type Configs = z.infer<typeof configsSchema>;
export type Autoscale = z.infer<typeof autoscaleSchema>;
export type Report = z.infer<typeof reportSchema>;
export type Comparison = z.infer<typeof compareSchema>;
export type Dispatched = z.infer<typeof dispatchedSchema>;
export type PushedCode = z.infer<typeof pushCodeSchema>;
export type PushedData = z.infer<typeof pushDataSchema>;
export type Compacted = z.infer<typeof compactSchema>;
export type Promoted = z.infer<typeof promoteSchema>;
