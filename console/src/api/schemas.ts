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
      z.object({ job: z.string(), state: z.string().nullable(), tasks: z.array(taskSchema) }),
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

export const costSchema = z
  .object({
    op: z.literal("cost"),
    hours: z.number(),
    task_hours: z.number(),
    tasks: z.number(),
    peak_concurrency: z.number(),
    first_at: z.string().nullable(),
    last_at: z.string().nullable(),
    rate_per_node_hour: z.number().nullable(),
    dollars: z.number().nullable(),
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
export type Progress = z.infer<typeof progressSchema>;
export type LogLines = z.infer<typeof logSchema>;
