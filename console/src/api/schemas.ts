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

export const legRowSchema = z
  .object({
    task_id: z.string(),
    attempt: z.number().nullable(),
    op: z.string().nullable(),
    run_id: z.string().nullable(),
    cause: z.string().nullable(),
    exit_code: z.number().nullable(),
    ended_at: z.string().nullable(),
  })
  .passthrough();

export const legsSchema = z
  .object({
    op: z.literal("legs"),
    rows: z.array(legRowSchema),
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
    attempts: z.number().nullable(),
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
    legs: z.number(),
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
export type Legs = z.infer<typeof legsSchema>;
export type LegRow = z.infer<typeof legRowSchema>;
export type Runs = z.infer<typeof runsSchema>;
export type RunSummary = z.infer<typeof runSummarySchema>;
export type RunInfo = z.infer<typeof runinfoSchema>;
export type Curve = z.infer<typeof curveSchema>;
export type Ledger = z.infer<typeof ledgerSchema>;
export type Cost = z.infer<typeof costSchema>;
