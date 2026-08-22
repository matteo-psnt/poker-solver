/**
 * The payload types, named. Every one of them is GENERATED.
 *
 * The only hand-written file in the contract chain, and it declares no shapes --
 * it gives the generated types the names the UI uses, so a component imports
 * `Pool` rather than `components["schemas"]["Pool"]`. Adding a field to a payload
 * does not touch this file; adding a whole new payload adds one line.
 *
 * The chain: `contract.py` declares -> FastAPI exports `openapi.json` (a Python
 * test fails if it is stale) -> `openapi-typescript` generates `types.gen.ts`
 * (`npm run build` regenerates it every time) -> this names them. Never hand-write
 * a schema here.
 */
import type { components } from "./types.gen";

type S = components["schemas"];

/** The pool, and what Batch is running. */
export type Pool = S["PoolPayload"];
export type NodeStatus = S["NodeStatus"];
export type NodePhase = S["NodePhase"];
export type AutoscaleRun = S["AutoscaleRun"];
export type Jobs = S["JobsPayload"];
export type Job = S["Job"];
export type BatchTask = S["BatchTask"];

/**
 * What a Batch task's state MEANS, decided server-side.
 *
 * `phase` and `outcome` replaced an Azure-semantics module in this console:
 * `shortState`, `taskOutcome` and `exitMeaning` all existed because `/api/jobs`
 * shipped `"BatchTaskState.ACTIVE"` while `/api/tasks` shipped `"active"`. The
 * words are `src/shared/task_states.py`'s and arrive already classified.
 */
export type Phase = S["Phase"];
export type Outcome = S["Outcome"];

/** The durable task account -- the only thing that can say why a task DIED. */
export type Tasks = S["TasksPayload"];
export type TaskRow = S["TaskRow"];

/**
 * A `tasks` part that a view fetched only to JOIN, with its rows removed.
 *
 * A separate type from `Tasks`, and it has no `rows` field — which is the
 * point. One model used to describe both, so `parts.tasks.payload.rows` was `[]`
 * on a trimmed part and correct about nothing; a page only avoided that by
 * remembering to read the join (`run_tasks`) instead. Now there is nothing to
 * remember.
 */
export type TasksSummary = S["TasksSummary"];

/** The record. */
export type Runs = S["RunsPayload"];
export type RunSummary = S["RunSummary"];
export type RunInfo = S["RunInfoPayload"];
export type Progress = S["ProgressPayload"];
/**
 * A row of `progress.jsonl`, which is a RECORD rather than a modelled payload:
 * a resumed run appends across code versions, so the fields present vary by
 * when the row was written and the server does not pretend otherwise.
 */
export type ProgressRow = Record<string, unknown>;
export type Curve = S["CurvePayload"];
export type Ledger = S["LedgerPayload"];
export type LogLines = S["LogsPayload"];
export type Cost = S["CostPayload"];
export type Billed = S["BilledPayload"];

/** This tool's own behaviour, and the local reads. */
export type Activity = S["ActivityPayload"];
export type Configs = S["ConfigsPayload"];
export type Autoscale = S["AutoscalePayload"];

/** The writes. */
/**
 * What a dispatch reports back. A UNION, because each command declares its own:
 * `score` carries the rungs it covered, `submit-precompute` the name it will
 * publish as. They shared one shape while `contract.py` restated it, and the
 * specific halves were invisible in the schema.
 */
export type Dispatched = S["SubmitPayload"] | S["ScorePayload"] | S["PrecomputeDispatchPayload"];
export type DispatchedVector = S["SubmitVectorPayload"];
export type PushedCode = S["PushedCodePayload"];
export type PushedData = S["PushedDataPayload"];
export type Compacted = S["CompactedPayload"];
export type Cancelled = S["CancelledPayload"];
export type Box = S["BoxPayload"];

/** The blueprint server's shapes, proxied through `/api/blueprint/*`. */
export type BlueprintRun = S["BlueprintRun"];
export type BlueprintLoad = S["BlueprintLoad"];
export type Combos = S["Combos"];
export type SolverNode = S["SolverNode"];
export type NodeGrid = S["NodeGrid"];
export type Bucket = S["Bucket"];
export type Hand = S["Hand"];
export type HandEvent = S["HandEvent"];
export type LeftSession = S["LeftSession"];

/**
 * The composed views: one screen, one request.
 *
 * `parts` carries each command's payload OR the reason there is not one, which
 * is what lets a page grey a single panel and keep the rest. The joins
 * (`run_tasks`, `arm_runs`) are cross-references the server drew between parts
 * that have already been answered -- the browser used to draw them itself, after
 * downloading the whole task log to do it.
 */
export type NowView = S["NowView"];
export type RunsView = S["RunsView"];
export type RunView = S["RunView"];

/**
 * One part of a view. Generic so a page can say what it is holding.
 *
 * Exactly one of `payload`/`error` is set. Written as the generated Part shape
 * with the payload narrowed, because openapi-typescript emits one concrete
 * `Part_*_` per instantiation and naming them individually would be a list to
 * keep in step with the Python.
 */
export type Part<T> = { payload?: T | null; error?: string | null };
