/**
 * The payload types, named. Every one of them is GENERATED.
 *
 * This file is the only hand-written thing in the contract chain, and it
 * declares no shapes -- it gives the generated types the names the UI uses, so
 * a component imports `Pool` rather than
 * `components["schemas"]["Pool"]`. Adding a field to a payload does not touch
 * this file; adding a whole new payload adds one line.
 *
 * What this replaced
 * ------------------
 * `schemas.ts`: 682 lines of hand-written Zod, kept in sync with Python by
 * reading a fixture generated from a Python test. It opened by claiming it was
 * "ONE declaration rather than two" and was in fact the second declaration of
 * every payload in the system. It also disagreed with the server about `cancel`
 * -- it required `job`/`task` where the command returns `job_id`/`task_id` -- so
 * the console cancelled the task and then reported that it had failed. That went
 * unnoticed because the list of schemas checked against the fixture was
 * hand-maintained too, and never named `cancel`.
 *
 * The chain now: `contract.py` declares → FastAPI exports `openapi.json` (a
 * Python test fails if it is stale) → `openapi-typescript` generates
 * `types.gen.ts` (`npm run build` regenerates it every time) → this names them.
 */
import type { components } from "./types.gen";

type S = components["schemas"];

/** The pool, and what Batch is running. */
export type Pool = S["Pool"];
export type Jobs = S["Jobs"];
export type Job = S["Job"];
export type BatchTask = S["BatchTask"];

/** The durable task account -- the only thing that can say why a task DIED. */
export type Tasks = S["Tasks"];
export type TaskRow = S["TaskRow"];

/** The record. */
export type Runs = S["Runs"];
export type RunSummary = S["RunSummary"];
export type RunInfo = S["RunInfo"];
export type Progress = S["Progress"];
export type ProgressRow = S["ProgressRow"];
export type Curve = S["Curve"];
export type Ledger = S["Ledger"];
export type LogLines = S["LogLines"];
export type Cost = S["Cost"];
export type Billed = S["Billed"];

/** Experiments and comparisons. */
export type Report = S["Report"];
export type Arm = S["Arm"];
export type Comparison = S["Comparison"];

/** This tool's own behaviour, and the local reads. */
export type Activity = S["Activity"];
export type Configs = S["Configs"];
export type Autoscale = S["Autoscale"];

/** The writes. */
export type Dispatched = S["Dispatched"];
export type DispatchedVector = S["DispatchedVector"];
export type PushedCode = S["PushedCode"];
export type PushedData = S["PushedData"];
export type Compacted = S["Compacted"];
export type Promoted = S["Promoted"];
export type Cancelled = S["Cancelled"];
export type Box = S["Box"];

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
export type RunView = S["RunView"];
export type ExperimentView = S["ExperimentView"];

/**
 * One part of a view. Generic so a page can say what it is holding.
 *
 * Exactly one of `payload`/`error` is set. Written as the generated Part shape
 * with the payload narrowed, because openapi-typescript emits one concrete
 * `Part_*_` per instantiation and naming them individually would be a list to
 * keep in step with the Python.
 */
export type Part<T> = { payload?: T | null; error?: string | null };
