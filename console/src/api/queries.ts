/**
 * One hook per endpoint. Components read these; they never fetch.
 *
 * `refetchInterval` is where the polling debate ended up: cadence is one field
 * per query rather than a scheduler, and `dataUpdatedAt` gives every panel its
 * own age badge for free. A failing query keeps its last good data while
 * exposing `error`, which IS the per-panel isolation the design calls for.
 */
import { useQuery } from "@tanstack/react-query";
import { get } from "./client";
import {
  type Cost,
  type Curve,
  type Jobs,
  type Ledger,
  type Legs,
  type LogLines,
  type Pool,
  type Progress,
  type RunInfo,
  type Runs,
  costSchema,
  curveSchema,
  jobsSchema,
  ledgerSchema,
  legsSchema,
  logSchema,
  poolSchema,
  progressSchema,
  runinfoSchema,
  runsSchema,
} from "./schemas";

/** Cheap reads (~2s) can be frequent; the share reads (~5s) should not be. */
const FAST = 15_000;
const SLOW = 60_000;

export const usePool = () =>
  useQuery<Pool>({
    queryKey: ["pool"],
    queryFn: () => get("/api/pool", poolSchema),
    refetchInterval: FAST,
  });

export const useJobs = (limit = 20) =>
  useQuery<Jobs>({
    queryKey: ["jobs", limit],
    queryFn: () => get(`/api/jobs?limit=${limit}`, jobsSchema),
    refetchInterval: FAST,
  });

export const useLegs = (limit = 0) =>
  useQuery<Legs>({
    queryKey: ["legs", limit],
    queryFn: () => get(`/api/legs?limit=${limit}`, legsSchema),
    refetchInterval: SLOW,
  });

export const useRuns = () =>
  useQuery<Runs>({
    queryKey: ["runs"],
    queryFn: () => get("/api/runs", runsSchema),
    refetchInterval: SLOW,
  });

export const useRun = (runId: string) =>
  useQuery<RunInfo>({
    queryKey: ["run", runId],
    queryFn: () => get(`/api/runs/${encodeURIComponent(runId)}`, runinfoSchema),
    refetchInterval: SLOW,
  });

export const useCurve = (runId: string) =>
  useQuery<Curve>({
    queryKey: ["curve", runId],
    queryFn: () => get(`/api/runs/${encodeURIComponent(runId)}/curve`, curveSchema),
    refetchInterval: SLOW,
  });

export const useCost = (hours = 0) =>
  useQuery<Cost>({
    queryKey: ["cost", hours],
    queryFn: () => get(`/api/cost?hours=${hours}`, costSchema),
    // Derived from the leg log, so it costs what `legs` costs.
    refetchInterval: SLOW,
  });

export const useProgress = (runId: string) =>
  useQuery<Progress>({
    queryKey: ["progress", runId],
    queryFn: () => get(`/api/runs/${encodeURIComponent(runId)}/progress`, progressSchema),
    refetchInterval: SLOW,
  });

/**
 * A leg's published log. `enabled` so the query does not fire until a task is
 * actually selected — this is the slowest read in the console and there is no
 * reason to pay for it on a page nobody has opened.
 */
export const useLog = (taskId: string | null, lines = 400) =>
  useQuery<LogLines>({
    queryKey: ["log", taskId, lines],
    queryFn: () => get(`/api/logs/${encodeURIComponent(taskId ?? "")}?lines=${lines}`, logSchema),
    enabled: Boolean(taskId),
    // A published log for a finished leg does not change. Refetching it would
    // be a cloud read for an answer that cannot have moved.
    refetchInterval: false,
    staleTime: 5 * 60_000,
  });

export const useEvals = (limit = 50) =>
  useQuery<Ledger>({
    queryKey: ["evals", limit],
    queryFn: () => get(`/api/evals?limit=${limit}`, ledgerSchema),
    refetchInterval: SLOW,
  });
