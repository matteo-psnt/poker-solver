/**
 * One hook per endpoint. Components read these; they never fetch.
 *
 * `refetchInterval` is where the polling debate ended up: cadence is one field
 * per query rather than a scheduler, and `dataUpdatedAt` gives every panel its
 * own age badge for free. A failing query keeps its last good data while
 * exposing `error`, which IS the per-panel isolation the design calls for.
 */
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { get, send } from "./client";
import {
  type BlueprintRun,
  type Box,
  type Cancelled,
  type Combos,
  type Cost,
  type Curve,
  type Hand,
  type Jobs,
  type Ledger,
  type LogLines,
  type Pool,
  type Progress,
  type RunInfo,
  type Runs,
  type SolverNode,
  type Tasks,
  blueprintRunSchema,
  boxSchema,
  cancelSchema,
  combosSchema,
  costSchema,
  curveSchema,
  handSchema,
  jobsSchema,
  ledgerSchema,
  logSchema,
  nodeSchema,
  poolSchema,
  progressSchema,
  runinfoSchema,
  runsSchema,
  tasksSchema,
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

export const useTasks = (limit = 0) =>
  useQuery<Tasks>({
    queryKey: ["tasks", limit],
    queryFn: () => get(`/api/tasks?limit=${limit}`, tasksSchema),
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
    // Derived from the task log, so it costs what `tasks` costs. The billing
    // half rides along free: Cost Management is rate-limited hard and its data
    // lags hours, so `cloud/billing.py` memoises it for 15 minutes server-side
    // and a poll at this interval mostly re-serves that.
    refetchInterval: SLOW,
  });

export const useProgress = (runId: string) =>
  useQuery<Progress>({
    queryKey: ["progress", runId],
    queryFn: () => get(`/api/runs/${encodeURIComponent(runId)}/progress`, progressSchema),
    refetchInterval: SLOW,
  });

/**
 * A task's published log. `enabled` so the query does not fire until a task is
 * actually selected — this is the slowest read in the console and there is no
 * reason to pay for it on a page nobody has opened.
 */
export const useLog = (taskId: string | null, lines = 400) =>
  useQuery<LogLines>({
    queryKey: ["log", taskId, lines],
    queryFn: () => get(`/api/logs/${encodeURIComponent(taskId ?? "")}?lines=${lines}`, logSchema),
    enabled: Boolean(taskId),
    // A published log for a finished task does not change. Refetching it would
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

/**
 * The blueprint server. Not polled: a loaded run does not change under you, so
 * a refetch interval would be pure cost. `combos` is the canonical 1326-entry
 * order and never changes at all, hence `staleTime: Infinity`.
 */
export const useBlueprintRun = () =>
  useQuery<BlueprintRun>({
    queryKey: ["blueprint", "run"],
    queryFn: () => get("/api/blueprint/run", blueprintRunSchema),
    staleTime: Number.POSITIVE_INFINITY,
  });

export const useCombos = (enabled: boolean) =>
  useQuery<Combos>({
    queryKey: ["blueprint", "combos"],
    queryFn: () => get("/api/blueprint/combos", combosSchema),
    staleTime: Number.POSITIVE_INFINITY,
    enabled,
  });

export const useSolverNode = (path: string, board: string, average: boolean, enabled: boolean) =>
  useQuery<SolverNode>({
    queryKey: ["blueprint", "node", path, board, average],
    queryFn: () =>
      get(
        `/api/blueprint/node?path=${encodeURIComponent(path)}&board=${encodeURIComponent(board)}&average=${average}`,
        nodeSchema,
      ),
    enabled,
  });

/**
 * Play. Mutations rather than queries: a hand advances because you acted, so
 * there is nothing to poll and a refetch would replay a move.
 */
export const useDealHand = () =>
  useMutation<Hand, Error, { human_seat: number; seed?: number | null }>({
    mutationFn: (body) => send("/api/blueprint/play", handSchema, body),
  });

export const useSubmitAction = () =>
  useMutation<Hand, Error, { session: string; token: string }>({
    mutationFn: ({ session, token }) =>
      send(`/api/blueprint/play/${session}/action`, handSchema, { token }),
  });

/**
 * The host's power state. Polled FAST only while it is mid-transition: a box
 * that is settled will not change without someone clicking, and a box that is
 * waking needs to be watched or the page lies for two minutes.
 */
export const useBox = () =>
  useQuery<Box>({
    queryKey: ["blueprint", "box"],
    queryFn: () => get("/api/box", boxSchema),
    refetchInterval: (query) => {
      const power = query.state.data?.power;
      return power === "running" || power === "deallocated" ? 30_000 : 5_000;
    },
  });

export const useBoxAction = () =>
  useMutation<Box, Error, "start" | "stop">({
    mutationFn: (action) => send(`/api/box/${action}`, boxSchema),
  });

/**
 * Cancel one task. Invalidates the task list rather than patching it: the
 * authority on whether a task is still running is Batch, not this click.
 */
export const useCancelTask = () => {
  const queryClient = useQueryClient();
  return useMutation<Cancelled, Error, { job: string; task: string }>({
    mutationFn: ({ job, task }) =>
      send(
        `/api/tasks/${encodeURIComponent(job)}/${encodeURIComponent(task)}/cancel`,
        cancelSchema,
      ),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["tasks"] }),
  });
};
