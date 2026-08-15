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
import type {
  Activity,
  BlueprintLoad,
  BlueprintRun,
  Box,
  Cancelled,
  Combos,
  Compacted,
  Comparison,
  Configs,
  Cost,
  Dispatched,
  ExperimentView,
  Hand,
  Jobs,
  LeftSession,
  LogLines,
  NowView,
  Pool,
  Promoted,
  PushedCode,
  PushedData,
  RunView,
  Runs,
  RunsView,
  SolverNode,
  Tasks,
} from "./types";

/** Cheap reads (~2s) can be frequent; the share reads (~5s) should not be. */
const FAST = 15_000;
const SLOW = 60_000;

/**
 * The composed views: one screen, one request.
 *
 * These are what a PAGE should use. The per-command hooks below stay for the
 * ad-hoc questions — a picker that needs only `configs`, a log nobody has opened
 * — but a screen assembled out of four of them pays four round trips and gets
 * four different ages, which is what made the Overview feel stale while every
 * individual panel was behaving correctly.
 *
 * One cadence per view rather than one per panel, and that is the point: every
 * number on the page is now the same age as every other, and the age badge is
 * telling the truth about all of them at once.
 */
export const useNow = () =>
  useQuery<NowView>({
    queryKey: ["view", "now"],
    queryFn: () => get("/api/view/now"),
    refetchInterval: FAST,
  });

export const useRunsView = () =>
  useQuery<RunsView>({
    queryKey: ["view", "runs"],
    queryFn: () => get("/api/view/runs"),
    refetchInterval: SLOW,
  });

export const useRunView = (runId: string) =>
  useQuery<RunView>({
    queryKey: ["view", "run", runId],
    queryFn: () => get(`/api/view/run/${encodeURIComponent(runId)}`),
    refetchInterval: SLOW,
  });

/**
 * `enabled` on the id: the page opens with no experiment selected, and a request
 * for `""` would be a share read that can only refuse.
 */
export const useExperimentView = (experimentId: string | null) =>
  useQuery<ExperimentView>({
    queryKey: ["view", "experiment", experimentId],
    queryFn: () => get(`/api/view/experiment/${encodeURIComponent(experimentId ?? "")}`),
    enabled: Boolean(experimentId),
    refetchInterval: SLOW,
  });

export const usePool = () =>
  useQuery<Pool>({
    queryKey: ["pool"],
    queryFn: () => get("/api/pool"),
    refetchInterval: FAST,
  });

export const useJobs = (limit = 20) =>
  useQuery<Jobs>({
    queryKey: ["jobs", limit],
    queryFn: () => get(`/api/jobs?limit=${limit}`),
    refetchInterval: FAST,
  });

export const useTasks = (limit = 0) =>
  useQuery<Tasks>({
    queryKey: ["tasks", limit],
    queryFn: () => get(`/api/tasks?limit=${limit}`),
    refetchInterval: SLOW,
  });

export const useRuns = () =>
  useQuery<Runs>({
    queryKey: ["runs"],
    queryFn: () => get("/api/runs"),
    refetchInterval: SLOW,
  });

export const useCost = (hours = 0) =>
  useQuery<Cost>({
    queryKey: ["cost", hours],
    queryFn: () => get(`/api/cost?hours=${hours}`),
    // Derived from the task log, so it costs what `tasks` costs. The billing
    // half rides along free: Cost Management is rate-limited hard and its data
    // lags hours, so `cloud/billing.py` memoises it for 15 minutes server-side
    // and a poll at this interval mostly re-serves that.
    refetchInterval: SLOW,
  });

/**
 * A task's published log. `enabled` so the query does not fire until a task is
 * actually selected — this is the slowest read in the console and there is no
 * reason to pay for it on a page nobody has opened.
 *
 * `live` is the caller's answer to "is this task still going", decided from the
 * row it already has (no `ended_at`) — the client owns that judgement, the same
 * way it owns which Batch states mean running.
 *
 * It has to be asked, because this used to be `refetchInterval: false` for
 * every task on the grounds that "a published log for a finished task does not
 * change". True of a finished one, and the whole problem for a running one: the
 * node republishes the log every 60s and the console never asked again, so an
 * open log was frozen at whatever it held when the page mounted.
 */
export const useLog = (taskId: string | null, lines = 400, live = false) =>
  useQuery<LogLines>({
    queryKey: ["log", taskId, lines],
    queryFn: () => get(`/api/logs/${encodeURIComponent(taskId ?? "")}?lines=${lines}`),
    enabled: Boolean(taskId),
    // FAST rather than SLOW: it matches the server's own 15s memo, so a tab
    // polling this cannot cost more share reads than a tab that is merely open.
    refetchInterval: live ? FAST : false,
    staleTime: live ? 0 : 5 * 60_000,
  });

/**
 * A local directory read, and the only query here that touches neither Azure
 * nor the share. `staleTime: Infinity` because a config file appearing while
 * the console is open is not a thing worth polling for — the picker has a
 * refresh, and that is the honest cost of a list that changes when someone
 * edits the repo.
 */
export const useConfigs = () =>
  useQuery<Configs>({
    queryKey: ["configs"],
    queryFn: () => get("/api/configs"),
    staleTime: Number.POSITIVE_INFINITY,
  });

/**
 * The local activity log. A file read, so it is cheap — but it is also the only
 * query whose answer THIS PAGE changes, since asking is itself an invocation.
 * Polled slowly for that reason as much as for cost.
 */
export const useActivity = (days = 7) =>
  useQuery<Activity>({
    queryKey: ["activity", days],
    queryFn: () => get(`/api/activity?days=${days}`),
    refetchInterval: SLOW,
  });

/**
 * A paired comparison. Not polled at all, and `enabled` only once BOTH runs are
 * chosen — this is a question someone asks deliberately, and the answer cannot
 * change without a new evaluation landing.
 */
export const useCompare = (a: string, b: string, force: boolean) =>
  useQuery<Comparison>({
    queryKey: ["compare", a, b, force],
    queryFn: () =>
      get(`/api/compare?a=${encodeURIComponent(a)}&b=${encodeURIComponent(b)}&force=${force}`),
    enabled: Boolean(a && b),
    refetchInterval: false,
  });

/**
 * The dispatching writes.
 *
 * Each invalidates what its work will show up in — `jobs` and `tasks` for a
 * queued task, `runs` for a promotion. The click is not the authority on any of
 * it: Batch is, and the invalidation is how the page goes and asks.
 *
 * The bodies are `Record<string, unknown>` rather than mirrored interfaces on
 * purpose. The server's models already declare what each command accepts, and
 * they declare it by DROPPING what the caller omitted — a TypeScript interface
 * that filled in defaults would put them back, which is the exact disagreement
 * that design exists to prevent. The form builds the body it means to send.
 */
const useDispatch = <T>(path: string) => {
  const queryClient = useQueryClient();
  return useMutation<T, Error, Record<string, unknown>>({
    mutationFn: (body) => send<T>(path, body),
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
      queryClient.invalidateQueries({ queryKey: ["tasks"] });
    },
  });
};

export const useSubmit = () => useDispatch<Dispatched>("/api/submit");
export const useScore = () => useDispatch<Dispatched>("/api/score");
export const usePrecompute = () => useDispatch<Dispatched>("/api/precompute");

export const usePushCode = () =>
  useMutation<PushedCode, Error, Record<string, unknown>>({
    mutationFn: (body) => send("/api/push-code", body),
  });

export const usePushData = () =>
  useMutation<PushedData, Error, Record<string, unknown>>({
    mutationFn: (body) => send("/api/push-data", body),
  });

/**
 * `compact-legs`, both halves. The dry run and the apply are the same endpoint
 * and differ only in the body, which is why they are one hook: a page that had
 * two could show a preview from one and apply the other.
 */
export const useCompactLegs = () => {
  const queryClient = useQueryClient();
  return useMutation<Compacted, Error, Record<string, unknown>>({
    mutationFn: (body) => send("/api/compact-legs", body),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["tasks"] }),
  });
};

export const usePromote = () => {
  const queryClient = useQueryClient();
  return useMutation<Promoted, Error, Record<string, unknown>>({
    mutationFn: (body) => send("/api/promote", body),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["runs"] }),
  });
};

/**
 * The blueprint server. Polled ONLY while a swap is in flight.
 *
 * A loaded run does not change under you, so an interval would normally be pure
 * cost — but it CAN now change, because this console can ask for it, and a load
 * takes about a minute on the far side. So the cadence follows the state: fast
 * while `loading` is set, never otherwise. `combos` is the canonical 1326-entry
 * order and never changes at all, hence `staleTime: Infinity`.
 */
export const useBlueprintRun = () =>
  useQuery<BlueprintRun>({
    queryKey: ["blueprint", "run"],
    queryFn: () => get("/api/blueprint/run"),
    refetchInterval: (query) => (query.state.data?.loading ? 2_000 : false),
    staleTime: Number.POSITIVE_INFINITY,
  });

/**
 * Ask the box to serve a different run.
 *
 * Returns as soon as the far side has ACCEPTED the job (202), not when the run
 * is in — the load takes about a minute and holding a request open that long
 * would cross three proxies with shorter timeouts than that. Invalidating the
 * blueprint queries is what starts the polling above; the node and combo caches
 * go with it, because they describe the run that is being replaced.
 */
export const useLoadBlueprintRun = () => {
  const queryClient = useQueryClient();
  return useMutation<BlueprintLoad, Error, { run: string; at?: number | null }>({
    mutationFn: (body) => send("/api/blueprint/load", body),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["blueprint"] }),
  });
};

export const useCombos = (enabled: boolean) =>
  useQuery<Combos>({
    queryKey: ["blueprint", "combos"],
    queryFn: () => get("/api/blueprint/combos"),
    staleTime: Number.POSITIVE_INFINITY,
    enabled,
  });

export const useSolverNode = (path: string, board: string, average: boolean, enabled: boolean) =>
  useQuery<SolverNode>({
    queryKey: ["blueprint", "node", path, board, average],
    queryFn: () =>
      get(
        `/api/blueprint/node?path=${encodeURIComponent(path)}&board=${encodeURIComponent(board)}&average=${average}`,
      ),
    enabled,
  });

/**
 * Play. Mutations rather than queries: a hand advances because you acted, so
 * there is nothing to poll and a refetch would replay a move.
 */
export const useDealHand = () =>
  useMutation<Hand, Error, { human_seat: number; seed?: number | null }>({
    mutationFn: (body) => send("/api/blueprint/play", body),
  });

export const useSubmitAction = () =>
  useMutation<Hand, Error, { session: string; token: string }>({
    mutationFn: ({ session, token }) => send(`/api/blueprint/play/${session}/action`, { token }),
  });

/**
 * Give a hand back when you walk away from it.
 *
 * `Blueprint.tsx` already claimed this happened — it mounts the two tabs one at
 * a time so that "unmounting ends it where it lives" — and nothing called the
 * endpoint, which has existed all along. A session lives on the BLUEPRINT
 * SERVER, in a store of 64 that drops the oldest to make room, so an abandoned
 * one is not free: it evicts someone's hand in progress. The store is bounded
 * precisely because a browser tab never says goodbye; this is the goodbye.
 *
 * No `onSettled` invalidation, and no error handling at the call site: leaving
 * is idempotent on the far side, and a failure to do it costs one slot in a
 * cache that already evicts.
 */
export const useLeaveHand = () =>
  useMutation<LeftSession, Error, { session: string }>({
    mutationFn: ({ session }) =>
      send(`/api/blueprint/play/${encodeURIComponent(session)}`, undefined, "DELETE"),
  });

/**
 * The host's power state. Polled FAST only while it is mid-transition: a box
 * that is settled will not change without someone clicking, and a box that is
 * waking needs to be watched or the page lies for two minutes.
 */
export const useBox = () =>
  useQuery<Box>({
    queryKey: ["blueprint", "box"],
    queryFn: () => get("/api/box"),
    refetchInterval: (query) => {
      const power = query.state.data?.power;
      return power === "running" || power === "deallocated" ? 30_000 : 5_000;
    },
  });

export const useBoxAction = () =>
  useMutation<Box, Error, "start" | "stop">({
    mutationFn: (action) => send(`/api/box/${action}`),
  });

/**
 * Cancel one task. Invalidates the task list rather than patching it: the
 * authority on whether a task is still running is Batch, not this click.
 */
export const useCancelTask = () => {
  const queryClient = useQueryClient();
  return useMutation<Cancelled, Error, { job: string; task: string }>({
    mutationFn: ({ job, task }) =>
      send(`/api/tasks/${encodeURIComponent(job)}/${encodeURIComponent(task)}/cancel`),
    onSettled: () => queryClient.invalidateQueries({ queryKey: ["tasks"] }),
  });
};
