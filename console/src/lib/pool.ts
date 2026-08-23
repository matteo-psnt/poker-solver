import type { Jobs, NodePhase, NodeStatus, Phase } from "@/api/types";

/**
 * What the pool IS, rather than what the API happened to return.
 *
 * A flat list of tasks hides the two things a pool actually has, so this splits
 * it into them: NODES (real machines, each in a state, some holding a task) and
 * a QUEUE (ordered, waiting for a node).
 *
 * The nodes are Batch's own list, not a count drawn as hollow dots. That is the
 * difference between "7 of 7" and "7, two of them still booting" -- which is
 * the answer to why a queued task is not running, and nothing else on the
 * screen can give it.
 */

export type Task = Jobs["jobs"][number]["tasks"][number] & { job: string };

/**
 * Which task phases hold a node, and which wait for one.
 *
 * These used to be Batch's raw strings, re-derived here from
 * `"BatchTaskState.*"` — one of four sites that each classified those strings
 * independently. The phases are the server's now
 * (`src/shared/task_states.py`), so this says only which of them this SCREEN
 * draws on a node and which in the queue.
 */
const OCCUPYING: ReadonlySet<Phase> = new Set<Phase>(["running", "starting"]);
const WAITING: Phase = "queued";

/** Node phases that will take a task without anyone doing anything. */
const WILL_FREE: ReadonlySet<NodePhase> = new Set<NodePhase>(["idle", "booting"]);

/** Rows are grouped by what the reader wants to see first: work, then what is coming, then what is wrong. */
const PHASE_ORDER: Record<NodePhase, number> = {
  busy: 0,
  booting: 1,
  idle: 2,
  down: 3,
  leaving: 4,
  unknown: 5,
};

/** A node and what it holds. `node` is null for a task Batch says is running on a node it did not list. */
export type Slot = { node: NodeStatus | null; task: Task | null };

export type PoolShape = {
  slots: Slot[];
  queue: Task[];
  /** Waiting tasks with no node idle or booting to go to: the queue cannot drain as-is. */
  starved: number;
  /** How many nodes are in each phase, for the panel's title. Zero counts are absent. */
  byPhase: Partial<Record<NodePhase, number>>;
};

/** Epoch millis, or `fallback` when absent/unparseable, so sorts stay total. */
function at(value: string | null | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? fallback : parsed;
}

export function poolShape(jobs: Jobs | undefined, nodes: NodeStatus[] | undefined): PoolShape {
  const tasks: Task[] = (jobs?.jobs ?? []).flatMap((job) =>
    job.tasks.map((task) => ({ ...task, job: job.job })),
  );

  const occupying = tasks.filter((t) => OCCUPYING.has(t.phase));
  // Oldest first: a task that has been running longest is nearest to finishing,
  // so it is the one whose node frees up next.
  occupying.sort((a, b) => at(a.start_time, at(a.created, 0)) - at(b.start_time, at(b.created, 0)));

  // Submission order — the order Batch will actually dispatch them.
  const queue = tasks
    .filter((t) => t.phase === WAITING)
    .sort(
      (a, b) => at(a.created, Number.MAX_SAFE_INTEGER) - at(b.created, Number.MAX_SAFE_INTEGER),
    );

  // The join, both ways: Batch names the node on the task AND the task on the
  // node, and a freshly started task can be on one list before the other.
  const byId = new Map(occupying.map((task) => [task.task, task]));
  const placed = new Set<string>();
  const slots: Slot[] = [...(nodes ?? [])]
    .sort(
      (a, b) =>
        PHASE_ORDER[a.phase] - PHASE_ORDER[b.phase] ||
        at(a.since, 0) - at(b.since, 0) ||
        a.id.localeCompare(b.id),
    )
    .map((node) => {
      const task =
        node.tasks.map((id) => byId.get(id)).find(Boolean) ??
        occupying.find((t) => t.node === node.id) ??
        null;
      if (task) placed.add(task.task);
      return { node, task };
    });
  // Never hide a running task: the pool can be mid-resize, or the node list
  // can have failed while the job list did not.
  for (const task of occupying) {
    if (!placed.has(task.task)) slots.push({ node: null, task });
  }

  const byPhase: Partial<Record<NodePhase, number>> = {};
  for (const node of nodes ?? []) byPhase[node.phase] = (byPhase[node.phase] ?? 0) + 1;

  const free = (nodes ?? []).filter(
    (node) => WILL_FREE.has(node.phase) && !node.tasks.length,
  ).length;
  return { slots, queue, starved: Math.max(0, queue.length - free), byPhase };
}

/** `2h 14m`, `3m`, `just now` — how long something has been as it is. */
export function elapsed(since: string | null | undefined, now: number): string {
  if (!since) return "";
  const start = Date.parse(since);
  if (Number.isNaN(start)) return "";
  const minutes = Math.floor(Math.max(0, now - start) / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}

/**
 * `tvmps_…74de44` — the tail of a node id, which is the only part that differs.
 *
 * Batch node ids are `tvmps_<64 hex>_d`; the prefix and suffix are the same on
 * every node in the pool. Always pair it with the full id in a `title`.
 */
export function nodeLabel(id: string): string {
  const hex = id.replace(/^tvmps_/, "").replace(/_d$/, "");
  return hex.length > 6 ? `…${hex.slice(-6)}` : id;
}
