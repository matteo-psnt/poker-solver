import type { Jobs, Phase } from "@/api/types";

/**
 * What the pool IS, rather than what the API happened to return.
 *
 * A flat list of tasks hides the three things a pool actually has, so this splits
 * it into them: SLOTS (finite, some occupied), a QUEUE (ordered, waiting for a
 * slot), and HISTORY (done, newest first).
 *
 * Occupancy is the point of the first: a running task holds one of a small, fixed
 * number of nodes, and whether the pool is full or half idle does not show
 * anywhere else.
 */

export type Task = Jobs["jobs"][number]["tasks"][number] & { job: string };

/**
 * Which phases hold a node, and which wait for one.
 *
 * These used to be Batch's raw strings, re-derived here from
 * `"BatchTaskState.*"` — one of four sites that each classified those strings
 * independently. The phases are the server's now
 * (`src/shared/task_states.py`), so this says only which of them this SCREEN
 * draws as a slot and which as a queue.
 */
const OCCUPYING: ReadonlySet<Phase> = new Set<Phase>(["running", "starting"]);
const WAITING: Phase = "queued";

export type Slot = { index: number; task: Task | null };

export type PoolShape = {
  slots: Slot[];
  queue: Task[];
  history: Task[];
  /** Waiting tasks with no slot to go to, i.e. the queue cannot drain as-is. */
  starved: number;
};

/** Epoch millis, or `fallback` when absent/unparseable, so sorts stay total. */
function at(value: string | null | undefined, fallback: number): number {
  if (!value) return fallback;
  const parsed = Date.parse(value);
  return Number.isNaN(parsed) ? fallback : parsed;
}

export function poolShape(jobs: Jobs | undefined, nodes: number | null | undefined): PoolShape {
  const tasks: Task[] = (jobs?.jobs ?? []).flatMap((job) =>
    job.tasks.map((task) => ({ ...task, job: job.job })),
  );

  const occupying = tasks.filter((t) => OCCUPYING.has(t.phase));
  // Oldest first: a task that has been running longest is nearest to finishing,
  // so it is the one whose slot frees up next.
  occupying.sort((a, b) => at(a.start_time, at(a.created, 0)) - at(b.start_time, at(b.created, 0)));

  // Submission order — the order Batch will actually dispatch them.
  const queue = tasks
    .filter((t) => t.phase === WAITING)
    .sort(
      (a, b) => at(a.created, Number.MAX_SAFE_INTEGER) - at(b.created, Number.MAX_SAFE_INTEGER),
    );

  const history = tasks
    .filter((t) => !OCCUPYING.has(t.phase) && t.phase !== WAITING)
    .sort((a, b) => at(b.end_time, at(b.created, 0)) - at(a.end_time, at(a.created, 0)));

  // Never fewer slots than there are tasks running: the pool can be mid-resize,
  // and drawing 2 slots while 4 things run would hide two of them entirely.
  const width = Math.max(nodes ?? 0, occupying.length);
  const slots: Slot[] = Array.from({ length: width }, (_, index) => ({
    index,
    task: occupying[index] ?? null,
  }));

  const free = slots.filter((slot) => slot.task === null).length;
  return { slots, queue, history, starved: Math.max(0, queue.length - free) };
}

/** `2h 14m`, `3m`, `just now` — how long a task has held its slot. */
export function elapsed(since: string | null | undefined, now: number): string {
  if (!since) return "";
  const start = Date.parse(since);
  if (Number.isNaN(start)) return "";
  const minutes = Math.floor(Math.max(0, now - start) / 60_000);
  if (minutes < 1) return "just now";
  if (minutes < 60) return `${minutes}m`;
  return `${Math.floor(minutes / 60)}h ${minutes % 60}m`;
}
