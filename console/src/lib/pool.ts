import type { Jobs } from "@/api/types";
import { shortState } from "@/components/StatusBadge";

/**
 * What the pool IS, rather than what the API happened to return.
 *
 * The Batch panel used to be `jobs.flatMap(job => job.tasks)` with no sort at
 * all, so a task that finished two days ago sat between two that are running
 * now. Three things were invisible in that list:
 *
 *  - **The queue.** A waiting task was a row with a badge saying "queued". You
 *    could not see how many were ahead of it or which ran next.
 *  - **Occupancy.** A running task was a row with a colour, unconnected to the
 *    fact that it is holding one of a small, fixed number of nodes. Whether the
 *    pool was full or half idle did not show anywhere.
 *  - **Order.** There wasn't one.
 *
 * So this splits the flat list into the three things a pool actually has:
 * SLOTS (finite, some occupied), a QUEUE (ordered, waiting for a slot), and
 * HISTORY (done, newest first).
 */

export type Task = Jobs["jobs"][number]["tasks"][number] & { job: string };

/** Batch states meaning a node is committed to this task right now. */
const OCCUPYING = new Set(["running", "preparing"]);
/** Waiting for a node. Batch calls it `active`, which reads like "healthy". */
const WAITING = "active";

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

  const occupying = tasks.filter((t) => OCCUPYING.has(shortState(t.state)));
  // Oldest first: a task that has been running longest is nearest to finishing,
  // so it is the one whose slot frees up next.
  occupying.sort((a, b) => at(a.start_time, at(a.created, 0)) - at(b.start_time, at(b.created, 0)));

  // Submission order — the order Batch will actually dispatch them.
  const queue = tasks
    .filter((t) => shortState(t.state) === WAITING)
    .sort(
      (a, b) => at(a.created, Number.MAX_SAFE_INTEGER) - at(b.created, Number.MAX_SAFE_INTEGER),
    );

  const history = tasks
    .filter((t) => !OCCUPYING.has(shortState(t.state)) && shortState(t.state) !== WAITING)
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
