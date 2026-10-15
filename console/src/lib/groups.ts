import type { TaskRow } from "@/api/types";

/**
 * Finished attempts that are one piece of work: same run, same `what`, same
 * outcome, ending close together. A 13-seed gate is 13 rows in the task log and
 * one thing to a reader.
 *
 * The window is measured from the group's EARLIEST end so far, so a long tail of
 * stragglers cannot chain unrelated work together.
 */
export type TaskGroup = {
  rows: TaskRow[];
  /** Epoch millis of the newest and oldest end in the group. */
  last: number;
  first: number;
};

const WINDOW_MS = 30 * 60_000;

export function groupFinished(rows: readonly TaskRow[], windowMs = WINDOW_MS): TaskGroup[] {
  const ended = rows
    .filter((row) => row.ended_at)
    .map((row) => ({ row, end: Date.parse(row.ended_at as string) }))
    .filter(({ end }) => !Number.isNaN(end))
    .sort((a, b) => b.end - a.end);
  const groups: (TaskGroup & { key: string })[] = [];
  for (const { row, end } of ended) {
    const key = `${row.run_id ?? row.task_id}|${row.what}|${row.cause}`;
    // Interleaved work is common -- a settle read lands in the middle of a
    // gate -- so look for an open group with this key, not just the last one.
    const open = groups.find((g) => g.key === key && g.first - end < windowMs);
    if (open) {
      open.rows.push(row);
      open.first = end;
    } else {
      groups.push({ key, rows: [row], first: end, last: end });
    }
  }
  return groups.map(({ rows, first, last }) => ({ rows, first, last }));
}

/** The median wall time of a group's attempts, in seconds; null if none can be measured. */
export function medianSeconds(rows: readonly TaskRow[]): number | null {
  const took = rows
    .map((row) => (Date.parse(row.ended_at ?? "") - Date.parse(row.started_at ?? "")) / 1000)
    .filter((s) => Number.isFinite(s) && s >= 0)
    .sort((a, b) => a - b);
  return took[Math.floor(took.length / 2)] ?? null;
}
