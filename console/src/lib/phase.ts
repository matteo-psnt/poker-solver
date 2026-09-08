import type { Phase, TaskRow } from "@/api/types";

/**
 * Which task phases hold a node and which merely wait for one — the browser
 * half of `src/shared/task_states.py`, whose `Phase` these strings ARE.
 *
 * Three copies of this lived in three components, and a fourth question ("is it
 * still going?") was answered somewhere else entirely, as `!row.ended_at`. That
 * one was wrong: only a task that exits gracefully stamps an end, so 1,293
 * attempts killed by OOM, a wall clock or a lost node read as live forever —
 * which put a Cancel button on tasks that died weeks ago and refetched their
 * logs every fifteen seconds. The server sends `phase` now; ask it.
 */
export const OCCUPIES_A_NODE: ReadonlySet<Phase> = new Set<Phase>(["running", "starting"]);

/** Work that exists and holds no node. Queue depth, never node time. */
export const PENDING: Phase = "queued";

/** It has a node or is waiting for one: worth listing, worth cancelling. */
export const IN_FLIGHT: ReadonlySet<Phase> = new Set<Phase>([...OCCUPIES_A_NODE, PENDING]);

/** Still going, so it can be cancelled and its log is still growing. */
export function inFlight(row: TaskRow): boolean {
  return IN_FLIGHT.has(row.phase);
}
