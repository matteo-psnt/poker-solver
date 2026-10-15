import { describe, expect, it } from "vitest";
import type { TaskRow } from "@/api/types";
import { groupFinished, medianSeconds } from "./groups";

function row(over: Partial<TaskRow>): TaskRow {
  return {
    task_id: "t",
    attempt: 1,
    run_id: "run-a",
    what: "evaluate @24k",
    cause: "completed",
    started_at: "2026-10-15T16:25:00Z",
    ended_at: "2026-10-15T16:26:30Z",
    ...over,
  } as TaskRow;
}

describe("groupFinished", () => {
  it("folds a gate's seed tasks into one group even with other work between them", () => {
    const groups = groupFinished([
      row({ task_id: "s1", ended_at: "2026-10-15T16:26:00Z" }),
      row({ task_id: "settle", what: "evaluate @26k", ended_at: "2026-10-15T16:27:00Z" }),
      row({ task_id: "s2", ended_at: "2026-10-15T16:28:00Z" }),
    ]);
    expect(groups.map((g) => g.rows.length)).toEqual([2, 1]);
    expect(groups[0]?.rows.map((r) => r.task_id)).toEqual(["s2", "s1"]);
  });

  it("keeps a different outcome apart", () => {
    const groups = groupFinished([row({ cause: "failed" }), row({ cause: "completed" })]);
    expect(groups).toHaveLength(2);
  });

  it("starts a new group once the window from the earliest end has passed", () => {
    const groups = groupFinished([
      row({ ended_at: "2026-10-15T10:00:00Z" }),
      row({ ended_at: "2026-10-15T11:00:00Z" }),
    ]);
    expect(groups).toHaveLength(2);
  });

  it("skips attempts that never ended", () => {
    expect(groupFinished([row({ ended_at: null })])).toEqual([]);
  });
});

describe("medianSeconds", () => {
  it("returns the middle duration", () => {
    const rows = [
      row({ started_at: "2026-10-15T00:00:00Z", ended_at: "2026-10-15T00:01:00Z" }),
      row({ started_at: "2026-10-15T00:00:00Z", ended_at: "2026-10-15T00:02:00Z" }),
      row({ started_at: "2026-10-15T00:00:00Z", ended_at: "2026-10-15T00:10:00Z" }),
    ];
    expect(medianSeconds(rows)).toBe(120);
  });
});
