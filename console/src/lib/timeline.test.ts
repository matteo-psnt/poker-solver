import { describe, expect, it } from "vitest";
import { timelineBars } from "./timeline";

const NOW = Date.parse("2026-08-04T12:00:00Z");

const task = (over: Record<string, unknown>) =>
  ({
    task_id: "t",
    attempt: 1,
    op: "train",
    run_id: "run-a",
    cause: "completed",
    exit_code: 0,
    ended_at: null,
    ...over,
    // biome-ignore lint/suspicious/noExplicitAny: a hand-built row stands in for a parsed one
  }) as any;

describe("timelineBars", () => {
  it("lays tasks out against one shared axis, in the order they ran", () => {
    const t = timelineBars(
      [
        task({
          task_id: "second",
          started_at: "2026-08-04T11:00:00Z",
          ended_at: "2026-08-04T12:00:00Z",
        }),
        task({
          task_id: "first",
          started_at: "2026-08-04T10:00:00Z",
          ended_at: "2026-08-04T11:00:00Z",
        }),
      ],
      NOW,
    );
    expect(t?.bars.map((b) => b.taskId)).toEqual(["first", "second"]);
    expect(t?.bars[0]?.leftPct).toBeCloseTo(0);
    expect(t?.bars[0]?.widthPct).toBeCloseTo(50);
    expect(t?.bars[1]?.leftPct).toBeCloseTo(50);
  });

  it("makes a gap between tasks visible as a gap", () => {
    /** Two days where nothing touched the run is the fact a column of
        timestamps hides best. */
    const t = timelineBars(
      [
        task({ started_at: "2026-08-01T00:00:00Z", ended_at: "2026-08-01T01:00:00Z" }),
        task({
          task_id: "b",
          started_at: "2026-08-04T11:00:00Z",
          ended_at: "2026-08-04T12:00:00Z",
        }),
      ],
      NOW,
    );
    const [first, second] = t?.bars ?? [];
    const gap = (second?.leftPct ?? 0) - ((first?.leftPct ?? 0) + (first?.widthPct ?? 0));
    expect(gap).toBeGreaterThan(90);
  });

  it("keeps a task that died instantly visible rather than zero-width", () => {
    /** The shortest bar is the most interesting one on the chart. */
    const t = timelineBars(
      [
        task({ started_at: "2026-08-01T00:00:00Z", ended_at: "2026-08-04T00:00:00Z" }),
        task({
          task_id: "died",
          cause: "failed",
          started_at: "2026-08-04T11:59:00Z",
          ended_at: "2026-08-04T11:59:02Z",
        }),
      ],
      NOW,
    );
    const died = t?.bars.find((b) => b.taskId === "died");
    expect(died?.widthPct).toBeGreaterThan(0.5);
    expect(died?.tone).toBe("bad");
  });

  it("runs an unfinished task to now and says so", () => {
    const t = timelineBars(
      [task({ task_id: "live", cause: "running", started_at: "2026-08-04T11:00:00Z" })],
      NOW,
    );
    expect(t?.bars[0]?.running).toBe(true);
    expect(t?.to).toBe(NOW);
  });

  it("never lets a bar overflow the axis", () => {
    const t = timelineBars(
      [
        task({ started_at: "2026-08-04T10:00:00Z", ended_at: "2026-08-04T10:30:00Z" }),
        task({ task_id: "late", started_at: "2026-08-04T11:59:59Z", ended_at: null }),
      ],
      NOW,
    );
    for (const bar of t?.bars ?? []) {
      expect(bar.leftPct + bar.widthPct).toBeLessThanOrEqual(100.001);
    }
  });

  it("is null rather than a broken chart when nothing carries a time", () => {
    expect(timelineBars([task({ started_at: null })], NOW)).toBeNull();
    expect(timelineBars([], NOW)).toBeNull();
  });

  it("labels a bar with what the task did, falling back to its op", () => {
    const t = timelineBars(
      [
        task({ what: "evaluate @150M seed7", started_at: "2026-08-04T11:00:00Z" }),
        task({ task_id: "b", what: "", op: "train", started_at: "2026-08-04T11:30:00Z" }),
      ],
      NOW,
    );
    expect(t?.bars.map((b) => b.label)).toEqual(["evaluate @150M seed7", "train"]);
  });
});
