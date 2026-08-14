import { describe, expect, it } from "vitest";
import { elapsed, poolShape } from "./pool";

/**
 * A task as the SERVER sends it: already classified.
 *
 * These fixtures used to carry `"BatchTaskState.COMPLETED"` and let `poolShape`
 * work out what that meant. It does not any more -- `phase` arrives decided
 * (`src/shared/task_states.py`), so a fixture that spelled a raw Batch state
 * here would be testing a translation this console no longer performs.
 */
const T = (over: Record<string, unknown>) => ({
  task: "t",
  phase: "finished",
  exit_code: 0,
  ...over,
});

const jobs = (...tasks: Record<string, unknown>[]) =>
  ({
    op: "jobs",
    jobs: [{ job: "poker-20260804", state: "BatchJobState.ACTIVE", tasks }],
    total_jobs: 1,
    hidden_jobs: 0,
    // biome-ignore lint/suspicious/noExplicitAny: a hand-built payload stands in for the parsed one
  }) as any;

describe("poolShape", () => {
  it("separates the three things a pool has, which one flat list could not show", () => {
    const shape = poolShape(
      jobs(
        T({
          task: "done",
          phase: "finished",
          end_time: "2026-08-04T09:00:00Z",
        }),
        T({
          task: "live",
          phase: "running",
          start_time: "2026-08-04T09:30:00Z",
        }),
        T({
          task: "next",
          phase: "queued",
          created: "2026-08-04T09:31:00Z",
        }),
      ),
      2,
    );
    expect(shape.slots.map((s) => s.task?.task ?? null)).toEqual(["live", null]);
    expect(shape.queue.map((t) => t.task)).toEqual(["next"]);
    expect(shape.history.map((t) => t.task)).toEqual(["done"]);
  });

  it("draws a slot per node even when nothing is in it", () => {
    /** A list of 2 running tasks looks identical whether the pool holds 2 or 8. */
    const shape = poolShape(jobs(T({ phase: "running" })), 4);
    expect(shape.slots).toHaveLength(4);
    expect(shape.slots.filter((s) => s.task === null)).toHaveLength(3);
  });

  it("never hides a running task when the pool is mid-resize", () => {
    /** Target can lag reality downward; drawing 1 slot for 3 running tasks
        would drop two of them off the page entirely. */
    const shape = poolShape(
      jobs(
        T({ task: "a", phase: "running" }),
        T({ task: "b", phase: "running" }),
        T({ task: "c", phase: "running" }),
      ),
      1,
    );
    expect(shape.slots).toHaveLength(3);
  });

  it("orders the queue by submission, which is the order Batch dispatches", () => {
    const shape = poolShape(
      jobs(
        T({
          task: "third",
          phase: "queued",
          created: "2026-08-04T09:05:00Z",
        }),
        T({
          task: "first",
          phase: "queued",
          created: "2026-08-04T09:01:00Z",
        }),
        T({
          task: "second",
          phase: "queued",
          created: "2026-08-04T09:03:00Z",
        }),
      ),
      0,
    );
    expect(shape.queue.map((t) => t.task)).toEqual(["first", "second", "third"]);
  });

  it("puts the longest-running task first, because its slot frees up next", () => {
    const shape = poolShape(
      jobs(
        T({
          task: "young",
          phase: "running",
          start_time: "2026-08-04T09:30:00Z",
        }),
        T({
          task: "old",
          phase: "running",
          start_time: "2026-08-04T06:00:00Z",
        }),
      ),
      2,
    );
    expect(shape.slots.map((s) => s.task?.task)).toEqual(["old", "young"]);
  });

  it("orders finished work newest first", () => {
    const shape = poolShape(
      jobs(
        T({ task: "older", end_time: "2026-08-04T08:00:00Z" }),
        T({ task: "newest", end_time: "2026-08-04T10:00:00Z" }),
        T({ task: "middle", end_time: "2026-08-04T09:00:00Z" }),
      ),
      0,
    );
    expect(shape.history.map((t) => t.task)).toEqual(["newest", "middle", "older"]);
  });

  it("counts what cannot start even if every slot freed right now", () => {
    /** The pool is at its ceiling; the tail waits on a resize, not on a task. */
    const shape = poolShape(
      jobs(
        T({ task: "r", phase: "running" }),
        ...["a", "b", "c"].map((task) => T({ task, phase: "queued" })),
      ),
      2,
    );
    expect(shape.queue).toHaveLength(3);
    expect(shape.starved).toBe(2); // 3 waiting, 1 free slot
  });

  it("treats preparing as occupying — the node is committed", () => {
    const shape = poolShape(jobs(T({ task: "p", phase: "starting" })), 1);
    expect(shape.slots[0]?.task?.task).toBe("p");
    expect(shape.queue).toHaveLength(0);
  });

  it("survives tasks with no timestamps at all rather than sorting randomly", () => {
    const shape = poolShape(jobs(T({ task: "a" }), T({ task: "b" })), 0);
    expect(shape.history).toHaveLength(2);
  });

  it("is empty, not broken, before the first fetch", () => {
    const shape = poolShape(undefined, null);
    expect(shape).toMatchObject({
      slots: [],
      queue: [],
      history: [],
      starved: 0,
    });
  });
});

describe("elapsed", () => {
  const now = Date.parse("2026-08-04T12:00:00Z");

  it("reads as a person would say it", () => {
    expect(elapsed("2026-08-04T09:46:00Z", now)).toBe("2h 14m");
    expect(elapsed("2026-08-04T11:57:00Z", now)).toBe("3m");
    expect(elapsed("2026-08-04T11:59:59Z", now)).toBe("just now");
  });

  it("is blank rather than wrong when there is no time to show", () => {
    expect(elapsed(null, now)).toBe("");
    expect(elapsed("not a date", now)).toBe("");
  });
});
