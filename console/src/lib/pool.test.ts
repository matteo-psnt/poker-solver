import type { NodeStatus } from "@/api/types";
import { describe, expect, it } from "vitest";
import { elapsed, nodeLabel, poolShape } from "./pool";

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

const N = (over: Partial<NodeStatus>): NodeStatus => ({
  id: "tvmps_0000000000000000000000000000000000000000000000000000000000aaaaaa_d",
  state: "idle",
  phase: "idle",
  tasks: [],
  errors: [],
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
        T({ task: "done", phase: "finished", end_time: "2026-08-04T09:00:00Z" }),
        T({ task: "live", phase: "running", start_time: "2026-08-04T09:30:00Z", node: "n1" }),
        T({ task: "next", phase: "queued", created: "2026-08-04T09:31:00Z" }),
      ),
      [N({ id: "n1", state: "running", phase: "busy", tasks: ["live"] }), N({ id: "n2" })],
    );
    expect(shape.slots.map((s) => [s.node?.id, s.task?.task ?? null])).toEqual([
      ["n1", "live"],
      ["n2", null],
    ]);
    expect(shape.queue.map((t) => t.task)).toEqual(["next"]);
    expect(shape.history.map((t) => t.task)).toEqual(["done"]);
    expect(shape.byPhase).toEqual({ busy: 1, idle: 1 });
  });

  it("draws every node, including the ones still booting, which is the point", () => {
    /** "7 of 7" cannot say that two of them will not take a task for minutes yet. */
    const shape = poolShape(jobs(T({ task: "next", phase: "queued" })), [
      N({ id: "n1", state: "waitingforstarttask", phase: "booting" }),
      N({ id: "n2", state: "unusable", phase: "down", errors: ["MountConfigurationError: x"] }),
      N({ id: "n3" }),
    ]);
    expect(shape.slots.map((s) => s.node?.phase)).toEqual(["booting", "idle", "down"]);
    expect(shape.byPhase).toEqual({ booting: 1, down: 1, idle: 1 });
  });

  it("joins a task to its node from either side of the relation", () => {
    /** A task just started can be on the task's record before the node's. */
    const shape = poolShape(jobs(T({ task: "by-task", phase: "running", node: "n1" })), [
      N({ id: "n1", state: "running", phase: "busy", tasks: [] }),
    ]);
    expect(shape.slots[0]?.task?.task).toBe("by-task");
  });

  it("never hides a running task whose node is not listed", () => {
    /** Mid-resize, or the node list failed while the job list did not. */
    const shape = poolShape(jobs(T({ task: "orphan", phase: "running" })), []);
    expect(shape.slots).toEqual([
      { node: null, task: expect.objectContaining({ task: "orphan" }) },
    ]);
  });

  it("counts the queue as starved only beyond the nodes that will free up on their own", () => {
    const queued = [
      T({ task: "q1", phase: "queued", created: "2026-08-04T09:31:00Z" }),
      T({ task: "q2", phase: "queued", created: "2026-08-04T09:32:00Z" }),
      T({ task: "q3", phase: "queued", created: "2026-08-04T09:33:00Z" }),
    ];
    const shape = poolShape(jobs(...queued), [
      N({ id: "idle" }),
      N({ id: "booting", state: "starting", phase: "booting" }),
      N({ id: "busy", state: "running", phase: "busy", tasks: ["elsewhere"] }),
      N({ id: "dead", state: "unusable", phase: "down" }),
    ]);
    expect(shape.queue.map((t) => t.task)).toEqual(["q1", "q2", "q3"]);
    expect(shape.starved).toBe(1);
  });

  it("orders the queue by submission, which is the order Batch dispatches it", () => {
    const shape = poolShape(
      jobs(
        T({ task: "second", phase: "queued", created: "2026-08-04T09:32:00Z" }),
        T({ task: "first", phase: "queued", created: "2026-08-04T09:31:00Z" }),
      ),
      [],
    );
    expect(shape.queue.map((t) => t.task)).toEqual(["first", "second"]);
  });
});

describe("elapsed", () => {
  const now = Date.parse("2026-08-04T12:00:00Z");
  it("rounds to what a glance needs", () => {
    expect(elapsed("2026-08-04T11:59:30Z", now)).toBe("just now");
    expect(elapsed("2026-08-04T11:45:00Z", now)).toBe("15m");
    expect(elapsed("2026-08-04T09:46:00Z", now)).toBe("2h 14m");
  });
  it("is blank rather than wrong for an absent or unparseable instant", () => {
    expect(elapsed(null, now)).toBe("");
    expect(elapsed("yesterday", now)).toBe("");
  });
});

describe("nodeLabel", () => {
  it("keeps the tail, which is the only part that differs between nodes", () => {
    expect(
      nodeLabel("tvmps_ea246bb49543854e30eb048a2dbe2727be252f8e0cb4ea8f678e9cb2cf74de44_d"),
    ).toBe("…74de44");
  });
  it("leaves an unfamiliar id alone", () => {
    expect(nodeLabel("node-7")).toBe("node-7");
  });
});
