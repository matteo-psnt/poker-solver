import { describe, expect, it } from "vitest";
import {
  displayName,
  exitMeaning,
  shortState,
  taskOutcome,
  taskTone,
  toneFor,
} from "./StatusBadge";

/**
 * The console shows three state vocabularies with one colour scheme, and the
 * same word means different things in each. These pin the distinctions that
 * were actually wrong on screen.
 */
describe("a Batch task's colour needs its exit code, not just its state", () => {
  it("completed with exit 0 is success", () => {
    expect(taskTone("BatchTaskState.COMPLETED", 0)).toBe("ok");
  });

  it("completed with a non-zero exit is NOT success", () => {
    // The bug: Batch's `completed` means finished, not succeeded, so a badge
    // coloured on state alone painted a crashed task green.
    expect(taskTone("BatchTaskState.COMPLETED", 1)).toBe("bad");
    expect(taskTone("BatchTaskState.COMPLETED", 137)).toBe("bad");
  });

  it("a cancelled task is muted, not a failure", () => {
    // -9 is Batch's SIGKILL on cancel. Reading it as a crash sent us looking
    // for an OOM that never happened.
    expect(taskTone("BatchTaskState.COMPLETED", -9)).toBe("muted");
  });

  it("a timed-out task is a warning, not a failure", () => {
    expect(taskTone("BatchTaskState.COMPLETED", 124)).toBe("warn");
  });

  it("active is QUEUED, so it must not pulse like live work", () => {
    // A task frozen `active` inside a finished job will never run at all.
    expect(taskTone("BatchTaskState.ACTIVE", null)).toBe("muted");
  });

  it("running and preparing are live", () => {
    expect(taskTone("BatchTaskState.RUNNING", null)).toBe("live");
    expect(taskTone("BatchTaskState.PREPARING", null)).toBe("live");
  });
});

describe("the task log's causes, where the word IS the outcome", () => {
  it("separates the three ways a task stops being alive", () => {
    expect(toneFor("completed")).toBe("ok");
    expect(toneFor("failed")).toBe("bad");
    expect(toneFor("timeout")).toBe("warn");
    expect(toneFor("cancelled")).toBe("muted");
  });

  it("killed is a failure and partial is a warning", () => {
    expect(toneFor("killed")).toBe("bad");
    expect(toneFor("partial")).toBe("warn");
  });

  it("an unresolved task is pending, not fine", () => {
    // `started` with no terminal record: the trap never ran. Showing it as
    // success would hide exactly the deaths the task log exists to catch.
    expect(toneFor("started")).toBe("pending");
  });

  it("an unknown cause is muted rather than guessed at", () => {
    expect(toneFor("something-new")).toBe("muted");
  });
});

describe("exit codes are explained rather than left as numbers", () => {
  it("names the ones that recur here", () => {
    expect(exitMeaning(2)).toContain("rejected a flag");
    expect(exitMeaning(124)).toContain("wall-clock");
    expect(exitMeaning(137)).toContain("OOM");
    expect(exitMeaning(-9)).toContain("cancellation");
  });

  it("leaves an unfamiliar code alone rather than inventing a meaning", () => {
    expect(exitMeaning(42)).toBeNull();
  });
});

describe("shortState", () => {
  it("strips Batch's enum prefix", () => {
    expect(shortState("BatchTaskState.RUNNING")).toBe("running");
    expect(shortState(null)).toBe("");
  });
});

describe("displayed names are not always the names on the wire", () => {
  it("shows a queued task as queued, not 'active'", () => {
    // Azure's word reads as the opposite of what it means.
    expect(taskOutcome("BatchTaskState.ACTIVE", null)).toBe("queued");
  });

  it("turns a stopped task into its outcome, one word", () => {
    expect(taskOutcome("BatchTaskState.COMPLETED", 0)).toBe("done");
    expect(taskOutcome("BatchTaskState.COMPLETED", 124)).toBe("timed out");
    expect(taskOutcome("BatchTaskState.COMPLETED", -9)).toBe("cancelled");
    expect(taskOutcome("BatchTaskState.COMPLETED", 137)).toBe("failed");
  });

  it("renames the task causes that hid what they meant", () => {
    expect(displayName("started")).toBe("unresolved");
    expect(displayName("killed")).toBe("killed (oom)");
  });

  it("leaves the wire value alone where it is already clear", () => {
    for (const word of ["running", "completed", "failed", "timeout", "cancelled"]) {
      expect(displayName(word)).toBe(word);
    }
  });

  it("colours still key off the WIRE value, not the display name", () => {
    // The trap in this change: pass a renamed word to `toneFor` and every badge
    // silently falls through to muted.
    expect(toneFor("started")).toBe("pending");
    expect(toneFor(displayName("started"))).not.toBe("pending");
  });
});
