import { describe, expect, it } from "vitest";
import { exitMeaning, shortState, taskTone, toneFor } from "./StatusBadge";

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

describe("the leg log's causes, where the word IS the outcome", () => {
  it("separates the three ways a leg stops being alive", () => {
    expect(toneFor("completed")).toBe("ok");
    expect(toneFor("failed")).toBe("bad");
    expect(toneFor("timeout")).toBe("warn");
    expect(toneFor("cancelled")).toBe("muted");
  });

  it("killed is a failure and partial is a warning", () => {
    expect(toneFor("killed")).toBe("bad");
    expect(toneFor("partial")).toBe("warn");
  });

  it("an unresolved leg is pending, not fine", () => {
    // `started` with no terminal record: the trap never ran. Showing it as
    // success would hide exactly the deaths the leg log exists to catch.
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
