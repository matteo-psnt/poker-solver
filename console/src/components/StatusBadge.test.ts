import { describe, expect, it } from "vitest";
import { displayName, phaseTone, toneFor } from "./StatusBadge";

/**
 * What this console still decides about a task's appearance.
 *
 * It used to decide more. `taskTone`, `taskOutcome`, `shortState` and
 * `exitMeaning` classified Batch's raw enum strings here, and the tests for
 * them lived here too — a browser test suite that knew exit 137 means the OOM
 * killer. That classification is the server's now
 * (`src/shared/task_states.py`), and its tests moved with it to
 * `tests/shared/test_task_states.py`, which is also where the two deaths that
 * must stay apart (124 vs 137) are pinned.
 *
 * What is left is presentation: which phase pulses, which word is shown, and
 * the trap that separates the two.
 */
describe("a finished task's colour comes from its OUTCOME, not the phase", () => {
  it("finished says stopped, so it is never green on its own", () => {
    // The original bug: Batch's `completed` means finished, not succeeded, so a
    // badge coloured on the phase alone painted a crashed task green.
    expect(phaseTone("finished", "done")).toBe("ok");
    expect(phaseTone("finished", "failed")).toBe("bad");
  });

  it("a cancelled task is muted and a timed-out one is a warning", () => {
    expect(phaseTone("finished", "cancelled")).toBe("muted");
    expect(phaseTone("finished", "timed out")).toBe("warn");
  });

  it("queued does not pulse like live work", () => {
    // A task waiting for a node is not working, and a frozen one never will be.
    expect(phaseTone("queued", null)).toBe("pending");
  });

  it("running and starting are live", () => {
    expect(phaseTone("running", null)).toBe("live");
    expect(phaseTone("starting", null)).toBe("live");
  });

  it("an outcome the server has not sent yet is muted, not a guess", () => {
    expect(phaseTone("finished", null)).toBe("muted");
    expect(phaseTone("unknown", null)).toBe("muted");
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

  it("reads the server's outcome words too, since they say the same things", () => {
    // `timeout` (the task log) and `timed out` (a Batch outcome) are the same
    // fact from two records, and a panel showing both must not colour them
    // differently.
    expect(toneFor("done")).toBe(toneFor("completed"));
    expect(toneFor("timed out")).toBe(toneFor("timeout"));
  });
});

describe("displayed names are not always the names on the wire", () => {
  it("renames the task causes that hid what they meant", () => {
    expect(displayName("started")).toBe("unresolved");
    expect(displayName("killed")).toBe("killed (oom)");
  });

  it("no longer renames Batch's words, because the server already did", () => {
    // `active` -> `queued` used to happen here. It happens in `phase_of` now,
    // and a second rename on this side would be a vocabulary to keep in step.
    expect(displayName("queued")).toBe("queued");
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
