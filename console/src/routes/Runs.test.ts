import { describe, expect, it } from "vitest";
import { verdictFor } from "./Runs";

/**
 * A run's `status` is written by the training process itself, so it records
 * what a LIVING process did and cannot record how an attempt died. Four runs
 * on this share have claimed `running` since 07-31 with nothing executing.
 *
 * These pin the three answers apart, including the one that must stay hedged.
 */
const LIVE = new Set(["run-live"]);
const WITH_LEGS = new Set(["run-live", "run-dead"]);

describe("a run is only running if Batch says a leg is live", () => {
  it("confirms running when a leg is executing", () => {
    expect(verdictFor("running", "run-live", LIVE, WITH_LEGS)?.label).toBe("running");
  });

  it("calls it abandoned when legs exist and none is live", () => {
    // Evidence: the run has leg records, so the observer half exists and says
    // nothing is running.
    const verdict = verdictFor("running", "run-dead", LIVE, WITH_LEGS);
    expect(verdict?.label).toBe("abandoned");
    expect(verdict?.tone).toBe("warn");
  });

  it("HEDGES when the run predates the leg log", () => {
    // No leg records at all — there is no observer half to reconcile against,
    // so this is inference, not a finding. Asserting it would be a claim the
    // data cannot support.
    const verdict = verdictFor("running", "run-ancient", LIVE, WITH_LEGS);
    expect(verdict?.label).toBe("abandoned?");
    expect(verdict?.tone).toBe("muted");
  });

  it("leaves a settled status alone", () => {
    for (const status of ["completed", "failed", null]) {
      expect(verdictFor(status, "run-dead", LIVE, WITH_LEGS)).toBeNull();
    }
  });
});
