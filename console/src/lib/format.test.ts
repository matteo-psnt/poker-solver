import { describe, expect, it } from "vitest";
import { legLabel, runLabel, span } from "./format";

/**
 * These mirror `run_token` and `task_id` in `src/interfaces/cloud/spec.py`.
 *
 * If the Python rule changes and these do not, the console and Batch start
 * showing different words for one leg — which is exactly the confusion the
 * labels were introduced to remove.
 */
describe("runLabel", () => {
  it("drops the timestamp nobody reads, keeping config and discriminator", () => {
    expect(runLabel("run-production-025433-1095")).toBe("production-1095");
  });

  it("leaves a two-segment id alone rather than inventing a shape", () => {
    expect(runLabel("run-20260802_201939-ee77cb")).toBe("20260802_201939-ee77cb");
  });

  it("keeps ids distinct that differ only in the tail", () => {
    expect(runLabel("run-ochs_dose_r100-105223-25247")).not.toBe(
      runLabel("run-ochs_dose_r100-105241-16780"),
    );
  });
});

describe("legLabel", () => {
  it("strips the queue-time suffix, leaving what the leg does", () => {
    expect(legLabel("score-production-1095-150M-seed7-090456-18475")).toBe(
      "score-production-1095-150M-seed7",
    );
  });

  it("keeps three same-checkpoint evals telling themselves apart", () => {
    const ids = ["seed7-090456-18475", "seed13-090501-1849", "seed29-090506-24091"].map((tail) =>
      legLabel(`score-production-1095-150M-${tail}`),
    );
    expect(new Set(ids).size).toBe(3);
  });

  it("reduces a pre-label id to its run, which is all it ever carried", () => {
    expect(legLabel("run-production-025433-1095-090456-18475")).toBe("run-production-025433-1095");
  });

  it("strips only the trailing queue suffix when the label itself ends in digits", () => {
    // The strip is `-<6 digits>-<digits>` anchored at the END, and a label can
    // legitimately end that way too. Anchoring is what keeps this from eating
    // real content one segment at a time.
    expect(legLabel("score-x-250000-3-090456-18475")).toBe("score-x-250000-3");
  });
});

describe("span", () => {
  const now = Date.parse("2026-08-04T12:00:00Z");

  it("measures a finished interval from its two ends", () => {
    expect(span("2026-08-04T09:46:00Z", "2026-08-04T12:00:00Z")).toBe("2h 14m");
  });

  it("runs an open interval to now, so 'running 2h' is not 'unknown'", () => {
    expect(span("2026-08-04T10:00:00Z", null, now)).toBe("2h 0m");
  });

  it("is a dash when there is nothing to measure, not a zero", () => {
    expect(span(null, "2026-08-04T12:00:00Z")).toBe("—");
    expect(span("2026-08-04T10:00:00Z", null)).toBe("—");
  });

  it("never reports a negative interval from clock skew across machines", () => {
    expect(span("2026-08-04T12:00:00Z", "2026-08-04T11:00:00Z")).toBe("0s");
  });
});
