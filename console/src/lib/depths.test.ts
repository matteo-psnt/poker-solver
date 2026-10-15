import { describe, expect, it } from "vitest";
import { kindMix, kindOf, totalVariation } from "./depths";

describe("kindOf", () => {
  it("reads the wire tokens", () => {
    expect(["f", "c", "x", "r2", "b4", "A"].map(kindOf)).toEqual([
      "fold",
      "call",
      "call",
      "raise",
      "raise",
      "allin",
    ]);
  });
});

describe("kindMix", () => {
  it("sums every raise size into one kind", () => {
    const mix = kindMix([0.1, 0.2, 0.3, 0.3, 0.1], ["f", "c", "r2", "r3", "A"]);
    expect(mix.raise).toBeCloseTo(0.6);
    expect(mix.allin).toBeCloseTo(0.1);
  });

  it("is empty for a hand with no strategy", () => {
    expect(kindMix(null, ["f"])).toEqual({ fold: 0, call: 0, raise: 0, allin: 0 });
  });
});

describe("totalVariation", () => {
  it("is zero for the same mix and one for disjoint ones", () => {
    const fold = { fold: 1, call: 0, raise: 0, allin: 0 };
    const raise = { fold: 0, call: 0, raise: 1, allin: 0 };
    expect(totalVariation(fold, fold)).toBe(0);
    expect(totalVariation(fold, raise)).toBe(1);
  });
});
