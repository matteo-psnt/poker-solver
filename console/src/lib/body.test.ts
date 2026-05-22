import { describe, expect, it } from "vitest";
import { given, int, overrides, rungs } from "./body";

describe("given", () => {
  it("drops a field the operator left blank", () => {
    expect(given({ config: "production", run: "" })).toEqual({ config: "production" });
  });

  it("drops whitespace, which is what a cleared field leaves behind", () => {
    expect(given({ run: "   " })).toEqual({});
  });

  it("keeps a zero, which is a value someone typed", () => {
    expect(given({ workers: 0 })).toEqual({ workers: 0 });
  });

  it("drops an unchecked guard rather than sending false", () => {
    // The command's own `store_true` default is what should answer. Sending
    // `false` would work identically today and stop working the moment a flag
    // acquires a non-False default.
    expect(given({ force: false, delete: true })).toEqual({ delete: true });
  });

  it("drops an empty override list", () => {
    expect(given({ sets: [] })).toEqual({});
  });
});

describe("int", () => {
  it("reads a grouped number the way it was typed", () => {
    expect(int("25,000,000")).toBe(25_000_000);
  });

  it("is undefined for a blank field, not zero", () => {
    // 0 is a value a command would act on; blank means no argument at all.
    expect(int("")).toBeUndefined();
    expect(int("  ")).toBeUndefined();
  });

  it("refuses something that is not a whole number", () => {
    expect(int("abc")).toBeUndefined();
    expect(int("1.5")).toBeUndefined();
  });
});

describe("overrides", () => {
  it("takes one KEY=VALUE per line", () => {
    expect(overrides("solver__pruning=true\ntraining__batch=64")).toEqual([
      "solver__pruning=true",
      "training__batch=64",
    ]);
  });

  it("ignores a line that is not an override", () => {
    // Losing a nine-field form to a stray blank line is the worse trade.
    expect(overrides("solver__pruning=true\n\n  \nnonsense")).toEqual(["solver__pruning=true"]);
  });
});

describe("rungs", () => {
  it("normalises spacing and underscores into what --at parses", () => {
    expect(rungs("10_000_000, 20000000")).toBe("10000000,20000000");
  });

  it("is empty for a blank field, which means the latest checkpoint", () => {
    expect(rungs(" , ")).toBe("");
  });
});
