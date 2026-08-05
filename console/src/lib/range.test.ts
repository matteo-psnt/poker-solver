import { describe, expect, it } from "vitest";
import { aggregate, cellFor, classLabel } from "./range";

/**
 * The aggregation is the only real logic on the client, and its job is to keep
 * three states apart: a hand you cannot hold, a hand the solver never learned,
 * and a hand it has a strategy for. Collapsing any two of those is the most
 * misleading thing this page could do, so each is asserted directly.
 */

describe("placing a combo in the grid", () => {
  it("puts pairs on the diagonal", () => {
    expect(cellFor("AsAd")).toEqual({ row: 0, col: 0 });
    expect(classLabel(0, 0)).toBe("AA");
  });

  it("puts suited hands above the diagonal and offsuit below", () => {
    expect(cellFor("AsKs")).toEqual({ row: 0, col: 1 });
    expect(cellFor("AsKd")).toEqual({ row: 1, col: 0 });
    expect(classLabel(0, 1)).toBe("AKs");
    expect(classLabel(1, 0)).toBe("AKo");
  });

  it("does not care which card came first", () => {
    expect(cellFor("KsAs")).toEqual(cellFor("AsKs"));
  });

  it("returns null rather than guessing at nonsense", () => {
    expect(cellFor("")).toBeNull();
    expect(cellFor("XxYy")).toBeNull();
  });
});

describe("aggregating combos into cells", () => {
  const trained = { trained: true, strategy: [0.25, 0.75] };
  const untrained = { trained: false, strategy: null };

  it("averages over the combos of a class", () => {
    const cells = aggregate({
      combos: ["AsAd", "AsAh"],
      comboBuckets: [1, 2],
      buckets: {
        "1": { trained: true, strategy: [0, 1] },
        "2": { trained: true, strategy: [1, 0] },
      },
      actionCount: 2,
    });

    expect(cells[0]?.[0]?.strategy).toEqual([0.5, 0.5]);
    expect(cells[0]?.[0]?.combos).toBe(2);
  });

  it("excludes blocked combos from the denominator entirely", () => {
    const cells = aggregate({
      combos: ["AsAd", "AsAh"],
      comboBuckets: [1, -1],
      buckets: { "1": trained },
      actionCount: 2,
    });

    expect(cells[0]?.[0]?.combos).toBe(1);
    expect(cells[0]?.[0]?.strategy).toEqual([0.25, 0.75]);
  });

  it("reports a fully blocked class as absent, not as a zero row", () => {
    const cells = aggregate({
      combos: ["AsAd"],
      comboBuckets: [-1],
      buckets: {},
      actionCount: 2,
    });

    expect(cells[0]?.[0]?.combos).toBe(0);
    expect(cells[0]?.[0]?.strategy).toBeNull();
  });

  it("distinguishes untrained from blocked", () => {
    const cells = aggregate({
      combos: ["AsAd"],
      comboBuckets: [7],
      buckets: { "7": untrained },
      actionCount: 2,
    });

    // Holdable, so it counts; never learned, so there is nothing to show.
    expect(cells[0]?.[0]?.combos).toBe(1);
    expect(cells[0]?.[0]?.untrained).toBe(1);
    expect(cells[0]?.[0]?.strategy).toBeNull();
  });

  it("averages a partly-trained class over only its trained combos", () => {
    const cells = aggregate({
      combos: ["AsAd", "AsAh"],
      comboBuckets: [1, 7],
      buckets: { "1": trained, "7": untrained },
      actionCount: 2,
    });

    expect(cells[0]?.[0]?.strategy).toEqual([0.25, 0.75]);
    expect(cells[0]?.[0]?.untrained).toBe(1);
    expect(cells[0]?.[0]?.combos).toBe(2);
  });

  it("always returns a full 13x13 whatever it was given", () => {
    const cells = aggregate({
      combos: [],
      comboBuckets: [],
      buckets: {},
      actionCount: 2,
    });

    expect(cells).toHaveLength(13);
    expect(cells.every((row) => row.length === 13)).toBe(true);
  });
});
