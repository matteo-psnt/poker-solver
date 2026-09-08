import { describe, expect, it } from "vitest";
import { aggregate, bucketGroups, cellFor, classLabel, combosIn, summarise } from "./range";

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

/**
 * The rail's headline number, and the one the grid is worst at: a class gets
 * one square whether it holds 4 combos or 12, so the eye reads an offsuit-heavy
 * fold as smaller than it is. Weighting is the whole point, so it is asserted
 * against a case where weighted and unweighted disagree.
 */
describe("summarising the whole range", () => {
  const cell = (label: string, strategy: number[] | null, combos: number, untrained = 0) => ({
    label,
    strategy,
    combos,
    untrained,
  });
  const gridOf = (...cells: ReturnType<typeof cell>[]) => [cells];

  it("weights a class by how many combos it holds", () => {
    // Unweighted this would be 50/50; AKo holds three times what AKs does.
    const summary = summarise(gridOf(cell("AKs", [1, 0], 4), cell("AKo", [0, 1], 12)), 2);
    expect(summary?.strategy).toEqual([0.25, 0.75]);
    expect(summary?.trained).toBe(16);
  });

  it("leaves untrained combos out of the mean and reports them beside it", () => {
    const summary = summarise(gridOf(cell("AA", [1, 0], 6, 2), cell("KK", null, 6, 6)), 2);
    expect(summary?.strategy).toEqual([1, 0]);
    expect(summary?.trained).toBe(4);
    expect(summary?.untrained).toBe(8);
  });

  it("ignores combos the board blocks entirely", () => {
    const summary = summarise(gridOf(cell("AA", [1, 0], 6), cell("KK", null, 0)), 2);
    expect(summary?.trained).toBe(6);
    expect(summary?.untrained).toBe(0);
  });

  it("is null rather than zero when nothing here was trained", () => {
    expect(summarise(gridOf(cell("AA", null, 6, 6)), 2)).toBeNull();
    expect(summarise([], 2)).toBeNull();
  });
});

/**
 * The drill-down exists because the cell above it is an AVERAGE, and the
 * average is built to hide exactly the thing a player needs: whether the
 * solver treats a hand's suits alike. These pin the three ways it can answer.
 */
describe("the combos behind one cell", () => {
  const AKS = {
    // Deck order, which is how the wire spells a combo -- king before ace.
    combos: ["KsAs", "KhAh", "KdAd", "KcAc", "KsAd"],
    comboBuckets: [7, 7, 7, 9, 3],
    buckets: {
      "3": { trained: true, strategy: [1, 0] },
      "7": { trained: true, strategy: [0.2, 0.8] },
      "9": { trained: true, strategy: [0.6, 0.4] },
    },
    actionCount: 2,
  };

  it("takes only the combos of the class asked for", () => {
    expect(combosIn("AKs", AKS).map((row) => row.combo)).toEqual(["KsAs", "KhAh", "KdAd", "KcAc"]);
    expect(combosIn("AKo", AKS).map((row) => row.combo)).toEqual(["KsAd"]);
  });

  it("puts the high card first, whatever order the wire used", () => {
    // `KsAs` rendered verbatim reads as a king-high hand under a label saying
    // AKs -- the chart contradicting itself two lines apart.
    expect(combosIn("AKs", AKS)[0]?.cards).toEqual(["As", "Ks"]);
  });

  it("orders by suit, so one hand lists the same way in every spot", () => {
    expect(combosIn("AKs", AKS).map((row) => row.cards[0])).toEqual(["As", "Ah", "Ad", "Ac"]);
  });

  it("keeps a blocked combo and marks it, rather than dropping it", () => {
    // Dropping it leaves "3 of 4 playable" with nothing saying which one, in
    // the one place on the page with room to say.
    const rows = combosIn("AKs", { ...AKS, comboBuckets: [-1, 7, 7, 9, 3] });
    expect(rows).toHaveLength(4);
    expect(rows[0]).toMatchObject({ combo: "KsAs", blocked: true, strategy: null });
  });

  it("separates never-trained from blocked", () => {
    const rows = combosIn("AKs", {
      ...AKS,
      buckets: { ...AKS.buckets, "9": { trained: false, strategy: null } },
    });
    expect(rows[3]).toMatchObject({ blocked: false, untrained: true, strategy: null });
  });

  it("groups the combos the solver treats alike, ignoring what the board blocks", () => {
    // The answer to "do the suits matter here". Preflop it is one group -- the
    // abstraction is 169 buckets for 169 classes, so four identical bars would
    // imply a distinction the solver never made.
    const split = bucketGroups(combosIn("AKs", AKS));
    expect(split.size).toBe(2);
    // Numbered by first appearance, not by bucket id: 7 is seen before 9.
    expect([split.get(7), split.get(9)]).toEqual([1, 2]);
    expect(bucketGroups(combosIn("AKs", { ...AKS, comboBuckets: [7, 7, 7, 7, 3] })).size).toBe(1);
    // A blocked combo is in no bucket and must not split a group that agrees.
    expect(bucketGroups(combosIn("AKs", { ...AKS, comboBuckets: [7, 7, 7, -1, 3] })).size).toBe(1);
  });
});
