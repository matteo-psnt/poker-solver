import { describe, expect, it } from "vitest";
import { deck, parseCard, readBoard, writeBoard } from "./cards";

describe("parseCard", () => {
  it("reads the wire spelling", () => {
    expect(parseCard("Ah")).toMatchObject({ text: "Ah", rank: "A", suit: "h", glyph: "♥" });
    expect(parseCard("Tc")).toMatchObject({ rank: "T", suit: "c", glyph: "♣" });
  });

  it("normalises case, because a person typing will not", () => {
    expect(parseCard("aH")?.text).toBe("Ah");
    expect(parseCard("2C")?.text).toBe("2c");
  });

  it("refuses what is not a card in this deck", () => {
    expect(parseCard("1h")).toBeNull();
    expect(parseCard("Ax")).toBeNull();
    expect(parseCard("A")).toBeNull();
    expect(parseCard("Ahh")).toBeNull();
  });
});

describe("readBoard", () => {
  it("takes the same shapes parse_board takes", () => {
    expect(readBoard("2c7d9h").map((card) => card.text)).toEqual(["2c", "7d", "9h"]);
    expect(readBoard("2c 7d 9h").map((card) => card.text)).toEqual(["2c", "7d", "9h"]);
    expect(readBoard("2c,7d,9h").map((card) => card.text)).toEqual(["2c", "7d", "9h"]);
  });

  it("ignores a trailing half-card, so the slots do not flicker mid-type", () => {
    expect(readBoard("2c7d9").map((card) => card.text)).toEqual(["2c", "7d"]);
  });

  it("drops what it cannot read rather than throwing — the server refuses", () => {
    expect(readBoard("zz7d").map((card) => card.text)).toEqual(["7d"]);
    expect(readBoard("")).toEqual([]);
  });

  it("round trips through writeBoard", () => {
    expect(writeBoard(readBoard("As Kd 7c"))).toBe("AsKd7c");
  });
});

describe("deck", () => {
  it("is 52 distinct cards", () => {
    const cards = deck();
    expect(cards).toHaveLength(52);
    expect(new Set(cards.map((card) => card.text)).size).toBe(52);
  });

  it("lays out four rows of thirteen, aces first", () => {
    const cards = deck();
    expect(cards[0]?.text).toBe("As");
    expect(cards[12]?.text).toBe("2s");
    expect(cards[13]?.text).toBe("Ah");
  });
});
