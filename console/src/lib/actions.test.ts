import { describe, expect, it } from "vitest";
import { describeAction, describeActions } from "./actions";

/**
 * The token stays available even though it is never shown: it is what appears in
 * the path, so a person reading a URL and a person reading a button have to be
 * able to reconcile them.
 */

const BB = 2;
const at = (token: string, to = Number.NaN) => ({ token, to });

describe("naming an action", () => {
  it("gives the plain word for the unsized ones", () => {
    expect(describeAction(at("f"), BB).text).toBe("fold");
    expect(describeAction(at("x"), BB).text).toBe("check");
    expect(describeAction(at("c"), BB).text).toBe("call");
    expect(describeAction(at("A"), BB).text).toBe("all-in");
  });

  it("sizes from the server's raise-to, never from the token", () => {
    // The small blind's r2 adds 2 chips over its call: a raise TO 4 chips, 2bb.
    expect(describeAction(at("r2", 4), BB).text).toBe("raise to 2bb");
    expect(describeAction(at("b6", 6), BB).text).toBe("bet 3bb");
  });

  it("keeps whole blinds whole and halves to one decimal", () => {
    expect(describeAction(at("r3", 5), BB).text).toBe("raise to 2.5bb");
  });

  it("always carries the token, which is what the path uses", () => {
    expect(describeAction(at("r2", 4), BB).token).toBe("r2");
    expect(describeAction(at("f"), BB).token).toBe("f");
  });

  it("falls back to the raw token rather than inventing a size", () => {
    expect(describeAction(at("zzz", 4), BB).text).toBe("zzz");
    expect(describeAction(at("r2"), BB).text).toBe("r2");
  });

  it("shows chips when the big blind is not known yet", () => {
    // /run has not answered; a wrong conversion would be worse than none.
    expect(describeAction(at("r2", 4), 0).text).toBe("raise to 4bb");
  });

  it("labels the opening menu in the grid's order from the node's edges", () => {
    const edges = [
      at("f", 1),
      at("c", 2),
      at("r2", 4),
      at("r3", 5),
      at("r4", 6),
      at("r6", 8),
      at("r9", 11),
      at("A", 200),
    ];
    const menu = describeActions(["f", "c", "r2", "r3", "r4", "r6", "r9", "A"], edges, BB);

    expect(menu.map((m) => m.text)).toEqual([
      "fold",
      "call",
      "raise to 2bb",
      "raise to 2.5bb",
      "raise to 3bb",
      "raise to 4bb",
      "raise to 5.5bb",
      "all-in",
    ]);
  });
});
