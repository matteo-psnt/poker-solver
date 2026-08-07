import { describe, expect, it } from "vitest";
import { describeAction, describeActions } from "./actions";

/**
 * The token stays available even though it is never shown: it is what appears in
 * the path, so a person reading a URL and a person reading a button have to be
 * able to reconcile them.
 */

const BB = 2;

describe("naming an action", () => {
  it("gives the plain word for the unsized ones", () => {
    expect(describeAction("f", BB).text).toBe("fold");
    expect(describeAction("x", BB).text).toBe("check");
    expect(describeAction("c", BB).text).toBe("call");
    expect(describeAction("A", BB).text).toBe("all-in");
  });

  it("converts chips to big blinds", () => {
    expect(describeAction("r4", BB).text).toBe("raise to 2bb");
    expect(describeAction("b6", BB).text).toBe("bet 3bb");
  });

  it("says raise TO, because the amount is the total not the increment", () => {
    // The distinction that makes a 3-bet read as 9 rather than as 5.
    expect(describeAction("r9", BB).text).toBe("raise to 4.5bb");
  });

  it("keeps whole blinds whole and halves to one decimal", () => {
    expect(describeAction("r4", BB).text).toContain("2bb");
    expect(describeAction("r5", BB).text).toContain("2.5bb");
  });

  it("always carries the token, which is what the path uses", () => {
    expect(describeAction("r4", BB).token).toBe("r4");
    expect(describeAction("f", BB).token).toBe("f");
  });

  it("falls back to the raw token rather than inventing a size", () => {
    expect(describeAction("zzz", BB).text).toBe("zzz");
    expect(describeAction("rXY", BB).text).toBe("rXY");
  });

  it("shows chips when the big blind is not known yet", () => {
    // /run has not answered; a wrong conversion would be worse than none.
    expect(describeAction("r4", 0).text).toBe("raise to 4bb");
  });

  it("labels a whole menu in order", () => {
    const menu = describeActions(["f", "c", "r4", "r6", "r9", "A"], BB);

    expect(menu.map((m) => m.text)).toEqual([
      "fold",
      "call",
      "raise to 2bb",
      "raise to 3bb",
      "raise to 4.5bb",
      "all-in",
    ]);
  });
});
