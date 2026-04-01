import { describe, expect, it } from "vitest";
import { chartTheme, token } from "./theme";

/**
 * The bug this exists for: a `var(--…)` string handed to a canvas is not an
 * error. `ctx.strokeStyle = "var(--fg-faint)"` is silently ignored and the
 * previous value stands — black — so a chart renders black axis labels on a
 * dark panel while every DOM element beside it themes correctly. Nothing
 * throws, and nothing but looking at it reveals the problem.
 */
describe("theme tokens reach the canvas as real colours", () => {
  it("resolves a declared custom property", () => {
    document.documentElement.style.setProperty("--fg-faint", "#abcdef");
    expect(token("--fg-faint", "#000")).toBe("#abcdef");
  });

  it("falls back when the property is not declared", () => {
    expect(token("--nothing-declares-this", "#123456")).toBe("#123456");
  });

  it("never returns a var() reference", () => {
    for (const [key, value] of Object.entries(chartTheme())) {
      expect(value, `${key} must be a concrete colour, not a CSS reference`).not.toMatch(/var\(/);
      expect(value.length, `${key} is empty`).toBeGreaterThan(0);
    }
  });
});
