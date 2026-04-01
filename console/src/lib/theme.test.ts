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

/**
 * The check that would have caught the real failure.
 *
 * The test above proves `chartTheme()` returns concrete colours. It says
 * nothing about whether the chart USES them — and it did not: a `var(--…)`
 * literal survived in the uPlot config, the page rendered black axis labels,
 * and every test still passed. Assert on the source that reaches the canvas.
 */
describe("no CANVAS config references a CSS variable", () => {
  // Scoped to StepChart on purpose. Recharts (RunDetail) renders SVG, where a
  // presentation attribute DOES resolve `var(--…)` — those charts theme
  // correctly and should keep the maintainable form. The rule is about canvas,
  // not about `var()`.
  it("StepChart hands uPlot resolved colours only", async () => {
    const fs = await import("node:fs/promises");
    // From the project root, not `import.meta.url`: under the jsdom transform
    // that is not a file: URL and the read throws.
    const source = await fs.readFile("src/components/StepChart.tsx", "utf-8");
    // Comments explain the bug by name, so strip them before looking.
    const code = source.replace(/\/\*[\s\S]*?\*\//g, "").replace(/\/\/.*$/gm, "");
    expect(code, "a var() literal in a canvas config renders as black").not.toMatch(/var\(--/);
  });
});
