/**
 * Resolve a CSS custom property to a concrete colour.
 *
 * Needed because <canvas> does not understand them. `ctx.strokeStyle =
 * "var(--fg-faint)"` is not an error — the assignment is silently ignored and
 * the previous value stands, which is black. That is how a chart ends up with
 * black axis labels on a dark panel while every DOM element on the same page
 * themes correctly.
 *
 * Read at draw time rather than cached at module load: the value depends on
 * which theme class is on the root, and a module-level constant would be
 * whatever the theme was when the bundle first evaluated.
 */
export function token(name: string, fallback: string): string {
  if (typeof window === "undefined") return fallback;
  const value = getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  return value || fallback;
}

/** The chart palette, resolved. Fallbacks are the dark theme's values. */
export function chartTheme() {
  return {
    // MUTED, not faint. Faint is for a label sitting beside the value that
    // carries the meaning; an axis tick IS the value, and at 10px on a dark
    // panel the faint weight measures 3.7:1.
    axis: token("--fg-muted", "#a1a1aa"),
    grid: token("--border", "#232329"),
    series: "#3b82f6",
    seriesFill: "rgba(59,130,246,0.14)",
  };
}
