import type { LegRow } from "@/api/schemas";
import type { Tone } from "@/components/StatusBadge";
import { toneFor } from "@/components/StatusBadge";
import { instant } from "@/lib/format";

/**
 * A run, drawn as the thing it actually is: a lineage assembled over time.
 *
 * A run is not one job. It is many legs, across many days and several daily
 * Batch jobs, each picking up the last one's checkpoint — and a table of legs
 * hides exactly the facts that matter about that shape. Read down a column of
 * timestamps and you cannot see that two attempts overlapped, that a leg died
 * eight hours in, or that nothing touched the run for two days between rungs.
 * Laid out against a shared axis, all three are the first thing you notice.
 *
 * The bars are the relation. Everything here is a pure function of the leg
 * rows so the geometry can be tested without rendering anything.
 */

export type Bar = {
  key: string;
  taskId: string;
  label: string;
  tone: Tone;
  /** Percentages of the whole span, for CSS `left`/`width`. */
  leftPct: number;
  widthPct: number;
  running: boolean;
};

export type Timeline = { bars: Bar[]; from: number; to: number };

/**
 * Narrow enough to still be visible.
 *
 * A 40-second leg inside a three-day run is ~0.01% wide and would render as
 * nothing — and a leg that died instantly is the single most interesting bar
 * on the chart, so vanishing is the worst possible failure here.
 */
const MIN_WIDTH_PCT = 1.2;

export function timelineBars(legs: LegRow[], now: number): Timeline | null {
  const dated = legs
    .map((leg) => ({ leg, start: instant(leg.started_at), end: instant(leg.ended_at) }))
    .filter((row): row is { leg: LegRow; start: number; end: number | null } => row.start != null)
    .sort((a, b) => a.start - b.start);

  const first = dated[0];
  if (first === undefined) return null;

  const from = first.start;
  // An unfinished leg extends the axis to now; otherwise a run whose only leg
  // is still going would have zero width.
  const to = Math.max(...dated.map((row) => row.end ?? now), from + 1);
  const total = to - from;

  const bars = dated.map(({ leg, start, end }, index) => {
    const finish = end ?? now;
    // Clamp the left edge FIRST, then bound the width by the CLAMPED edge.
    // Bounding by the raw one undoes the minimum exactly where it is needed:
    // the last leg of a long run starts at ~99.99%, leaving 0.01% of room, so
    // the two-second bar this constant exists to keep visible vanished anyway.
    const leftPct = Math.min(((start - from) / total) * 100, 100 - MIN_WIDTH_PCT);
    const rawWidth = ((finish - start) / total) * 100;
    return {
      key: `${leg.task_id}-${leg.attempt ?? index}`,
      taskId: leg.task_id,
      label: leg.what || leg.op || "leg",
      tone: toneFor(leg.cause),
      leftPct,
      widthPct: Math.min(Math.max(rawWidth, MIN_WIDTH_PCT), 100 - leftPct),
      running: end == null,
    };
  });

  return { bars, from, to };
}
