/**
 * Every number the console shows passes through here.
 *
 * Not a convenience: the same value formatted two ways on two pages reads as
 * two different measurements. Iterations are grouped and never exponential;
 * durations are never raw seconds; run ids are truncated in the MIDDLE, because
 * they share a long prefix (`run-production-…`) and differ only at the end.
 */

const RELATIVE = new Intl.RelativeTimeFormat("en", { numeric: "auto", style: "narrow" });

export function count(value: number | null | undefined): string {
  return value == null ? "—" : value.toLocaleString("en-US");
}

export function percent(value: number | null | undefined, digits = 1): string {
  return value == null ? "—" : `${(value * 100).toFixed(digits)}%`;
}

export function rate(value: number | null | undefined, unit = "it/s"): string {
  return value == null ? "—" : `${Math.round(value).toLocaleString("en-US")} ${unit}`;
}

/** `2122.2 ± 12.0` — the interval is part of the measurement, not a footnote. */
export function mbb(value: number | null | undefined, error?: number | null): string {
  if (value == null) return "—";
  const point = value.toFixed(1);
  return error == null ? point : `${point} ± ${error.toFixed(1)}`;
}

export function duration(seconds: number | null | undefined): string {
  if (seconds == null) return "—";
  const s = Math.max(0, Math.round(seconds));
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  if (m < 60) return `${m}m ${s % 60}s`;
  const h = Math.floor(m / 60);
  if (h < 24) return `${h}h ${m % 60}m`;
  return `${Math.floor(h / 24)}d ${h % 24}h`;
}

/** Epoch millis, or null when absent/unparseable — so callers can branch once. */
export function instant(iso: string | null | undefined): number | null {
  if (!iso) return null;
  const parsed = Date.parse(iso);
  return Number.isNaN(parsed) ? null : parsed;
}

/**
 * How long something took, from its two ends.
 *
 * `now` closes an open interval, so a task still running reports how long it has
 * been going rather than a dash. That distinction is the point: "running 2h"
 * and "took 2h" are different facts and a blank cell conflates them with
 * "unknown".
 */
export function span(
  from: string | null | undefined,
  to: string | null | undefined,
  now?: number,
): string {
  const start = instant(from);
  if (start == null) return "—";
  const end = instant(to) ?? now;
  if (end == null) return "—";
  return duration(Math.max(0, end - start) / 1000);
}

/** `09:27` in the viewer's zone — the wall-clock a task started, for correlating. */
export function clock(iso: string | null | undefined): string {
  const at = instant(iso);
  if (at == null) return "—";
  return new Date(at).toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

/** Relative, because "is this current?" is the only question being asked. */
export function since(iso: string | null | undefined, now = Date.now()): string {
  if (!iso) return "—";
  const then = Date.parse(iso);
  if (Number.isNaN(then)) return "—";
  const seconds = Math.round((then - now) / 1000);
  const abs = Math.abs(seconds);
  if (abs < 60) return RELATIVE.format(seconds, "second");
  if (abs < 3600) return RELATIVE.format(Math.round(seconds / 60), "minute");
  if (abs < 86400) return RELATIVE.format(Math.round(seconds / 3600), "hour");
  return RELATIVE.format(Math.round(seconds / 86400), "day");
}

/** Keeps both the config and the discriminator visible. */
export function shortId(id: string, head = 18, tail = 8): string {
  if (id.length <= head + tail + 1) return id;
  return `${id.slice(0, head)}…${id.slice(-tail)}`;
}

/**
 * `run-production-025433-1095` → `production-1095`.
 *
 * Ids are long, share a prefix, and differ only at the END, so a column of them
 * reads as one repeated string with noise on the tail. The config and the
 * discriminator are the two parts that identify a run; the timestamp between
 * them is the part nobody reads.
 *
 * Mirrors `run_token` in `src/interfaces/cloud/spec.py` — the SAME rule that
 * now builds task ids — so what the console shows and what Batch shows are the
 * same words. Always pair it with the full id in a `title`: this is a display
 * form, never an identifier to copy.
 */
export function runLabel(id: string): string {
  const stem = id.replace(/^run-/, "");
  const parts = stem.split("-").filter(Boolean);
  return parts.length > 2 ? `${parts[0]}-${parts[parts.length - 1]}` : stem;
}

/**
 * `score-production-1095-150M-seed7-090456-18475` → `score-production-1095-150M-seed7`.
 *
 * A task id is `<label>-<HHMMSS>-<nonce>` (`task_id` in spec.py). The label
 * says what the task does; the suffix exists only to keep two submissions in
 * one second apart, and it is what made a column of ids look identical.
 *
 * Tasks queued before labels carried the op strip down to a bare run id — which
 * is honest, because that is genuinely all those ids ever recorded.
 */
export function taskLabel(taskId: string): string {
  return taskId.replace(/-\d{6}-\d+$/, "");
}
