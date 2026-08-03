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
