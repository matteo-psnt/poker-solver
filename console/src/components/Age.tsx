import { since } from "@/lib/format";
import { cn } from "@/lib/utils";

/**
 * How old the data is, always shown.
 *
 * A dashboard that renders stale numbers as current is worse than one that
 * admits it. Past twice the refresh interval the badge turns amber — the point
 * where "this is live" stops being true.
 */
export function Age({ at, staleAfterMs }: { at: number | null; staleAfterMs: number }) {
  if (!at) return <span className="text-[var(--fg-faint)]">—</span>;
  const stale = Date.now() - at > staleAfterMs;
  const iso = new Date(at).toISOString();
  return (
    <span
      title={iso}
      className={cn("tnum text-[11px]", stale ? "text-amber-400" : "text-[var(--fg-faint)]")}
    >
      {since(iso)}
    </span>
  );
}
