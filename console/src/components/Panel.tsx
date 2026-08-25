import { cn } from "@/lib/utils";
import { RefreshCw } from "lucide-react";
import { Children, type ReactNode } from "react";
import { Age } from "./Age";

/**
 * The unit every page is built from, and the five states it can be in.
 *
 * The one that matters is `error`: the header rule turns red, the reason is
 * shown verbatim, and **the last good content stays visible, dimmed**. During
 * an incident the stale answer is usually still the useful one, and blanking it
 * destroys the only information on the screen.
 */
export function Panel({
  title,
  aside,
  updatedAt,
  staleAfterMs,
  error,
  loading,
  empty,
  onRefresh,
  refreshing,
  children,
}: {
  title: string;
  /** Controls or context belonging to the panel as a whole, beside its name. */
  aside?: ReactNode;
  updatedAt?: number | null;
  staleAfterMs?: number;
  error?: string | null;
  loading?: boolean;
  empty?: string | null;
  onRefresh?: () => void;
  refreshing?: boolean;
  children?: ReactNode;
}) {
  // `children != null` was true for `false` (a guard that did not fire) and for
  // any array (two or more child expressions), so `empty`/`loading` never
  // rendered in 13 panels -- a fresh share showed a titled panel with a blank
  // body. Children.toArray discards null/undefined/booleans and keeps elements.
  const hasContent = Children.toArray(children).length > 0;
  return (
    <section
      className={cn(
        "rounded-md border bg-[var(--panel)]",
        error ? "border-red-500/40" : "border-[var(--border)]",
      )}
    >
      <header
        className={cn(
          "flex items-center gap-3 border-b px-3 py-1.5",
          error ? "border-red-500/40" : "border-[var(--border)]",
        )}
      >
        <h2 className="shrink-0 font-mono text-[11px] tracking-widest text-[var(--fg-muted)] uppercase">
          {title}
        </h2>
        {aside && <div className="min-w-0">{aside}</div>}
        <div className="ml-auto flex shrink-0 items-center gap-2">
          <Age at={updatedAt ?? null} staleAfterMs={staleAfterMs ?? 60_000} />
          {onRefresh && (
            <button
              type="button"
              onClick={onRefresh}
              aria-label={`Refresh ${title}`}
              className="rounded p-1 text-[var(--fg-faint)] hover:bg-white/5 hover:text-[var(--fg)]"
            >
              <RefreshCw className={cn("size-3", refreshing && "motion-safe:animate-spin")} />
            </button>
          )}
        </div>
      </header>

      {error && (
        <p className="border-b border-red-500/25 bg-red-500/5 px-3 py-2 font-mono text-[12px] text-red-400">
          unavailable: {error}
        </p>
      )}

      {loading && !hasContent ? (
        <Skeleton />
      ) : empty && !hasContent ? (
        <p className="px-3 py-6 text-center text-[var(--fg-faint)]">{empty}</p>
      ) : (
        <div className={cn(error && "opacity-45")}>{children}</div>
      )}
    </section>
  );
}

/** Rows at the real height, so nothing jumps when the data lands. */
function Skeleton() {
  return (
    <div className="space-y-2 p-3">
      {[0, 1, 2].map((row) => (
        <div key={row} className="h-4 rounded bg-white/5 motion-safe:animate-pulse" />
      ))}
    </div>
  );
}
