/**
 * ONE mapping from state to colour, used everywhere.
 *
 * If the same cause looks different on two pages, the colour stops meaning
 * anything. Two distinctions are load-bearing and were both learned the hard
 * way:
 *
 *   `timeout` is NOT `failed` — an 8h wall-clock ceiling firing is information.
 *   `cancelled` is NOT `failed` — exit -9 on a cancelled task reads as a crash
 *   and is not one; Batch reports it as `TaskEnded` / UserError.
 */
import { cn } from "@/lib/utils";

type Tone = "live" | "ok" | "bad" | "warn" | "muted" | "pending";

const TONES: Record<Tone, string> = {
  live: "bg-blue-500/12 text-blue-400 ring-blue-500/25",
  ok: "bg-emerald-500/10 text-emerald-400 ring-emerald-500/20",
  bad: "bg-red-500/12 text-red-400 ring-red-500/25",
  warn: "bg-amber-500/12 text-amber-400 ring-amber-500/25",
  muted: "bg-zinc-500/10 text-zinc-400 ring-zinc-500/20",
  pending: "bg-transparent text-amber-400 ring-amber-500/40",
};

const BY_STATE: Record<string, Tone> = {
  running: "live",
  active: "live",
  preparing: "live",
  completed: "ok",
  success: "ok",
  failed: "bad",
  failure: "bad",
  timeout: "warn",
  cancelled: "muted",
  canceled: "muted",
  started: "pending",
};

export function toneFor(state: string | null | undefined): Tone {
  return BY_STATE[(state ?? "").toLowerCase()] ?? "muted";
}

export function StatusBadge({ state, className }: { state: string | null; className?: string }) {
  const tone = toneFor(state);
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 rounded px-1.5 py-0.5 font-mono text-[11px] ring-1 ring-inset",
        TONES[tone],
        className,
      )}
    >
      {tone === "live" && (
        <span className="size-1.5 rounded-full bg-current motion-safe:animate-pulse" />
      )}
      {state ?? "—"}
    </span>
  );
}
