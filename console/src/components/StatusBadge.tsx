/**
 * State to colour, over the ONE vocabulary this file still owns.
 *
 * The server classifies Azure's states once (`BatchTask.phase`, `.outcome`,
 * `.exit_meaning`, from `src/shared/task_states.py`); this file must not undo that.
 *
 * What remains is the TASK LOG's own causes, and they stay here on purpose: they
 * are words the node wrote into records on the share, `TERMINAL_CAUSES` gates
 * reconciliation against them, and renaming one at the source would make every
 * historical task read as unresolved. So they are renamed for DISPLAY, which is
 * the only kind of decision this file should be making.
 *
 * Two distinctions inside that vocabulary are load-bearing:
 *
 *   `timeout` is NOT `failed` -- an 8h wall-clock ceiling firing is information.
 *   `cancelled` is NOT `failed` -- exit -9 there reads as a crash and is not one;
 *   Batch reports it as `TaskEnded` / UserError.
 */
import { cn } from "@/lib/utils";

export type Tone = "live" | "ok" | "bad" | "warn" | "muted" | "pending";

const TONES: Record<Tone, string> = {
  live: "bg-blue-500/12 text-blue-400 ring-blue-500/25",
  ok: "bg-emerald-500/10 text-emerald-400 ring-emerald-500/20",
  bad: "bg-red-500/12 text-red-400 ring-red-500/25",
  warn: "bg-amber-500/12 text-amber-400 ring-amber-500/25",
  muted: "bg-zinc-500/10 text-zinc-400 ring-zinc-500/20",
  pending: "bg-transparent text-amber-400 ring-amber-500/40",
};

/**
 * Where the word IS the outcome: the task log's own causes, and the server's
 * `Outcome` for a finished Batch task. One map because they say the same
 * things — `timeout` and `timed out` are the same fact from two records.
 */
const BY_CAUSE: Record<string, Tone> = {
  running: "live",
  completed: "ok",
  success: "ok",
  done: "ok",
  failed: "bad",
  killed: "bad",
  failure: "bad",
  timeout: "warn",
  "timed out": "warn",
  partial: "warn",
  cancelled: "muted",
  canceled: "muted",
  started: "pending",
};

export function toneFor(state: string | null | undefined): Tone {
  return BY_CAUSE[(state ?? "").toLowerCase()] ?? "muted";
}

/**
 * The word to SHOW for a task-log cause.
 *
 * Renamed here rather than at the source: these are already written into
 * records on the share, and `TERMINAL_CAUSES` gates reconciliation against
 * them, so a renamed cause would make every historical task look unresolved.
 *
 * Batch's own vocabulary is deliberately absent. `active` used to be mapped to
 * `queued` here; the server says `queued` now.
 */
const DISPLAY: Record<string, string> = {
  started: "unresolved",
  killed: "killed (oom)",
};

export function displayName(word: string | null | undefined): string {
  const cause = (word ?? "").toLowerCase();
  return DISPLAY[cause] ?? cause;
}

export function StatusBadge({
  state,
  tone,
  title,
  className,
}: {
  state: string | null;
  tone?: Tone;
  title?: string;
  className?: string;
}) {
  const resolved = tone ?? toneFor(state);
  return (
    <span
      title={title}
      className={cn(
        "inline-flex items-center gap-1.5 rounded px-1.5 py-0.5 font-mono text-[11px] ring-1 ring-inset",
        TONES[resolved],
        className,
      )}
    >
      {resolved === "live" && (
        <span className="size-1.5 rounded-full bg-current motion-safe:animate-pulse" />
      )}
      {state ?? "—"}
    </span>
  );
}
