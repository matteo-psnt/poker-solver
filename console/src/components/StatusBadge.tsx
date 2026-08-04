/**
 * ONE mapping from state to colour — over THREE different vocabularies.
 *
 * That is the subtlety this file exists for. The console shows Batch job
 * states, Batch task states and the leg log's own causes, and the same word
 * means different things in each:
 *
 *   Job `active`     the container is open. It stays open all day with nothing
 *                    running, so it is not evidence of work.
 *   Task `active`    QUEUED, waiting for a node — not running. A task frozen
 *                    here inside a finished job will never run at all.
 *   Task `completed` FINISHED, not succeeded. Success lives in `exit_code` /
 *                    `result`; `completed` with -9 is a cancellation.
 *   Leg `completed`  the node's own terminal record. This one does mean success.
 *
 * Painting Batch's `completed` green was therefore wrong: a cancelled task and
 * a clean one carried the same badge. `taskTone` combines state WITH outcome;
 * `toneFor` is for vocabularies where the word alone is the whole answer.
 *
 * Two distinctions inside the leg vocabulary are load-bearing and were both
 * learned the hard way:
 *
 *   `timeout` is NOT `failed` — an 8h wall-clock ceiling firing is information.
 *   `cancelled` is NOT `failed` — exit -9 there reads as a crash and is not
 *   one; Batch reports it as `TaskEnded` / UserError.
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

/** The leg log's causes, where the word IS the outcome. */
const BY_CAUSE: Record<string, Tone> = {
  running: "live",
  completed: "ok",
  success: "ok",
  failed: "bad",
  killed: "bad",
  failure: "bad",
  timeout: "warn",
  partial: "warn",
  cancelled: "muted",
  canceled: "muted",
  started: "pending",
};

export function toneFor(state: string | null | undefined): Tone {
  return BY_CAUSE[(state ?? "").toLowerCase()] ?? "muted";
}

/**
 * The word to SHOW, which is not always the word on the wire.
 *
 * Azure calls a queued task `active`, which reads as the opposite of what it
 * means, and `completed` for anything that stopped — success or not. Those are
 * Batch's names and cannot be changed; what is shown can. Our own leg causes
 * are renamed here too rather than at the source, because they are already
 * written into records on the share and `TERMINAL_CAUSES` gates reconciliation
 * against them: a renamed cause would make every historical leg look
 * unresolved.
 */
const DISPLAY: Record<string, string> = {
  active: "queued",
  started: "unresolved",
  killed: "killed (oom)",
};

export function displayName(word: string | null | undefined): string {
  const short = shortState(word);
  return DISPLAY[short] ?? short;
}

/**
 * A Batch task's OUTCOME as a word, rather than its state plus a number.
 *
 * `completed` says only that it stopped; the exit code says what happened. One
 * word carries both, and it uses the same vocabulary as the leg log so the two
 * panels can be read the same way.
 */
export function taskOutcome(state: string | null | undefined, exitCode: number | null): string {
  const short = shortState(state);
  if (short !== "completed") return displayName(short);
  if (exitCode === 0 || exitCode === null) return "done";
  if (exitCode === 124) return "timed out";
  if (exitCode === -9) return "cancelled";
  return "failed";
}

/** `BatchTaskState.RUNNING` -> `running`. */
export function shortState(raw: string | null | undefined): string {
  return (raw ?? "").split(".").pop()?.toLowerCase() ?? "";
}

/**
 * A Batch TASK's tone, which needs the exit code as well as the state.
 *
 * `completed` alone says only that the task stopped. Colouring on it made a
 * cancelled task (`-9`) and a crashed one (`137`) look identical to a clean
 * run.
 */
export function taskTone(state: string | null | undefined, exitCode: number | null): Tone {
  const short = shortState(state);
  if (short !== "completed") {
    // `active` is queued, not running — it should not pulse like live work.
    return short === "running" || short === "preparing" ? "live" : "muted";
  }
  if (exitCode === 0 || exitCode === null) return "ok";
  if (exitCode === 124) return "warn"; // the wall-clock guard fired
  if (exitCode === -9) return "muted"; // Batch's SIGKILL on cancel
  return "bad";
}

/**
 * What a Batch task's exit code MEANT, in words.
 *
 * These are the ones that recur here; anything else is left as a bare number
 * rather than guessed at.
 */
export function exitMeaning(exitCode: number | null): string | null {
  switch (exitCode) {
    case null:
      return null;
    case 0:
      return "clean";
    case 2:
      return "the CLI rejected a flag — argparse exits before doing any work";
    case 124:
      return "the wall-clock guard fired (a hang, not a crash)";
    case 137:
      return "SIGKILL from outside — the OOM killer";
    case -9:
      return "Batch killed it on cancellation";
    default:
      return null;
  }
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
