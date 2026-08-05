import { useBox, useBoxAction } from "@/api/queries";
import { cn } from "@/lib/utils";

/**
 * The host's power state, and the button that changes it.
 *
 * The box wakes on demand and switches itself off when idle, so this is the
 * first thing on both pages that use it: without it, a stopped box looks
 * identical to a broken one, and the page below would just say "no server
 * configured" while the fix was one click away.
 *
 * Waking takes about two minutes. That is stated rather than hidden behind a
 * spinner — a spinner with no duration reads as a hang at about the thirty
 * second mark, which is when someone reloads and starts it again.
 */
export function BoxControl() {
  const box = useBox();
  const act = useBoxAction();

  // A 422 here means the VM does not exist — the ordinary state before
  // `just serve-create` has ever run. That is not a fault, and saying "error"
  // would send someone debugging a thing they have simply not made yet.
  const missing = box.error && String(box.error.message).includes("No blueprint host");
  const power = box.data?.power ?? (missing ? "not provisioned" : "unknown");
  const busy = act.isPending || power === "starting" || power === "stopping";

  return (
    <div className="flex flex-wrap items-center gap-3 rounded-md border border-[var(--border)] bg-[var(--panel)] px-3 py-2">
      <span className="font-mono text-[11px] tracking-widest text-[var(--fg-muted)] uppercase">
        host
      </span>
      <span className="flex items-center gap-2 font-mono text-[12px]">
        <span className={cn("size-2 rounded-full", dotFor(power))} />
        <span className="text-[var(--fg)]">{power}</span>
      </span>

      {box.data?.usable ? (
        <button
          type="button"
          disabled={busy}
          onClick={() => act.mutate("stop")}
          className="rounded border border-[var(--border)] px-2 py-1 font-mono text-[11px] text-[var(--fg-muted)] hover:bg-white/[0.06] hover:text-[var(--fg)] disabled:opacity-40"
        >
          stop
        </button>
      ) : missing ? (
        <span className="font-mono text-[11px] text-[var(--fg-faint)]">
          run `just serve-create` to make one
        </span>
      ) : (
        <button
          type="button"
          disabled={busy}
          onClick={() => act.mutate("start")}
          className="rounded border border-[var(--border)] px-2 py-1 font-mono text-[11px] text-[var(--fg)] hover:bg-white/[0.06] disabled:opacity-40"
        >
          start
        </button>
      )}

      {busy && (
        <span className="font-mono text-[11px] text-[var(--fg-faint)]">
          about two minutes — it boots, then loads the run
        </span>
      )}
      {act.error && (
        <span className="font-mono text-[11px] text-[#E0655C]">{String(act.error.message)}</span>
      )}
    </div>
  );
}

function dotFor(power: string): string {
  if (power === "running") return "bg-[#56B184]";
  if (power === "deallocated") return "bg-[var(--fg-faint)]";
  if (power === "starting" || power === "stopping") return "bg-[#D9A44E]";
  return "bg-[var(--fg-faint)]";
}
