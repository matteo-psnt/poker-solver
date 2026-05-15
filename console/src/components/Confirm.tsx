import { cn } from "@/lib/utils";
import { useEffect, useState } from "react";

/**
 * A button that asks once before doing something that cannot be un-clicked.
 *
 * The console has been read-only, where a stray click cost a round trip. It is
 * not any more, so the shape of a mistake changed: cancelling the wrong task
 * throws away hours of node time, and no amount of care at the keyboard makes
 * a single-click destructive control safe next to a list that reorders as it
 * polls.
 *
 * Two clicks rather than a modal. A modal steals focus and has to be dismissed
 * even when you meant it; arming in place keeps the row you are looking at on
 * screen, which is the thing you are actually confirming — that this is the
 * task you meant.
 *
 * It disarms itself after a few seconds. An armed button left on screen is a
 * trap for the NEXT click, and the list underneath is polling.
 */
export function Confirm({
  label,
  confirmLabel,
  onConfirm,
  pending,
  disabled,
  danger = true,
  title,
}: {
  label: string;
  confirmLabel: string;
  onConfirm: () => void;
  pending?: boolean;
  disabled?: boolean;
  danger?: boolean;
  title?: string;
}) {
  const [armed, setArmed] = useState(false);

  useEffect(() => {
    if (!armed) return;
    const timer = setTimeout(() => setArmed(false), 4000);
    return () => clearTimeout(timer);
  }, [armed]);

  return (
    <button
      type="button"
      title={title}
      disabled={disabled || pending}
      onClick={() => {
        if (armed) {
          setArmed(false);
          onConfirm();
        } else {
          setArmed(true);
        }
      }}
      className={cn(
        "rounded border px-2 py-0.5 font-mono text-[11px] transition-colors disabled:opacity-40",
        armed
          ? danger
            ? "border-[#E0655C] bg-[#E0655C]/15 text-[#E0655C]"
            : "border-[var(--fg)] text-[var(--fg)]"
          : "border-[var(--border)] text-[var(--fg-muted)] hover:text-[var(--fg)]",
      )}
    >
      {pending ? "…" : armed ? confirmLabel : label}
    </button>
  );
}
