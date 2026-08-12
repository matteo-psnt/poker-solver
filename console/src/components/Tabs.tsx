import { cn } from "@/lib/utils";

/**
 * A tab strip whose selection lives in the URL.
 *
 * **Not `useState`.** The console already learned this once: the chart page's
 * own docstring claimed a bookmarked spot stays the spot it was, while `path`,
 * `board` and `average` were all component state — so nothing on it was
 * shareable and the claim was simply false. A tab is the same kind of thing. If
 * `/blueprint` opens on the chart no matter what you were looking at, then
 * sending someone "the play table" is impossible and a refresh loses your place.
 *
 * Tabs are used here for pages that are one SUBJECT asked in two ways —
 * a blueprint read as a grid or played against; the operator's own doing and
 * what it cost. They are not a way to fit unrelated pages behind one nav item:
 * that is the fourteen-route problem wearing a different control.
 */
export function Tabs<T extends string>({
  tabs,
  active,
  onPick,
}: {
  tabs: readonly { id: T; label: string; hint?: string }[];
  active: T;
  onPick: (id: T) => void;
}) {
  return (
    <div role="tablist" className="flex items-center gap-1 border-b border-[var(--border)] px-1">
      {tabs.map(({ id, label, hint }) => {
        const selected = id === active;
        return (
          <button
            key={id}
            type="button"
            role="tab"
            aria-selected={selected}
            title={hint}
            onClick={() => onPick(id)}
            className={cn(
              "-mb-px border-b-2 px-3 py-1.5 text-[12px]",
              selected
                ? "border-[var(--fg)] text-[var(--fg)]"
                : "border-transparent text-[var(--fg-muted)] hover:text-[var(--fg)]",
            )}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}
