import type { ActionLabel } from "@/lib/actions";
import { type Cell, GRID_SIZE, actionColour } from "@/lib/range";
import { cn } from "@/lib/utils";

/**
 * The 13x13 chart, each cell a stacked bar of the actions taken with that class.
 *
 * Three cell states, drawn differently on purpose:
 *
 *   blocked    the board holds one of its cards — nothing to say, drawn empty
 *   untrained  reachable, but training never visited its bucket — hatched
 *   trained    a stacked bar in action order
 *
 * Collapsing the first two into "no bar" would be the single most misleading
 * thing this page could do: "you cannot hold this" and "the solver never
 * learned this" look identical on screen and mean opposite things about whether
 * the number below is worth reading.
 */
export function RangeGrid({
  cells,
  actions,
  onHover,
}: {
  cells: Cell[][];
  /** Display labels, in the same order as each cell's strategy. */
  actions: ActionLabel[];
  onHover?: (cell: Cell | null) => void;
}) {
  return (
    <div
      className="grid w-full gap-px"
      style={{ gridTemplateColumns: `repeat(${GRID_SIZE}, minmax(0, 1fr))` }}
      onMouseLeave={() => onHover?.(null)}
    >
      {cells.flatMap((row) =>
        row.map((cell) => (
          <button
            type="button"
            // Label is unique across the grid and stable, unlike the indices.
            key={cell.label}
            onMouseEnter={() => onHover?.(cell)}
            onFocus={() => onHover?.(cell)}
            className="relative aspect-square overflow-hidden rounded-[2px] border border-[var(--border)] text-left focus-visible:outline focus-visible:outline-2 focus-visible:outline-[var(--fg)]"
            title={describe(cell, actions)}
          >
            {cell.combos === 0 ? null : cell.strategy ? (
              <Bar strategy={cell.strategy} actions={actions} />
            ) : (
              <Hatch />
            )}
            <span
              className={cn(
                "absolute inset-0 flex items-center justify-center font-mono text-[9px] leading-none",
                cell.combos === 0 ? "text-[var(--fg-faint)]" : "text-white mix-blend-luminosity",
              )}
              // The diagonal (pairs) and the two triangles are easier to keep
              // oriented in with the label always legible over the bar.
              style={{ textShadow: "0 1px 2px rgba(0,0,0,.65)" }}
            >
              {cell.label}
            </span>
            {cell.untrained > 0 && cell.strategy && (
              // A partly-untrained class still shows its bar, but the corner
              // says the average is over fewer combos than it looks.
              <span className="absolute top-0 right-0 size-1 rounded-full bg-[var(--fg-faint)]" />
            )}
          </button>
        )),
      )}
    </div>
  );
}

function Bar({ strategy, actions }: { strategy: number[]; actions: ActionLabel[] }) {
  return (
    <span className="absolute inset-0 flex">
      {strategy.map((weight, index) => (
        <span
          key={actions[index]?.token ?? index}
          style={{
            width: `${weight * 100}%`,
            // Keyed on the TOKEN, not the label: colour encodes what kind of
            // action it is, and the label is prose that may be translated.
            backgroundColor: actionColour(actions[index]?.token ?? "", index, actions.length),
          }}
        />
      ))}
    </span>
  );
}

/** Reachable but never trained — visibly not a strategy. */
function Hatch() {
  return (
    <span
      className="absolute inset-0 opacity-50"
      style={{
        backgroundImage:
          "repeating-linear-gradient(45deg, var(--fg-faint) 0 1px, transparent 1px 4px)",
      }}
    />
  );
}

function describe(cell: Cell, actions: ActionLabel[]): string {
  if (cell.combos === 0) return `${cell.label} — blocked by the board`;
  if (!cell.strategy) return `${cell.label} — ${cell.combos} combos, none trained`;
  const parts = cell.strategy
    .map((weight, index) => `${actions[index]?.text ?? "?"} ${(weight * 100).toFixed(0)}%`)
    .join("  ");
  const caveat = cell.untrained > 0 ? `  (${cell.untrained}/${cell.combos} untrained)` : "";
  return `${cell.label}  ${parts}${caveat}`;
}
