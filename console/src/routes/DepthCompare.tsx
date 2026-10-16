import { useMemo } from "react";
import { useSolverNodes } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { KIND_LABEL, KINDS, type Kind, type KindMix, kindMix, totalVariation } from "@/lib/depths";
import { aggregate, type Cell, summarise } from "@/lib/range";
import { cn } from "@/lib/utils";

/** The solver colours, one per kind: fold blue, call green, raise red, the jam darkest. */
const KIND_COLOUR: Record<Kind, string> = {
  fold: "#3D6FC4",
  call: "#2F7F58",
  raise: "#B23A32",
  allin: "#6E1F1A",
};

/** Aggressive to passive, fold on the right -- the order the chart draws in. */
const DRAWN = [...KINDS].reverse();

const pct = (v: number) => `${Math.round(v * 100)}%`;

type DepthRow = {
  stack: number;
  bb: number;
  cells: Cell[][] | null;
  tokens: string[];
  failed: boolean;
};

/**
 * The spot on screen, at every depth the box serves.
 *
 * Three readings the single chart cannot give: how the whole range shifts as
 * stacks deepen, how ONE hand is played at each depth, and which hands change
 * most between the selected depth and the next one up. Click a depth to move
 * the chart there; click a hand to pin it.
 */
export function DepthCompare({
  path,
  board,
  depths,
  bigBlind,
  combos,
  stack,
  hand,
  onStack,
  onHand,
}: {
  path: string;
  board: string;
  depths: readonly number[];
  bigBlind: number;
  combos: string[] | null;
  stack: number | undefined;
  hand: string;
  onStack: (stack: number) => void;
  onHand: (label: string) => void;
}) {
  // Shallow first, so the eye reads depth left to right / top to bottom.
  const ordered = useMemo(() => [...depths].sort((a, b) => a - b), [depths]);
  const nodes = useSolverNodes(path, board, ordered, combos !== null && ordered.length > 1);
  const current = stack ?? ordered[ordered.length - 1];

  const rows: DepthRow[] = ordered.map((s, i) => {
    const node = nodes[i]?.data;
    const grid = node?.grid ?? null;
    const cells =
      grid && combos
        ? aggregate({
            combos,
            comboBuckets: grid.combo_buckets,
            buckets: grid.buckets,
            actionCount: grid.actions.length,
          })
        : null;
    return {
      stack: s,
      bb: s / (bigBlind || 2),
      cells,
      tokens: grid?.actions ?? [],
      failed: Boolean(nodes[i]?.error),
    };
  });

  const handAt = (row: DepthRow): KindMix | null => {
    for (const line of row.cells ?? []) {
      for (const cell of line) if (cell.label === hand) return kindMix(cell.strategy, row.tokens);
    }
    return null;
  };
  const rangeAt = (row: DepthRow): KindMix | null => {
    if (!row.cells) return null;
    const summary = summarise(row.cells, row.tokens.length);
    return summary ? kindMix(summary.strategy, row.tokens) : null;
  };

  const index = rows.findIndex((r) => r.stack === current);
  const [from, to] =
    index < rows.length - 1 ? [rows[index], rows[index + 1]] : [rows[index - 1], rows[index]];
  const movers = useMemo(() => {
    if (!from?.cells || !to?.cells) return [];
    const out: { label: string; a: KindMix; b: KindMix; tv: number }[] = [];
    for (const [r, line] of from.cells.entries()) {
      for (const [c, cell] of line.entries()) {
        const other = to.cells[r]?.[c];
        if (!cell.strategy || !other?.strategy) continue;
        const a = kindMix(cell.strategy, from.tokens);
        const b = kindMix(other.strategy, to.tokens);
        out.push({ label: cell.label, a, b, tv: totalVariation(a, b) });
      }
    }
    return out.sort((x, y) => y.tv - x.tv).slice(0, 8);
  }, [from, to]);

  if (ordered.length < 2) return null;
  const loading = nodes.some((q) => q.isLoading);

  return (
    <div className="grid gap-3 xl:grid-cols-3">
      <Panel title="Whole range by depth" loading={loading}>
        <DepthBars rows={rows} mixOf={rangeAt} current={current} onStack={onStack} />
      </Panel>
      <Panel title={`${hand} by depth`} loading={loading}>
        <DepthBars rows={rows} mixOf={handAt} current={current} onStack={onStack} />
      </Panel>
      <Panel
        title={from && to ? `Biggest changes, ${from.bb} → ${to.bb} bb` : "Biggest changes"}
        loading={loading}
      >
        <div className="space-y-1 p-3 text-[12px]">
          {movers.map((m) => (
            <button
              key={m.label}
              type="button"
              onClick={() => onHand(m.label)}
              className="grid w-full grid-cols-[44px_1fr_1fr_36px] items-center gap-2 text-left hover:bg-white/[0.03]"
            >
              <span className={cn("font-mono", m.label === hand && "text-[var(--fg)] underline")}>
                {m.label}
              </span>
              <Stack mix={m.a} />
              <Stack mix={m.b} />
              <span className="tnum text-right text-[var(--fg-muted)]">
                {Math.round(m.tv * 100)}
              </span>
            </button>
          ))}
          {movers.length > 0 && (
            <p className="pt-1 text-[11px] text-[var(--fg-faint)]">
              last column: how much of the hand's strategy moved, 0 to 100
            </p>
          )}
        </div>
      </Panel>
    </div>
  );
}

function DepthBars({
  rows,
  mixOf,
  current,
  onStack,
}: {
  rows: DepthRow[];
  mixOf: (row: DepthRow) => KindMix | null;
  current: number | undefined;
  onStack: (stack: number) => void;
}) {
  return (
    <div className="space-y-1 p-3 text-[12px]">
      {rows.map((row) => {
        const mix = mixOf(row);
        const chosen = row.stack === current;
        return (
          <button
            key={row.stack}
            type="button"
            onClick={() => onStack(row.stack)}
            className="grid w-full grid-cols-[52px_1fr_120px] items-center gap-2 text-left hover:bg-white/[0.03]"
          >
            <span
              className={cn(
                "tnum text-right",
                chosen ? "text-[var(--fg)]" : "text-[var(--fg-muted)]",
              )}
            >
              {row.bb} bb
            </span>
            {mix ? (
              <Stack mix={mix} ring={chosen} />
            ) : (
              <span className="text-[var(--fg-faint)]">{row.failed ? "not served" : "…"}</span>
            )}
            <span className="truncate text-[11px] text-[var(--fg-muted)]">
              {mix ? describe(mix) : ""}
            </span>
          </button>
        );
      })}
      <p className="flex gap-3 pt-1 text-[11px] text-[var(--fg-faint)]">
        {DRAWN.map((k) => (
          <span key={k} className="flex items-center gap-1">
            <span className="size-2 rounded-sm" style={{ background: KIND_COLOUR[k] }} />
            {KIND_LABEL[k]}
          </span>
        ))}
      </p>
    </div>
  );
}

function Stack({ mix, ring }: { mix: KindMix; ring?: boolean }) {
  return (
    <span
      className={cn(
        "flex h-2.5 overflow-hidden rounded-sm",
        ring && "outline outline-1 outline-[var(--fg-muted)]",
      )}
    >
      {DRAWN.map((k) => (
        <span key={k} style={{ width: `${mix[k] * 100}%`, background: KIND_COLOUR[k] }} />
      ))}
    </span>
  );
}

/** `raise 87%, all in 12%` -- the kinds worth naming, largest first. */
function describe(mix: KindMix): string {
  return KINDS.filter((k) => mix[k] >= 0.05)
    .sort((a, b) => mix[b] - mix[a])
    .map((k) => `${KIND_LABEL[k]} ${pct(mix[k])}`)
    .join(", ");
}
