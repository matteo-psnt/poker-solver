import { useBlueprintRun, useCombos, useSolverNode } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { RangeGrid } from "@/components/RangeGrid";
import { type ActionLabel, describeAction, describeActions } from "@/lib/actions";
import { type Cell, actionColours, aggregate } from "@/lib/range";
import { cn } from "@/lib/utils";
import { getRouteApi, useNavigate } from "@tanstack/react-router";
import { useMemo, useState } from "react";

const route = getRouteApi("/blueprint");

/**
 * The chart, as the thing the page is FOR.
 *
 * It was a 13x13 grid in a `minmax(0,1fr)` column beside a 220px sidebar,
 * inside a padded panel, under two other panels — so the object you opened the
 * page to read got about a third of the screen and the controls got the rest.
 * Here the grid is the page: the line and the board sit in one bar above it,
 * everything else is a rail beside it, and the grid takes what is left.
 *
 * **The spot is in the URL.** The old page's own docstring said a spot is
 * addressed by its line so "a bookmarked spot stays the spot it was" — but
 * `path`, `board` and `average` were all `useState`, so nothing on it was
 * bookmarkable or shareable at all. They are search params now, which is what
 * makes that sentence true and what lets you send someone a spot.
 */
export function Charts() {
  const { path, board, average } = route.useSearch();
  const navigate = useNavigate({ from: "/blueprint" });
  const [pinned, setPinned] = useState<Cell | null>(null);
  const [hovered, setHovered] = useState<Cell | null>(null);

  // What the box is actually holding, which `Loaded` can also change. Polled
  // only while a swap is in flight — see `useBlueprintRun`.
  const run = useBlueprintRun();
  const combos = useCombos(!!run.data);
  const node = useSolverNode(path, board, average, !!run.data);

  const set = (next: Partial<{ path: string; board: string; average: boolean }>) =>
    navigate({ search: (old) => ({ ...old, ...next }) });

  const grid = node.data?.grid ?? null;
  const cells = useMemo(() => {
    if (!grid || !combos.data) return null;
    return aggregate({
      combos: combos.data.combos,
      comboBuckets: grid.combo_buckets,
      buckets: grid.buckets,
      actionCount: grid.actions.length,
    });
  }, [grid, combos.data]);

  // Sizes are meaningless without the stakes, so labelling waits on /run.
  const bigBlind = run.data?.big_blind ?? 0;
  const labels = useMemo(
    () => describeActions(grid?.actions ?? [], bigBlind),
    [grid?.actions, bigBlind],
  );

  const steps = path ? path.split("/") : [];
  const shown = pinned ?? hovered;

  return (
    <div className="space-y-3">
      <Panel
        title={grid ? `${grid.street} · seat ${grid.actor} to act` : "chart"}
        aside={<Line steps={steps} bigBlind={bigBlind} onJump={(p) => set({ path: p })} />}
        error={node.error ? String(node.error.message ?? node.error) : null}
        loading={node.isFetching && !node.data}
        empty={node.data?.terminal ? "Nobody acts here — the hand is already over." : null}
      >
        <Controls
          board={board}
          average={average}
          atRoot={steps.length === 0 && !board}
          next={node.data?.children ?? []}
          bigBlind={bigBlind}
          onSet={set}
          onStep={(token) => set({ path: path ? `${path}/${token}` : token })}
        />

        {cells && grid ? (
          <div className="grid items-start gap-5 p-3 xl:grid-cols-[minmax(0,1fr)_15rem]">
            {/* Capped, not stretched: past ~44px a cell is empty space, and the
                label is what has to stay legible, not the square. Left-aligned,
                because centring it left a dead gutter the width of the rail. */}
            <div className="w-full max-w-[52rem]">
              <RangeGrid
                cells={cells}
                actions={labels}
                onHover={setHovered}
                onPick={(cell) =>
                  setPinned((was) => (was && cell && was.label === cell.label ? null : cell))
                }
              />
            </div>

            <aside className="space-y-3 text-[12px]">
              <Legend labels={labels} />
              <Coverage grid={grid} />
              <HandDetail cell={shown} actions={labels} pinned={pinned !== null} />
            </aside>
          </div>
        ) : undefined}
      </Panel>
    </div>
  );
}

/** The line that reaches this spot, in poker rather than in tokens. */
function Line({
  steps,
  bigBlind,
  onJump,
}: {
  steps: string[];
  bigBlind: number;
  onJump: (path: string) => void;
}) {
  return (
    <span className="flex flex-wrap items-center gap-1">
      <Crumb label="preflop" onClick={() => onJump("")} active={steps.length === 0} />
      {steps.map((step, index) => (
        <span key={`${index}-${step}`} className="flex items-center gap-1">
          <span className="text-[var(--fg-faint)]">›</span>
          <Crumb
            label={describeAction(step, bigBlind).text}
            onClick={() => onJump(steps.slice(0, index + 1).join("/"))}
            active={index === steps.length - 1}
          />
        </span>
      ))}
    </span>
  );
}

/** The bar above the grid: where to go next, and which board to go there on. */
function Controls({
  board,
  average,
  atRoot,
  next,
  bigBlind,
  onSet,
  onStep,
}: {
  board: string;
  average: boolean;
  atRoot: boolean;
  /** The actions available FROM this spot — not React children. */
  next: { token: string }[];
  bigBlind: number;
  onSet: (next: Partial<{ path: string; board: string; average: boolean }>) => void;
  onStep: (token: string) => void;
}) {
  return (
    <div className="space-y-2 border-b border-[var(--border)] px-3 py-2.5">
      <div className="flex flex-wrap items-center gap-1.5">
        {next.map((child) => (
          <button
            key={child.token}
            type="button"
            onClick={() => onStep(child.token)}
            title={`token: ${child.token}`}
            className="rounded border border-[var(--border)] px-2 py-1 font-mono text-[11px] text-[var(--fg-muted)] hover:border-[var(--fg-faint)] hover:text-[var(--fg)]"
          >
            {describeAction(child.token, bigBlind).text}
          </button>
        ))}
        {next.length === 0 && (
          <span className="text-[11px] text-[var(--fg-faint)]">no actions from here</span>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-4">
        <label className="flex items-center gap-2 text-[11px] text-[var(--fg-muted)]">
          <span className="tracking-wider uppercase text-[var(--fg-faint)]">board</span>
          <input
            value={board}
            onChange={(event) => onSet({ board: event.target.value })}
            placeholder="preflop"
            spellCheck={false}
            className="w-36 rounded border border-[var(--border)] bg-transparent px-2 py-1 font-mono text-[11px] text-[var(--fg)] placeholder:text-[var(--fg-faint)]"
          />
          {/* Preflop is a REAL state, not an empty field: the whole point of
              the page for most readers. One click back to it. */}
          {!atRoot && (
            <button
              type="button"
              onClick={() => onSet({ path: "", board: "" })}
              className="text-[11px] text-[var(--fg-faint)] underline-offset-2 hover:text-[var(--fg)] hover:underline"
            >
              back to preflop
            </button>
          )}
        </label>

        <label className="flex items-center gap-2 text-[11px] text-[var(--fg-muted)]">
          <input
            type="checkbox"
            checked={average}
            onChange={(event) => onSet({ average: event.target.checked })}
          />
          average strategy
          {/* Not a detail: the average is the blueprint and what converges;
              the current strategy is regret-matching's latest guess, and on an
              under-trained run they disagree sharply. */}
          <span className="text-[var(--fg-faint)]">
            {average ? "the blueprint proper" : "regret-matched current guess"}
          </span>
        </label>
      </div>
    </div>
  );
}

function Legend({ labels }: { labels: ActionLabel[] }) {
  const colours = actionColours(labels.map((label) => label.token));
  return (
    <div className="space-y-1">
      {labels.map((label, index) => (
        <div key={label.token} className="flex items-center gap-2">
          <span
            className="size-2.5 shrink-0 rounded-[2px]"
            style={{ backgroundColor: colours[index] }}
          />
          <span className="text-[var(--fg-muted)]">{label.text}</span>
        </div>
      ))}
    </div>
  );
}

/** How much of this spot the solver actually learned. */
function Coverage({
  grid,
}: {
  grid: { buckets: Record<string, unknown>; trained_buckets: number; blocked: number };
}) {
  const total = Object.keys(grid.buckets).length;
  const fraction = total ? grid.trained_buckets / total : 0;
  return (
    <div className="space-y-1.5 border-t border-[var(--border)] pt-2">
      <div className="flex justify-between gap-2">
        <span className="text-[var(--fg-faint)]">trained buckets</span>
        <span className="tabular-nums text-[var(--fg)]">
          {grid.trained_buckets}/{total}
        </span>
      </div>
      {/* The number most worth reading: a bucket the solver never visited has
          no strategy, only an allocated row — so a chart that looks complete
          can be mostly hatching. */}
      <span className="block h-1 overflow-hidden rounded-full bg-[var(--border)]">
        <span
          className={cn(
            "block h-full rounded-full",
            fraction > 0.9 ? "bg-emerald-500" : "bg-amber-500",
          )}
          style={{ width: `${fraction * 100}%` }}
        />
      </span>
      <div className="flex justify-between gap-2">
        <span className="text-[var(--fg-faint)]">blocked</span>
        <span className="tabular-nums text-[var(--fg-muted)]">{grid.blocked} combos</span>
      </div>
    </div>
  );
}

/**
 * One hand's mix. Pinned by clicking, because the old page showed it on hover
 * only — so reading a number meant holding the mouse still, and comparing two
 * hands meant remembering the first one.
 */
function HandDetail({
  cell,
  actions,
  pinned,
}: {
  cell: Cell | null;
  actions: ActionLabel[];
  pinned: boolean;
}) {
  if (!cell) {
    return (
      <div className="min-h-[7rem] border-t border-[var(--border)] pt-2 text-[var(--fg-faint)]">
        Hover a hand, or click to pin it.
      </div>
    );
  }
  return (
    <div className="min-h-[7rem] space-y-1 border-t border-[var(--border)] pt-2">
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-[14px] text-[var(--fg)]">{cell.label}</span>
        <span className="text-[11px] text-[var(--fg-faint)]">{cell.combos} combos</span>
        {pinned && <span className="ml-auto text-[11px] text-[var(--fg-faint)]">pinned</span>}
      </div>
      {cell.combos === 0 ? (
        <div className="text-[var(--fg-faint)]">blocked by the board</div>
      ) : cell.strategy ? (
        cell.strategy.map((weight, index) => (
          <div key={actions[index]?.token ?? index} className="flex justify-between gap-2">
            <span className="text-[var(--fg-muted)]">{actions[index]?.text}</span>
            <span className="tabular-nums text-[var(--fg)]">{(weight * 100).toFixed(1)}%</span>
          </div>
        ))
      ) : (
        <div className="text-[var(--fg-faint)]">never trained here</div>
      )}
      {cell.untrained > 0 && cell.strategy && (
        <div className="text-[11px] text-[var(--fg-faint)]">
          averaged over {cell.combos - cell.untrained} of {cell.combos}
        </div>
      )}
    </div>
  );
}

function Crumb({
  label,
  onClick,
  active,
}: { label: string; onClick: () => void; active: boolean }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded px-1.5 py-0.5 font-mono text-[11px]",
        active
          ? "bg-white/[0.09] text-[var(--fg)]"
          : "text-[var(--fg-muted)] hover:text-[var(--fg)]",
      )}
    >
      {label}
    </button>
  );
}
