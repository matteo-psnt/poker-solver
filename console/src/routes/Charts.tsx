import { ApiError } from "@/api/client";
import { useBlueprintRun, useCombos, useSolverNode } from "@/api/queries";
import { BoardPicker } from "@/components/BoardPicker";
import { Panel } from "@/components/Panel";
import { RangeGrid } from "@/components/RangeGrid";
import { type ActionLabel, describeAction, describeActions } from "@/lib/actions";
import { type Cell, type RangeSummary, actionColours, aggregate, summarise } from "@/lib/range";
import { cn } from "@/lib/utils";
import { getRouteApi, useNavigate } from "@tanstack/react-router";
import { useMemo, useState } from "react";

const route = getRouteApi("/blueprint");

/**
 * The chart, as the thing the page is FOR: the grid is the page, the line and the
 * board sit in one bar above it, everything else is a rail beside it.
 *
 * **The spot is in the URL.** `path`, `board` and `average` are search params, not
 * `useState`, which is what makes a bookmarked spot the spot it was and lets you
 * send someone one.
 *
 * Every line past the preflop needs a board and replay refuses without one. That
 * refusal is not a fault -- it is the page asking a question, and it is shown as
 * one with the deck open under it.
 */
export function Charts() {
  const { path, board, average } = route.useSearch();
  const navigate = useNavigate({ from: "/blueprint" });
  /**
   * A pin belongs to the SPOT it was taken in, so it is STORED with one.
   *
   * `cells` recomputes when the node changes, but the pin was a bare snapshot
   * of one cell and nothing dropped it — so pinning AA preflop and stepping
   * into a line whose board contains an ace left the grid drawing AA as blocked
   * while the rail beside it still showed the preflop strategy and combo count,
   * under a label saying "pinned". Two contradictory answers to one question,
   * on screen together.
   *
   * Carrying the spot and comparing beats resetting in an effect: there is no
   * frame where the stale pin is still rendered, and nothing to keep in step
   * with the list of things that make a spot.
   */
  const spot = `${path}|${board}|${average}`;
  const [held, setHeld] = useState<{ spot: string; cell: Cell } | null>(null);
  const pinned = held?.spot === spot ? held.cell : null;
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
  const summary = useMemo(
    () => (cells && grid ? summarise(cells, grid.actions.length) : null),
    [cells, grid],
  );

  // Sizes are meaningless without the stakes, so labelling waits on /run.
  const bigBlind = run.data?.big_blind ?? 0;
  const labels = useMemo(
    () => describeActions(grid?.actions ?? [], bigBlind),
    [grid?.actions, bigBlind],
  );

  const steps = path ? path.split("/") : [];
  const shown = pinned ?? hovered;
  const needsBoard = boardRefusal(node.error);

  return (
    <div className="space-y-3">
      <Panel
        title={grid ? `${grid.street} · seat ${grid.actor} to act` : "chart"}
        aside={<Line steps={steps} bigBlind={bigBlind} onJump={(p) => set({ path: p })} />}
        // A short board is the page's own question, answered right below with
        // the deck. Routing it here would grey the panel and file it as a fault.
        error={needsBoard ? null : (node.error && String(node.error.message)) || null}
        loading={node.isFetching && !node.data}
        empty={node.data?.terminal ? "Nobody acts here — the hand is already over." : null}
      >
        <Controls
          board={board}
          average={average}
          atRoot={steps.length === 0 && !board}
          next={node.data?.children ?? []}
          reached={node.data?.board.length ?? null}
          needsBoard={needsBoard}
          bigBlind={bigBlind}
          onSet={set}
          onStep={(token) => set({ path: path ? `${path}/${token}` : token })}
        />

        {cells && grid ? (
          <div className="grid items-start gap-5 p-3 xl:grid-cols-[minmax(0,1fr)_16rem]">
            {/* Capped, not stretched: past ~44px a cell is empty space, and the
                label is what has to stay legible, not the square. Left-aligned,
                because centring it left a dead gutter the width of the rail. */}
            <div className="w-full max-w-[52rem]">
              <RangeGrid
                cells={cells}
                actions={labels}
                selected={shown?.label ?? null}
                pinned={pinned?.label ?? null}
                onHover={setHovered}
                onPick={(cell) =>
                  setHeld((was) =>
                    was?.spot === spot && cell && was.cell.label === cell.label
                      ? null
                      : cell && { spot, cell },
                  )
                }
              />
            </div>

            <aside className="space-y-3 text-[12px]">
              <Summary summary={summary} labels={labels} />
              <Coverage grid={grid} />
              <HandDetail cell={shown} actions={labels} pinned={pinned !== null} />
            </aside>
          </div>
        ) : undefined}
      </Panel>
    </div>
  );
}

/**
 * A refusal about the BOARD, rather than a fault -- and the server's words for it.
 *
 * *"This line reaches Flop and needs 3 board cards, but 0 were given."* Both facts
 * a reader needs are in that sentence, so it is shown rather than paraphrased: a
 * copy here is a second thing to keep true when the streets or the wording move.
 *
 * The discriminator is that the message mentions a CARD. Every board refusal does
 * -- too few for the line, `'Ax' is not a card`, `repeats a card` -- and no path
 * refusal does: those name a token and list what was on offer, and they mean a
 * bookmark that outlived its action model, which IS worth greying the panel for.
 *
 * Recognised by matching, as `BoxControl` does for "no blueprint host". Working out
 * client-side which street a line reaches is engine logic, and the console does not
 * get to hold a second copy of the rules.
 */
function boardRefusal(error: unknown): string | null {
  if (!(error instanceof ApiError) || error.status !== 422) return null;
  return /card/i.test(error.message) ? error.message : null;
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
  reached,
  needsBoard,
  bigBlind,
  onSet,
  onStep,
}: {
  board: string;
  average: boolean;
  atRoot: boolean;
  /** The actions available FROM this spot — not React children. */
  next: { token: string }[];
  /** Board cards this line consumes, once it is answerable. */
  reached: number | null;
  /** The server's refusal when the board is short, or null. */
  needsBoard: string | null;
  bigBlind: number;
  onSet: (next: Partial<{ path: string; board: string; average: boolean }>) => void;
  onStep: (token: string) => void;
}) {
  return (
    <div className="space-y-3 border-b border-[var(--border)] px-3 py-2.5">
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
        {next.length === 0 && !needsBoard && (
          <span className="text-[11px] text-[var(--fg-faint)]">no actions from here</span>
        )}
        {!atRoot && (
          <button
            type="button"
            onClick={() => onSet({ path: "", board: "" })}
            className="ml-auto text-[11px] text-[var(--fg-faint)] underline-offset-2 hover:text-[var(--fg)] hover:underline"
          >
            back to preflop
          </button>
        )}
      </div>

      {needsBoard && (
        <p className="rounded border border-amber-500/30 bg-amber-500/5 px-2.5 py-1.5 text-[12px] text-amber-300/90">
          {needsBoard}
        </p>
      )}

      <BoardPicker
        board={board}
        onChange={(next) => onSet({ board: next })}
        live={reached}
        forceOpen={needsBoard !== null}
      />

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
  );
}

/**
 * What the range does overall, above the per-hand detail.
 *
 * First in the rail because it is the first question — "does this spot fold a
 * lot" — and the grid is bad at it: a class gets one square whether it holds 4
 * combos or 12, so an offsuit-heavy fold looks smaller than it is. See
 * `summarise`, which weights by combos for exactly that reason.
 */
function Summary({ summary, labels }: { summary: RangeSummary | null; labels: ActionLabel[] }) {
  if (!summary) {
    return (
      <div className="text-[var(--fg-faint)]">Nothing here was trained — no range to total.</div>
    );
  }
  const colours = actionColours(labels.map((label) => label.token));
  return (
    <div className="space-y-1.5">
      <div className="text-[11px] tracking-wider text-[var(--fg-faint)] uppercase">whole range</div>
      <span className="flex h-2.5 overflow-hidden rounded-[2px]">
        {summary.strategy.map((weight, index) => (
          <span
            key={labels[index]?.token ?? index}
            style={{ width: `${weight * 100}%`, backgroundColor: colours[index] }}
          />
        ))}
      </span>
      {labels.map((label, index) => (
        <div key={label.token} className="flex items-center gap-2">
          <span
            className="size-2.5 shrink-0 rounded-[2px]"
            style={{ backgroundColor: colours[index] }}
          />
          <span className="text-[var(--fg-muted)]">{label.text}</span>
          <span className="ml-auto tabular-nums text-[var(--fg)]">
            {((summary.strategy[index] ?? 0) * 100).toFixed(1)}%
          </span>
        </div>
      ))}
      {summary.untrained > 0 && (
        // Travels with the number it qualifies, never below the fold: a total
        // over a fifth of the range is not the range's strategy.
        <div className="text-[11px] text-[var(--fg-faint)]">
          over {summary.trained} combos; {summary.untrained} untrained and left out
        </div>
      )}
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
  const colours = actionColours(actions.map((action) => action.token));
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
          <div key={actions[index]?.token ?? index} className="flex items-center gap-2">
            <span
              className="size-2.5 shrink-0 rounded-[2px]"
              style={{ backgroundColor: colours[index] }}
            />
            <span className="text-[var(--fg-muted)]">{actions[index]?.text}</span>
            <span className="ml-auto tabular-nums text-[var(--fg)]">
              {(weight * 100).toFixed(1)}%
            </span>
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
