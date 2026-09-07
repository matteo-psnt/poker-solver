import { getRouteApi, useNavigate } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { useBlueprintRun, useCombos, useSolverNode } from "@/api/queries";
import type { Edge, SolverNode, Spot } from "@/api/types";
import { BoardPicker } from "@/components/BoardPicker";
import { Panel } from "@/components/Panel";
import { PlayingCard } from "@/components/PlayingCard";
import { RangeGrid } from "@/components/RangeGrid";
import { type ActionLabel, describeAction, describeActions, inBlinds } from "@/lib/actions";
import { actionColours, aggregate, type Cell, type RangeSummary, summarise } from "@/lib/range";
import { cn } from "@/lib/utils";

const route = getRouteApi("/blueprint");

/**
 * The chart: the line across the top, the grid under it, the mix beside it.
 *
 * **The spot is in the URL.** `path`, `board` and `average` are search params, not
 * `useState`, which is what makes a bookmarked spot the spot it was and lets you
 * send someone one.
 *
 * **A street appears when the line reaches it, and not before.** The board is not
 * a thing you set up first; it is a column that arrives mid-line, the way cards
 * arrive mid-hand. Walk into the flop and the flop column is there, empty, with
 * the deck already open under it -- because a line that has crossed to the flop
 * has no strategy at all until someone says which flop.
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
   */
  const spot = `${path}|${board}|${average}`;
  const [held, setHeld] = useState<{ spot: string; cell: Cell } | null>(null);
  const pinned = held?.spot === spot ? held.cell : null;
  const [hovered, setHovered] = useState<Cell | null>(null);
  // Opening the deck by hand, for changing a street already dealt. A pending
  // street opens it on its own and this never has to be true for that.
  const [editing, setEditing] = useState(false);

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

  const shown = pinned ?? hovered;
  const pending = node.data?.pending ?? null;
  const deckOpen = editing || pending !== null;

  return (
    <div className="space-y-3">
      <Sequence
        node={node.data ?? null}
        bigBlind={bigBlind}
        onGo={(next) => {
          setEditing(false);
          set({ path: next });
        }}
        onEditBoard={() => setEditing((was) => !was)}
        editing={editing}
      />

      {deckOpen && (
        <div className="rounded-md border border-[var(--border)] bg-[var(--panel)] p-3">
          {pending && (
            <p className="mb-2 text-[12px] text-[var(--fg-muted)]">
              This line is on the <span className="text-[var(--fg)]">{pending.street}</span>. Deal{" "}
              {pending.needed - pending.have} more card
              {pending.needed - pending.have === 1 ? "" : "s"} to see the strategy here.
            </p>
          )}
          <BoardPicker
            board={board}
            onChange={(next) => set({ board: next })}
            live={node.data?.board.length ?? null}
            forceOpen
          />
        </div>
      )}

      <Panel
        title={
          grid ? `${grid.street} · ${seatName(grid.actor, node.data?.button ?? 0)} to act` : "chart"
        }
        aside={<Average average={average} onSet={set} />}
        // `combos` is fetched once with staleTime Infinity and is not retried,
        // so a 503 from the blueprint box would otherwise blank the grid for the
        // life of the tab with no reason given anywhere.
        error={
          (node.error && String(node.error.message)) ||
          (combos.error && String(combos.error.message)) ||
          null
        }
        loading={node.isFetching && !node.data}
        empty={
          node.data?.terminal
            ? "Nobody acts here — the hand is already over."
            : pending
              ? `Waiting on the ${pending.street}.`
              : null
        }
      >
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
              <Actions summary={summary} labels={labels} />
              <Coverage grid={grid} />
              <HandDetail cell={shown} actions={labels} pinned={pinned !== null} />
            </aside>
          </div>
        ) : undefined}
      </Panel>
    </div>
  );
}

/** Who a seat is, rather than which index it has. */
function seatName(seat: number, button: number): string {
  // Heads-up, the button posts the small blind — the same naming `Play` uses,
  // and the server sends the button precisely so this is not a guess.
  return seat === button ? "BTN" : "BB";
}

/**
 * The line, as one column per thing that happened.
 *
 * Read left to right it is the hand: someone acted, someone acted, the flop came,
 * someone acted. Every past column keeps the WHOLE menu that was on offer there,
 * so stepping back into a spot and taking the other branch is one click and no
 * round trip — the server sent the options with the line for that reason.
 *
 * The last column is where you are, and it is the only one with nothing chosen
 * in it yet.
 */
function Sequence({
  node,
  bigBlind,
  onGo,
  onEditBoard,
  editing,
}: {
  node: SolverNode | null;
  bigBlind: number;
  onGo: (path: string) => void;
  onEditBoard: () => void;
  editing: boolean;
}) {
  if (!node) {
    return (
      <div className="h-[5.5rem] rounded-md border border-[var(--border)] bg-[var(--panel)]" />
    );
  }

  // A column's path is the tokens BEFORE it -- the line that leads TO this
  // spot, which is where clicking it goes. Looking at a spot never changes the
  // line: you step back into one and then take an action, rather than editing a
  // decision in place. Deals carry no token, which is why this counts spots
  // rather than using the column index.
  const tokens: string[] = [];
  const columns = (node.line ?? []).map((step) => {
    const before = tokens.join("/");
    if (step.kind === "spot") tokens.push(step.chosen);
    return { step, before };
  });

  return (
    <div className="flex items-stretch gap-1.5 overflow-x-auto rounded-md border border-[var(--border)] bg-[var(--panel)] p-1.5">
      {columns.map(({ step, before }, index) =>
        step.kind === "spot" ? (
          <SpotColumn
            key={`${index}-${step.chosen}`}
            spot={step}
            button={node.button ?? 0}
            bigBlind={bigBlind}
            onGo={() => onGo(before)}
          />
        ) : (
          <DealColumn
            key={`${index}-${step.street}`}
            street={step.street}
            cards={step.cards}
            pot={step.pot}
            bigBlind={bigBlind}
            onEdit={onEditBoard}
            editing={editing}
          />
        ),
      )}

      {node.pending && (
        <DealColumn
          street={node.pending.street}
          cards={[]}
          slots={node.pending.needed - node.pending.have}
          pot={null}
          bigBlind={bigBlind}
          onEdit={onEditBoard}
          editing
        />
      )}

      {!node.pending && !node.terminal && node.grid && (
        <HereColumn
          actor={node.grid.actor}
          button={node.button ?? 0}
          pot={node.pot ?? 0}
          stack={node.stack ?? null}
          bigBlind={bigBlind}
          options={node.children ?? []}
          onPick={(token) => onGo(node.path ? `${node.path}/${token}` : token)}
        />
      )}

      {node.terminal && (
        <div className="flex min-w-[8rem] items-center px-3 text-[11px] text-[var(--fg-faint)]">
          hand over
        </div>
      )}
    </div>
  );
}

/** One past decision: who, with how much, and every action they had. */
function SpotColumn({
  spot,
  button,
  bigBlind,
  onGo,
}: {
  spot: Spot;
  button: number;
  bigBlind: number;
  /** Read the chart at this spot. The line is not changed by looking at it. */
  onGo: () => void;
}) {
  const colours = actionColours(spot.options.map((option) => option.token));
  return (
    <Column
      label={seatName(spot.actor, button)}
      note={inBlinds(spot.stack, bigBlind)}
      onGo={onGo}
      title="read the chart at this spot"
    >
      {spot.options.map((option, index) => (
        <span
          key={option.token}
          className={rowClass(option.token === spot.chosen, false)}
          style={rowStyle(option.token === spot.chosen, colours[index] ?? "transparent")}
        >
          <Dot colour={colours[index] ?? "transparent"} show={false} />
          {describeAction(option.token, bigBlind).text}
        </span>
      ))}
    </Column>
  );
}

/** Where you are: the same column shape, with nothing taken yet. */
function HereColumn({
  actor,
  button,
  pot,
  stack,
  bigBlind,
  options,
  onPick,
}: {
  actor: number;
  button: number;
  pot: number;
  stack: number | null;
  bigBlind: number;
  options: Edge[];
  onPick: (token: string) => void;
}) {
  const colours = actionColours(options.map((option) => option.token));
  return (
    <Column
      label={seatName(actor, button)}
      note={stack === null ? `${inBlinds(pot, bigBlind)} pot` : inBlinds(stack, bigBlind)}
      here
    >
      {options.length === 0 && (
        <span className="px-1.5 py-1 text-[11px] text-[var(--fg-faint)]">no actions</span>
      )}
      {options.map((option, index) => {
        const label = describeAction(option.token, bigBlind).text;
        return (
          <ActionRow
            key={option.token}
            label={label}
            colour={colours[index] ?? "transparent"}
            title={`${label}, and read the spot after it`}
            onClick={() => onPick(option.token)}
          />
        );
      })}
    </Column>
  );
}

/**
 * A street, where it happens in the line.
 *
 * `slots` draws the cards this street is still waiting for. That is the whole
 * point of putting the board in the line rather than in a bar above it: at the
 * preflop there is no flop column at all, and the moment a line crosses into the
 * flop one appears with three empty frames in it.
 */
function DealColumn({
  street,
  cards,
  slots = 0,
  pot,
  bigBlind,
  onEdit,
  editing,
}: {
  street: string;
  cards: string[];
  slots?: number;
  pot: number | null;
  bigBlind: number;
  onEdit: () => void;
  editing: boolean;
}) {
  return (
    <Column
      label={street}
      note={pot === null ? "" : `${inBlinds(pot, bigBlind)} pot`}
      here={editing}
    >
      <button
        type="button"
        onClick={onEdit}
        title="change this street"
        className="flex flex-1 items-center justify-center gap-1 rounded px-1 py-2 hover:bg-white/[0.06]"
      >
        {cards.map((card) => (
          <PlayingCard key={card} card={card} size="sm" />
        ))}
        {/* Brighter than `PlayingCard`'s empty slot, which is a placeholder in
            a row of real cards. These are a QUESTION -- three cards the line is
            waiting on -- and have to read as one from across the page. */}
        {Array.from({ length: slots }, (_, index) => (
          <span
            key={`slot-${index}`}
            className="h-7 w-5 rounded-[3px] border border-dashed border-[var(--fg-faint)] bg-white/[0.03]"
          />
        ))}
      </button>
    </Column>
  );
}

/**
 * The frame every column shares: a header, then rows.
 *
 * A PAST column is one button covering the whole card, because every part of it
 * means the same thing -- show me this spot. Only the column you are standing in
 * has per-action controls, and that is where taking an action lives.
 */
function Column({
  label,
  note,
  here = false,
  onGo,
  title,
  children,
}: {
  label: string;
  note: string;
  /** This is the spot being read, or the street being dealt. */
  here?: boolean;
  /** Makes the whole card a link to one spot. Omit for the current column. */
  onGo?: () => void;
  title?: string;
  children: React.ReactNode;
}) {
  const frame = cn(
    "flex min-w-[8.75rem] shrink-0 flex-col rounded-[4px] border text-left",
    // Every column is a card, so the line reads as columns rather than as
    // one wide table; the spot being READ is the lit one among them.
    here ? "border-[var(--fg-faint)] bg-white/[0.05]" : "border-[var(--border)] bg-white/[0.015]",
    onGo && "hover:border-[var(--fg-faint)] hover:bg-white/[0.05]",
  );
  const inside = (
    <>
      <div className="flex items-baseline justify-between gap-2 border-b border-[var(--border)] px-2 py-1.5">
        <span
          className={cn(
            "font-mono text-[11px] tracking-widest uppercase",
            here ? "text-[var(--fg)]" : "text-[var(--fg-muted)]",
          )}
        >
          {label}
        </span>
        <span className="font-mono text-[11px] tabular-nums text-[var(--fg-faint)]">{note}</span>
      </div>
      <div className="flex flex-1 flex-col gap-0.5 p-1">{children}</div>
    </>
  );
  return onGo ? (
    <button type="button" onClick={onGo} title={title} className={frame}>
      {inside}
    </button>
  ) : (
    <div className={frame}>{inside}</div>
  );
}

/** The look of one action row, shared by the past columns and the live one. */
function rowClass(chosen: boolean, live: boolean): string {
  return cn(
    "flex items-center gap-2 rounded-[3px] border-l-2 py-1 pr-2 pl-1.5 text-left text-[12px] leading-tight transition-colors",
    chosen && "font-medium text-[var(--fg)]",
    !chosen && live && "border-l-transparent text-[var(--fg-muted)] hover:text-[var(--fg)]",
    !chosen && !live && "border-l-transparent text-[var(--fg-faint)]",
    live && !chosen && "hover:bg-white/[0.05]",
  );
}

/** The action's own colour, so a fold reads blue here as it does in the grid. */
function rowStyle(chosen: boolean, colour: string) {
  return chosen ? { borderLeftColor: colour, backgroundColor: `${colour}26` } : undefined;
}

/** Drawn on every row so the text starts at the same x, coloured only when live. */
function Dot({ colour, show }: { colour: string; show: boolean }) {
  return (
    <span
      className="size-1.5 shrink-0 rounded-full"
      style={{ backgroundColor: show ? colour : "transparent" }}
    />
  );
}

/** One action you can take from the spot being read. */
function ActionRow({
  label,
  colour,
  title,
  onClick,
}: {
  label: string;
  colour: string;
  title: string;
  onClick: () => void;
}) {
  return (
    <button type="button" title={title} onClick={onClick} className={rowClass(false, true)}>
      <Dot colour={colour} show />
      {label}
    </button>
  );
}

/** The average/current toggle, which belongs to the numbers rather than the line. */
function Average({
  average,
  onSet,
}: {
  average: boolean;
  onSet: (next: { average: boolean }) => void;
}) {
  return (
    <label className="flex items-center gap-2 text-[11px] text-[var(--fg-muted)]">
      <input
        type="checkbox"
        checked={average}
        onChange={(event) => onSet({ average: event.target.checked })}
      />
      average strategy
      {/* Not a detail: the average is the blueprint and what converges; the
          current strategy is regret-matching's latest guess, and on an
          under-trained run they disagree sharply. */}
      <span className="text-[var(--fg-faint)]">
        {average ? "the blueprint proper" : "regret-matched current guess"}
      </span>
    </label>
  );
}

/**
 * What the range DOES, as one tile per action.
 *
 * First in the rail because it is the first question — "does this spot fold a
 * lot" — and the grid is bad at it: a class gets one square whether it holds 4
 * combos or 12, so an offsuit-heavy fold looks smaller than it is. See
 * `summarise`, which weights by combos for exactly that reason.
 *
 * Tiles rather than a legend, sized by the number they carry, because the
 * frequency is the answer and a 4% action should not read the same as a 53% one.
 */
function Actions({ summary, labels }: { summary: RangeSummary | null; labels: ActionLabel[] }) {
  if (!summary) {
    return (
      <div className="text-[var(--fg-faint)]">Nothing here was trained — no range to total.</div>
    );
  }
  const colours = actionColours(labels.map((label) => label.token));
  return (
    <div className="space-y-2">
      <div className="text-[11px] tracking-widest text-[var(--fg-faint)] uppercase">
        whole range
      </div>
      <div className="space-y-2">
        {labels.map((label, index) => {
          const weight = summary.strategy[index] ?? 0;
          return (
            <div key={label.token} className="space-y-1">
              <div className="flex items-baseline gap-2">
                <span
                  className="size-2 shrink-0 rounded-[2px]"
                  style={{ backgroundColor: colours[index] }}
                />
                <span className="truncate text-[12px] text-[var(--fg-muted)]">{label.text}</span>
                <span className="ml-auto font-mono text-[13px] tabular-nums text-[var(--fg)]">
                  {(weight * 100).toFixed(1)}%
                </span>
              </div>
              {/* The bar sits UNDER the line rather than behind it. Behind it,
                  the fill edge landed mid-word and the label had to be read
                  through two different backgrounds. */}
              <span className="block h-1.5 overflow-hidden rounded-full bg-white/[0.06]">
                <span
                  className="block h-full rounded-full"
                  style={{ width: `${weight * 100}%`, backgroundColor: colours[index] }}
                />
              </span>
            </div>
          );
        })}
      </div>
      <div className="text-[11px] text-[var(--fg-faint)]">
        over {summary.trained.toLocaleString()} combos
        {/* Travels with the number it qualifies, never below the fold: a total
            over a fifth of the range is not the range's strategy. */}
        {summary.untrained > 0 && `; ${summary.untrained.toLocaleString()} untrained and left out`}
      </div>
    </div>
  );
}

/**
 * How much of this spot the solver learned, and how finely it can answer.
 *
 * The resolution line is the one that stops a flat grid reading as a fault. The
 * solver plays a BUCKET, so every combo sharing one carries an identical row:
 * where 1,176 combos map to 44 buckets there are 44 possible answers on screen,
 * and hands the abstraction cannot tell apart are drawn alike because they ARE
 * alike to the strategy. Without the count that looks like a broken chart.
 */
function Coverage({
  grid,
}: {
  grid: {
    buckets: Record<string, unknown>;
    trained_buckets: number;
    blocked: number;
    combo_buckets: number[];
  };
}) {
  const total = Object.keys(grid.buckets).length;
  const fraction = total ? grid.trained_buckets / total : 0;
  const live = grid.combo_buckets.length - grid.blocked;
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
      {/* One string, not four expressions: split across nodes it is neither
          readable in the DOM nor assertable in a test. */}
      <p className="text-[11px] leading-snug text-[var(--fg-faint)]">{resolution(live, total)}</p>
      <div className="flex justify-between gap-2">
        <span className="text-[var(--fg-faint)]">blocked</span>
        <span className="tabular-nums text-[var(--fg-muted)]">{grid.blocked} combos</span>
      </div>
    </div>
  );
}

/** How many different answers this spot can possibly show, and why. */
function resolution(live: number, buckets: number): string {
  const combos = `${live.toLocaleString()} combos over ${buckets.toLocaleString()}`;
  return buckets === 1
    ? `${combos} bucket — every hand here shares one mix.`
    : `${combos} buckets — at most ${buckets.toLocaleString()} different mixes on screen.`;
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
