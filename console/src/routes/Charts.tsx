import { getRouteApi, useNavigate } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { useBlueprintRun, useCombos, useSolverNode } from "@/api/queries";
import type { Edge, SolverNode } from "@/api/types";
import { BoardPicker } from "@/components/BoardPicker";
import { Panel } from "@/components/Panel";
import { PlayingCard } from "@/components/PlayingCard";
import { RangeGrid } from "@/components/RangeGrid";
import { type ActionLabel, describeAction, describeActions, inBlinds } from "@/lib/actions";
import {
  actionColours,
  aggregate,
  aggressiveFirst,
  bucketGroups,
  type Cell,
  type ComboRow,
  combosIn,
  type RangeSummary,
  summarise,
} from "@/lib/range";
import { cn } from "@/lib/utils";

const route = getRouteApi("/blueprint");

/**
 * The chart: the line across the top, the grid under it, the mix beside it.
 *
 * **The spot is in the URL.** `path`, `board` and `at` are search params, not
 * `useState`, which is what makes a bookmarked spot the spot it was and lets you
 * send someone one. `path` is the whole line; `at` is how much of it the chart
 * reads, so looking back at an earlier decision keeps the moves after it.
 *
 * **A street appears when the line reaches it, and not before.** The board is not
 * a thing you set up first; it is a column that arrives mid-line, the way cards
 * arrive mid-hand. Walk into the flop and the flop column is there, empty, with
 * the deck already open under it -- because a line that has crossed to the flop
 * has no strategy at all until someone says which flop.
 */
export function Charts() {
  const { path, board, stack, at } = route.useSearch();
  const line = path ? path.split("/") : [];
  // How many of the line's actions the chart reads; the whole line unless a past
  // spot was clicked.
  const reading = at === undefined ? line.length : Math.min(at, line.length);
  const read = line.slice(0, reading).join("/");
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
  // STACK IS PART OF THE SPOT. A pin taken at one depth must not survive a switch
  // to another: the grid would redraw from the new depth while the rail beside it
  // still showed the old one's strategy and combo count, under a label saying
  // "pinned" -- the same two-contradictory-answers bug the board once had.
  const spot = `${read}|${board}|${stack ?? "deepest"}`;
  const [held, setHeld] = useState<{ spot: string; cell: Cell } | null>(null);
  const pinned = held?.spot === spot ? held.cell : null;
  const [hovered, setHovered] = useState<Cell | null>(null);
  // Opening the deck by hand, for changing a street already dealt. A pending
  // street opens it on its own and this never has to be true for that.
  const [editing, setEditing] = useState(false);

  const run = useBlueprintRun();
  const combos = useCombos(!!run.data);
  // The whole line draws the sequence; the read prefix draws the chart. One
  // query when nothing earlier is being read, since the keys are then equal.
  const full = useSolverNode(path, board, !!run.data, stack);
  const node = useSolverNode(read, board, !!run.data, stack);

  const set = (
    next: Partial<{ path: string; board: string; stack: number; at: number | undefined }>,
  ) => navigate({ search: (old) => ({ ...old, ...next }) });
  /** Read the spot after `count` actions, leaving the line as it is. */
  const view = (count: number) => set({ at: count >= line.length ? undefined : count });
  /**
   * Take `token` at the spot after `count` actions. The action already on the
   * line just moves the reading on; any other one replaces the line from there.
   */
  const take = (count: number, token: string) => {
    setEditing(false);
    if (line[count] === token) view(count + 1);
    else set({ path: [...line.slice(0, count), token].join("/"), at: undefined });
  };

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
  /**
   * The combos behind the PINNED class, computed only when one is pinned.
   *
   * Pinned rather than hovered: this is a second reading of the same hand and
   * it appears under the grid, so making it follow the mouse would flicker
   * twelve rows in and out while you crossed the chart to reach a cell.
   */
  const breakdown = useMemo(() => {
    if (!pinned || !grid || !combos.data) return null;
    return combosIn(pinned.label, {
      combos: combos.data.combos,
      comboBuckets: grid.combo_buckets,
      buckets: grid.buckets,
      actionCount: grid.actions.length,
    });
  }, [pinned, grid, combos.data]);

  // Sizes are meaningless without the stakes, so labelling waits on /run.
  const bigBlind = run.data?.big_blind ?? 0;
  const labels = useMemo(
    () => describeActions(grid?.actions ?? [], node.data?.children ?? [], bigBlind),
    [grid?.actions, node.data?.children, bigBlind],
  );

  const shown = pinned ?? hovered;
  const pending = node.data?.pending ?? null;
  // The deck answers the LINE, which may be waiting on a street past the spot read.
  const linePending = full.data?.pending ?? null;
  const deckOpen = editing || linePending !== null;

  return (
    <div className="space-y-3">
      <Sequence
        node={full.data ?? null}
        reading={reading}
        bigBlind={bigBlind}
        onView={view}
        onTake={take}
        onEditBoard={() => setEditing((was) => !was)}
        editing={editing}
        depth={
          <Depth
            depths={run.data?.depths ?? []}
            stack={stack}
            bigBlind={run.data?.big_blind ?? 2}
            onSet={set}
          />
        }
      />

      {deckOpen && (
        <div className="rounded-md border border-[var(--border)] bg-[var(--panel)] p-3">
          {linePending && (
            <p className="mb-2 text-[12px] text-[var(--fg-muted)]">
              This line is on the <span className="text-[var(--fg)]">{linePending.street}</span>.
              Deal {linePending.needed - linePending.have} more card
              {linePending.needed - linePending.have === 1 ? "" : "s"} to see the strategy here.
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
          <div className="grid items-start gap-5 p-3 xl:grid-cols-[minmax(0,1fr)_19rem]">
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

            {/* Sticky, because the grid is taller than the screen and the rail
                is where every click is answered: pinning a hand in the bottom
                row put its mix off the top of the window. Scrolls inside itself
                on a short viewport rather than clipping the combo list. */}
            <aside className="space-y-3 text-[12px] xl:sticky xl:top-3 xl:max-h-[calc(100vh-1.5rem)] xl:overflow-y-auto">
              <Actions summary={summary} labels={labels} />
              <Coverage grid={grid} />
              <HandDetail cell={shown} actions={labels} pinned={pinned !== null} />
              {pinned && breakdown && <Combos rows={breakdown} actions={labels} />}
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
 * someone acted. Every column keeps the WHOLE menu that was on offer there.
 *
 * Clicking a column READS it: the chart shows what to do at that spot and the
 * moves after it stay on the line, dimmed. Inside the column being read the
 * actions are buttons, and taking one sets the line to that action from there.
 */
function Sequence({
  node,
  reading,
  bigBlind,
  onView,
  onTake,
  onEditBoard,
  editing,
  depth,
}: {
  node: SolverNode | null;
  /** How many of the line's actions the chart reads. */
  reading: number;
  bigBlind: number;
  onView: (count: number) => void;
  onTake: (count: number, token: string) => void;
  onEditBoard: () => void;
  editing: boolean;
  /** The depth picker, first in the row: the line starts from a stack. */
  depth: React.ReactNode;
}) {
  if (!node) {
    return (
      <div className="h-[5.5rem] rounded-md border border-[var(--border)] bg-[var(--panel)]" />
    );
  }

  // A spot column is numbered by the actions BEFORE it, which is the count the
  // chart reads to show it. Deals carry no action, so this counts spots rather
  // than using the column index.
  let count = 0;
  const columns = (node.line ?? []).map((step) => {
    const before = count;
    if (step.kind === "spot") count += 1;
    return { step, before };
  });
  const end = count;
  const button = node.button ?? 0;

  return (
    <div className="flex items-stretch gap-1.5 overflow-x-auto rounded-md border border-[var(--border)] bg-[var(--panel)] p-1.5">
      {depth}
      {columns.map(({ step, before }, index) =>
        step.kind === "spot" ? (
          <SpotColumn
            key={`${index}-${step.chosen}`}
            label={seatName(step.actor, button)}
            note={inBlinds(step.stack, bigBlind)}
            options={step.options}
            chosen={step.chosen}
            bigBlind={bigBlind}
            read={before === reading}
            ahead={before > reading}
            onView={() => onView(before)}
            onTake={(token) => onTake(before, token)}
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
            ahead={before > reading}
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
        <SpotColumn
          label={seatName(node.grid.actor, button)}
          note={
            node.stack === null || node.stack === undefined
              ? `${inBlinds(node.pot ?? 0, bigBlind)} pot`
              : inBlinds(node.stack, bigBlind)
          }
          options={node.children ?? []}
          chosen={null}
          bigBlind={bigBlind}
          read={reading === end}
          ahead={false}
          onView={() => onView(end)}
          onTake={(token) => onTake(end, token)}
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

/**
 * One decision: who, with how much, and every action they had, aggressive at
 * the top and fold at the bottom.
 *
 * Not being read, the whole card is one button that reads it. Being read, each
 * action is its own button, and the one already on the line is marked.
 */
function SpotColumn({
  label,
  note,
  options,
  chosen,
  bigBlind,
  read,
  ahead,
  onView,
  onTake,
}: {
  label: string;
  note: string;
  options: Edge[];
  /** The action the line took here; null at the end of the line. */
  chosen: string | null;
  bigBlind: number;
  read: boolean;
  /** Past the spot being read: still on the line, drawn dimmed. */
  ahead: boolean;
  onView: () => void;
  onTake: (token: string) => void;
}) {
  const tokens = options.map((option) => option.token);
  const colours = actionColours(tokens);
  const order = aggressiveFirst(tokens);
  return (
    <Column
      label={label}
      note={note}
      here={read}
      ahead={ahead}
      onGo={read ? undefined : onView}
      title="see what to do at this spot"
    >
      {options.length === 0 && (
        <span className="px-1.5 py-1 text-[11px] text-[var(--fg-faint)]">no actions</span>
      )}
      {order.map((index) => {
        const option = options[index];
        if (!option) return null;
        const text = describeAction(option, bigBlind).text;
        const colour = colours[index] ?? "transparent";
        const taken = option.token === chosen;
        return read ? (
          <ActionRow
            key={option.token}
            label={text}
            colour={colour}
            chosen={taken}
            title={taken ? `${text}, as the line does` : `${text} instead`}
            onClick={() => onTake(option.token)}
          />
        ) : (
          <span key={option.token} className={rowClass(taken, false)}>
            <Dot colour={colour} show={false} />
            {text}
          </span>
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
  ahead = false,
}: {
  street: string;
  cards: string[];
  slots?: number;
  pot: number | null;
  bigBlind: number;
  onEdit: () => void;
  editing: boolean;
  ahead?: boolean;
}) {
  return (
    <Column
      label={street}
      note={pot === null ? "" : `${inBlinds(pot, bigBlind)} pot`}
      here={editing}
      ahead={ahead}
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
 * A column not being read is one button covering the whole card, because every
 * part of it means the same thing -- show me this spot. Only the column being
 * read has per-action controls, and that is where taking an action lives.
 */
function Column({
  label,
  note,
  here = false,
  ahead = false,
  onGo,
  title,
  children,
}: {
  label: string;
  note: string;
  /** This is the spot being read, or the street being dealt. */
  here?: boolean;
  /** Later in the line than the spot being read. */
  ahead?: boolean;
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
    ahead && "opacity-45",
    onGo && "hover:border-[var(--fg-faint)] hover:bg-white/[0.05] hover:opacity-100",
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

/** The look of one action row, in a column being read or not. */
function rowClass(chosen: boolean, live: boolean): string {
  return cn(
    // Bordered on every row, transparent unless chosen, so taking one never shifts text.
    "flex items-center gap-2 rounded-[3px] border border-transparent py-1 pr-2 pl-1.5 text-left text-[12px] leading-tight transition-colors",
    // Chosen is the table's pressed preset chip (play.html `.preset[aria-pressed]`).
    chosen && "border-black bg-[linear-gradient(#6a6a6a,#4a4a4a)] font-semibold text-white",
    chosen && live && "hover:brightness-110",
    !chosen && live && "text-[var(--fg-muted)] hover:bg-white/[0.05] hover:text-[var(--fg)]",
    !chosen && !live && "text-[var(--fg-faint)]",
  );
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
  chosen,
  title,
  onClick,
}: {
  label: string;
  colour: string;
  /** The action the line already takes here. */
  chosen: boolean;
  title: string;
  onClick: () => void;
}) {
  return (
    <button type="button" title={title} onClick={onClick} className={rowClass(chosen, true)}>
      <Dot colour={colour} show />
      {label}
    </button>
  );
}

/**
 * Which DEPTH of the ladder the grid reads.
 *
 * Absent when the server holds one run, which is the ordinary case on a box that
 * is not also seating chipzen -- a selector offering one option is noise. The
 * stacks come from `/api/run`, deepest first, and are labelled in BIG BLINDS
 * because that is the unit the strategy is actually about: a 6 bb range is a
 * different game from a 100 bb one, and "12 chips" says that far less clearly.
 */
function Depth({
  depths,
  stack,
  bigBlind,
  onSet,
}: {
  depths: number[];
  stack?: number;
  bigBlind: number;
  onSet: (next: { stack: number }) => void;
}) {
  if (depths.length < 2) return null;
  const current = stack ?? depths[0];
  return (
    <Column label="depth" note="">
      {depths.map((chips) => {
        // From the server's own big blind, never a hardcoded 2: the label is
        // the only thing telling you which game this range belongs to.
        const text = `${Math.round(chips / bigBlind)} bb`;
        return (
          <ActionRow
            key={chips}
            label={text}
            colour="var(--fg-faint)"
            chosen={chips === current}
            title={`the ${text} blueprint`}
            onClick={() => onSet({ stack: chips })}
          />
        );
      })}
    </Column>
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
  const tokens = labels.map((label) => label.token);
  const colours = actionColours(tokens);
  return (
    <div className="space-y-2">
      <div className="text-[11px] tracking-widest text-[var(--fg-faint)] uppercase">
        whole range
      </div>
      <div className="space-y-2">
        {aggressiveFirst(tokens).map((index) => {
          const label = labels[index];
          if (!label) return null;
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
  const tokens = actions.map((action) => action.token);
  const colours = actionColours(tokens);
  const strategy = cell.strategy;
  return (
    <div className="min-h-[7rem] space-y-1 border-t border-[var(--border)] pt-2">
      <div className="flex items-baseline gap-2">
        <span className="font-mono text-[14px] text-[var(--fg)]">{cell.label}</span>
        <span className="text-[11px] text-[var(--fg-faint)]">{cell.combos} combos</span>
        {pinned && <span className="ml-auto text-[11px] text-[var(--fg-faint)]">pinned</span>}
      </div>
      {cell.combos === 0 ? (
        <div className="text-[var(--fg-faint)]">blocked by the board</div>
      ) : strategy ? (
        aggressiveFirst(tokens).map((index) => (
          <div key={tokens[index] ?? index} className="flex items-center gap-2">
            <span
              className="size-2.5 shrink-0 rounded-[2px]"
              style={{ backgroundColor: colours[index] }}
            />
            <span className="text-[var(--fg-muted)]">{actions[index]?.text}</span>
            <span className="ml-auto tabular-nums text-[var(--fg)]">
              {((strategy[index] ?? 0) * 100).toFixed(1)}%
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

/**
 * The hands behind one square, once you have pinned it.
 *
 * The chart's unit is a CLASS and a player's unit is a hand. `AKs` is four of
 * them and `AKo` is twelve, the board blocks them unevenly, and a cell reading
 * `62% raise` can be four combos raising 62% or two raising always and two
 * folding always -- a difference the average is built to hide and the one that
 * decides what you do with the two cards you actually hold.
 *
 * In the rail, under the class average it breaks down, so the two readings of
 * one hand are next to each other and the grid keeps its width. One row per
 * hand: the cards, then the same stacked bar the square above is drawn from.
 * The action names are not repeated per row -- they are in the mix directly
 * above, four inches away, and twelve copies of `raise to 4.5bb` would be the
 * widest thing in a 19rem column.
 */
function Combos({ rows, actions }: { rows: ComboRow[]; actions: ActionLabel[] }) {
  const tokens = actions.map((action) => action.token);
  const colours = actionColours(tokens);
  const order = aggressiveFirst(tokens);
  const groups = bucketGroups(rows);
  const playable = rows.filter((row) => !row.blocked).length;

  return (
    <section className="border-t border-[var(--border)] pt-2">
      <div className="mb-1.5 flex items-baseline gap-2">
        <h3 className="text-[11px] tracking-wide text-[var(--fg-muted)]">by combo</h3>
        <span className="text-[11px] text-[var(--fg-faint)]">
          {playable} of {rows.length} playable
        </span>
      </div>

      <ul>
        {rows.map((row) => (
          <ComboRowView
            key={row.combo}
            row={row}
            colours={colours}
            order={order}
            group={groups.size > 1 ? (groups.get(row.bucket) ?? null) : null}
          />
        ))}
      </ul>

      {/* The answer to "do the suits matter here", said once rather than implied
          twelve times by identical bars. Preflop the abstraction is 169 buckets
          for 169 classes, so one group is the ordinary case, not a fault. */}
      {playable > 0 && (
        <p className="mt-1.5 text-[11px] text-[var(--fg-faint)]">
          {groups.size === 1
            ? "one bucket — the solver plays these suits identically"
            : `${groups.size} buckets — the numbered groups are played alike`}
        </p>
      )}
    </section>
  );
}

/** One hand: its two cards and what the solver does with it. */
function ComboRowView({
  row,
  colours,
  order,
  group,
}: {
  row: ComboRow;
  colours: string[];
  /** Menu indices in drawing order, aggressive first. */
  order: number[];
  /** Which set of look-alikes this combo is in, or null when they all match. */
  group: number | null;
}) {
  return (
    <li className="flex items-center gap-1.5 py-[3px]">
      <span className={cn("flex shrink-0 gap-0.5", row.blocked && "opacity-25")}>
        <PlayingCard card={row.cards[0]} size="sm" />
        <PlayingCard card={row.cards[1]} size="sm" />
      </span>
      {group !== null && !row.blocked && (
        <span
          className="w-3 shrink-0 text-center font-mono text-[10px] text-[var(--fg-faint)]"
          title="combos sharing a number are one bucket to the solver — it plays them identically"
        >
          {group}
        </span>
      )}
      {row.blocked ? (
        <span className="text-[11px] text-[var(--fg-faint)]">on the board</span>
      ) : row.strategy ? (
        <span className="flex h-4 min-w-0 flex-1 overflow-hidden rounded-[2px]">
          {order.map((index) => {
            const weight = row.strategy?.[index] ?? 0;
            return (
              <span
                key={index}
                style={{ backgroundColor: colours[index], width: `${weight * 100}%` }}
                title={`${(weight * 100).toFixed(1)}%`}
              />
            );
          })}
        </span>
      ) : (
        <span className="text-[11px] text-[var(--fg-faint)]">never trained</span>
      )}
    </li>
  );
}
