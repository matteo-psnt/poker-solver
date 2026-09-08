/**
 * Turning 1326 combos into the 13x13 grid a poker player reads.
 *
 * The solver's unit is a bucket, the wire's unit is a combo, and the unit a person
 * recognises is a hand CLASS -- "AKs", "72o", "TT". This module is the only place
 * those three meet.
 *
 * A class holds up to 12 combos and the board may block some, so a cell averages
 * over the combos still POSSIBLE and a fully blocked class is drawn as ABSENT
 * rather than as a zero row: "you cannot hold this" and "you fold this" must not
 * render the same.
 *
 * Averaging is unweighted, which is right because every combo of a class is
 * equally likely a priori. It would be wrong the moment a range weight enters, and
 * that is where this stops being a display concern.
 */

const RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"] as const;

export const GRID_SIZE = RANKS.length;

const RANK_INDEX: ReadonlyMap<string, number> = new Map<string, number>(
  RANKS.map((rank, index) => [rank, index]),
);

/** Where a combo like `"AsKd"` belongs in the grid, or null if unparseable. */
export function cellFor(combo: string): { row: number; col: number } | null {
  if (combo.length !== 4) return null;
  const highRank = RANK_INDEX.get(combo.slice(0, 1));
  const lowRank = RANK_INDEX.get(combo.slice(2, 3));
  if (highRank === undefined || lowRank === undefined) return null;

  const suited = combo.slice(1, 2) === combo.slice(3, 4);
  // The convention every solver UI uses: high rank first, suited above the
  // diagonal, offsuit below. Pairs land on the diagonal either way.
  const [hi, lo] = highRank <= lowRank ? [highRank, lowRank] : [lowRank, highRank];
  return suited ? { row: hi, col: lo } : { row: lo, col: hi };
}

export function classLabel(row: number, col: number): string {
  const hi = RANKS[Math.min(row, col)] ?? "?";
  const lo = RANKS[Math.max(row, col)] ?? "?";
  if (row === col) return `${hi}${lo}`;
  return row < col ? `${hi}${lo}s` : `${hi}${lo}o`;
}

export interface Cell {
  label: string;
  /** Mean strategy over the class's still-possible combos, in `actions` order. */
  strategy: number[] | null;
  /** How many of the class's combos survive the board. */
  combos: number;
  /** How many of those sit in a bucket training never visited. */
  untrained: number;
}

export interface AggregateInput {
  combos: string[];
  comboBuckets: number[];
  buckets: Record<string, { trained: boolean; strategy: number[] | null }>;
  actionCount: number;
}

/**
 * Fold the per-combo payload into a 13x13 array of cells.
 *
 * A cell whose combos are all blocked returns `combos: 0` and no strategy; a
 * cell whose combos are all untrained returns `combos > 0` with no strategy. The
 * caller renders those two differently, which is the whole reason they are
 * distinguishable here.
 */
export function aggregate({
  combos,
  comboBuckets,
  buckets,
  actionCount,
}: AggregateInput): Cell[][] {
  interface Tally {
    total: number[] | null;
    combos: number;
    untrained: number;
  }
  // Keyed by "row,col" rather than nested arrays: with
  // `noUncheckedIndexedAccess` every nested read is an `| undefined` to unwrap,
  // and a Map makes the one absent case explicit instead of thirty.
  const tallies = new Map<string, Tally>();
  const keyOf = (row: number, col: number) => `${row},${col}`;

  for (let index = 0; index < comboBuckets.length; index++) {
    const bucket = comboBuckets[index] ?? -1;
    // -1 is the board blocking this combo: it is not a hand anyone can hold
    // here, so it contributes to nothing, not even the denominator.
    if (bucket < 0) continue;
    const cell = cellFor(combos[index] ?? "");
    if (!cell) continue;

    const key = keyOf(cell.row, cell.col);
    const tally = tallies.get(key) ?? { total: null, combos: 0, untrained: 0 };
    tally.combos += 1;

    const entry = buckets[String(bucket)];
    if (!entry?.trained || !entry.strategy) {
      tally.untrained += 1;
    } else {
      const running = tally.total ?? Array.from({ length: actionCount }, () => 0);
      for (let a = 0; a < actionCount; a++) {
        running[a] = (running[a] ?? 0) + (entry.strategy[a] ?? 0);
      }
      tally.total = running;
    }
    tallies.set(key, tally);
  }

  const grid: Cell[][] = [];
  for (let row = 0; row < GRID_SIZE; row++) {
    const line: Cell[] = [];
    for (let col = 0; col < GRID_SIZE; col++) {
      const tally = tallies.get(keyOf(row, col));
      const trained = (tally?.combos ?? 0) - (tally?.untrained ?? 0);
      const total = tally?.total ?? null;
      line.push({
        label: classLabel(row, col),
        strategy: total && trained > 0 ? total.map((sum) => sum / trained) : null,
        combos: tally?.combos ?? 0,
        untrained: tally?.untrained ?? 0,
      });
    }
    grid.push(line);
  }
  return grid;
}

export interface RangeSummary {
  /** Mean frequency of each action over the whole range, in `actions` order. */
  strategy: number[];
  /** Combos that contributed — the denominator, and the caveat. */
  trained: number;
  /** Combos the board allows but training never reached. */
  untrained: number;
}

/**
 * What this range DOES, in one line, before you read any single hand.
 *
 * The grid answers "what do I do with AKs" and answers it well; it is poor at
 * "does this spot fold a lot", which is the first thing anyone actually asks of
 * a strategy and previously had to be estimated by looking at how red the
 * squares were. That estimate is wrong in a specific way — the grid gives a
 * class one square whether it holds 4 combos or 12 — so the eye reads an
 * offsuit-heavy fold as smaller than it is.
 *
 * Weighted by combos for that reason, where a CELL is unweighted across its own
 * combos. Both follow the same rule: every combo is equally likely a priori, so
 * a class holding three times as many of them counts three times.
 *
 * Untrained combos are excluded from the mean and reported separately rather
 * than folded in as a uniform. A spot that is 80% untrained has no aggregate
 * worth reading, and the number that says so has to travel with the number it
 * disqualifies.
 */
export function summarise(cells: Cell[][], actionCount: number): RangeSummary | null {
  const totals = Array.from({ length: actionCount }, () => 0);
  let trained = 0;
  let untrained = 0;

  for (const row of cells) {
    for (const cell of row) {
      untrained += cell.untrained;
      const contributing = cell.combos - cell.untrained;
      if (!cell.strategy || contributing <= 0) continue;
      trained += contributing;
      for (let action = 0; action < actionCount; action++) {
        totals[action] = (totals[action] ?? 0) + (cell.strategy[action] ?? 0) * contributing;
      }
    }
  }

  if (trained === 0) return null;
  return { strategy: totals.map((total) => total / trained), trained, untrained };
}

/**
 * The colour of every action in a menu, computed together.
 *
 * The fold-blue / call-green / raise-red convention is what every solver uses,
 * so a poker player reads the grid without consulting a legend. Multiple raise
 * sizes shade from light to dark by size — bigger is darker, which is the only
 * ordering the eye reads for free.
 *
 * Computed over the WHOLE menu rather than one token at a time, because the
 * shade of a raise depends on how many OTHER raises there are. The previous
 * version spread `0.35 → 0.85` across every action's index, so in the ordinary
 * `f c r6 r9 A` menu the two raise sizes landed at 0.60 and 0.725 — a 12% step
 * between two mixes of the same two reds, which on the grid is no difference at
 * all. Ranking raises among raises gives them the full range.
 */
export function actionColours(tokens: readonly string[]): string[] {
  const isRaise = (token: string) => {
    const kind = token[0];
    return kind === "b" || kind === "r";
  };
  const raises = tokens.filter(isRaise);
  // `- 1` guards the single-size case against a divide by zero that would
  // render the only raise at the lightest end of the ramp.
  const steps = Math.max(raises.length - 1, 1);

  return tokens.map((token) => {
    const kind = token[0];
    if (kind === "f") return "#3D6FC4";
    if (kind === "x" || kind === "c") return "#2F7F58";
    // The jam is the darkest red there is: it is the top of the size ladder,
    // not a member of it.
    if (kind === "A") return "#6E1F1A";
    if (!isRaise(token)) return "#7A7A85";
    const depth = 0.15 + 0.7 * (raises.indexOf(token) / steps);
    return `color-mix(in srgb, #B23A32 ${Math.round(depth * 100)}%, #E8837C)`;
  });
}

export interface ComboRow {
  /** The wire spelling, `"AsKs"` — the same one the board and the URL use. */
  combo: string;
  cards: [string, string];
  /** Which bucket the solver put it in, or -1 when the board blocks it. */
  bucket: number;
  strategy: number[] | null;
  blocked: boolean;
  /** Reachable, but training never visited its bucket. */
  untrained: boolean;
}

/** Spades, hearts, diamonds, clubs — the order `lib/cards` draws a deck in. */
const SUIT_ORDER: ReadonlyMap<string, number> = new Map(
  ["s", "h", "d", "c"].map((suit, index) => [suit, index]),
);

function suitRank(card: string): number {
  return SUIT_ORDER.get(card.slice(1, 2)) ?? SUIT_ORDER.size;
}

/**
 * A combo's two cards, high rank first.
 *
 * The wire spells a combo in deck order, so `AKs` arrives as `"KsAs"` and
 * rendering it verbatim puts the king in front of the ace -- which no player
 * writes and the chart's own label contradicts two lines above it. `cellFor`
 * is already order-agnostic; this is the display half of the same fact.
 */
function highFirst(combo: string): [string, string] {
  const [first, second]: [string, string] = [combo.slice(0, 2), combo.slice(2, 4)];
  const rankOf = (card: string) => RANK_INDEX.get(card.slice(0, 1)) ?? RANKS.length;
  return rankOf(first) <= rankOf(second) ? [first, second] : [second, first];
}

/**
 * The individual combos behind one cell of the grid.
 *
 * A cell is an AVERAGE over up to twelve hands, and the average is the thing a
 * player reads first but not the thing they act on: `AKs` is four hands and
 * `AKo` is twelve, the board blocks them unevenly, and once a flush is possible
 * the solver need not treat two suits alike. A cell that reads `62% raise` can
 * be four combos raising 62% or two raising always and two folding always, and
 * nothing on the chart could tell those apart.
 *
 * Ordered by suit rather than left in wire order, so the same hand lists its
 * combos the same way in every spot and two spots can be compared by position.
 *
 * Blocked combos are KEPT and marked. They are the reason a cell holds fewer
 * hands than its class does, and dropping them here would leave the count
 * unexplained in the one place with room to explain it.
 */
export function combosIn(label: string, input: AggregateInput): ComboRow[] {
  const { combos, comboBuckets, buckets, actionCount } = input;
  const rows: ComboRow[] = [];

  for (let index = 0; index < combos.length; index++) {
    const combo = combos[index] ?? "";
    const at = cellFor(combo);
    if (!at || classLabel(at.row, at.col) !== label) continue;

    const bucket = comboBuckets[index] ?? -1;
    const entry = bucket < 0 ? undefined : buckets[String(bucket)];
    const trained = Boolean(entry?.trained && entry.strategy);
    rows.push({
      combo,
      cards: highFirst(combo),
      bucket,
      // Sliced to the menu's width: a bucket carries as many weights as the
      // node has actions, and a shorter one would misalign against the labels.
      strategy: trained ? (entry?.strategy ?? []).slice(0, actionCount) : null,
      blocked: bucket < 0,
      untrained: bucket >= 0 && !trained,
    });
  }

  return rows.sort((a, b) => {
    const [aHi, aLo] = a.cards;
    const [bHi, bLo] = b.cards;
    return suitRank(aHi) - suitRank(bHi) || suitRank(aLo) - suitRank(bLo);
  });
}

/**
 * Which combos the solver treats alike: `bucket -> a number a reader can use`.
 *
 * `size` is the honest answer to "do the suits differ here", and usually it is
 * one: preflop the abstraction is 169 buckets for 169 classes, so all four
 * combos of `AKs` share one and drawing four identical bars would imply a
 * distinction the solver never made. Postflop they can split, and then WHICH
 * combos go together is the thing worth seeing -- on `A♥7♦2♣` the three suits
 * with a card on the board play alike and the spade does not.
 *
 * Numbered by first appearance rather than by bucket id, which means nothing to
 * a reader and is not stable between spots. Blocked combos are left out: they
 * are in no bucket, and counting them would split a group that agrees.
 */
export function bucketGroups(rows: readonly ComboRow[]): Map<number, number> {
  const groups = new Map<number, number>();
  for (const row of rows) {
    if (!row.blocked && !groups.has(row.bucket)) groups.set(row.bucket, groups.size + 1);
  }
  return groups;
}
