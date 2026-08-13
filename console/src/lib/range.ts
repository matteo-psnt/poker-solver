/**
 * Turning 1326 combos into the 13x13 grid a poker player reads.
 *
 * The solver's unit is a bucket, the wire's unit is a combo, and the unit a
 * person recognises is a hand CLASS — "AKs", "72o", "TT". This module is the
 * only place those three meet.
 *
 * Aggregation is deliberate, not cosmetic
 * ---------------------------------------
 * A class holds up to 12 combos (4 suited, 12 offsuit, 6 paired), and the board
 * may block some of them. A cell therefore averages over the combos that are
 * still POSSIBLE, which is why a blocked-out class is drawn as absent rather
 * than as a zero row: "you cannot hold this" and "you fold this" are different
 * facts and must not render the same.
 *
 * Averaging is unweighted across surviving combos. That is right here because
 * every combo of a class is equally likely a priori; it would be wrong the
 * moment a range weight enters, and that is the line where this stops being a
 * display concern.
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
