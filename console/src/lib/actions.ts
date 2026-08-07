/**
 * Turning wire tokens into something a poker player reads.
 *
 * `f`, `c`, `r4`, `A` is the path vocabulary — deliberately terse, because it
 * has to survive being a URL and a stored bookmark. That is the right shape for
 * an identifier and the wrong one for a button, so it is translated here rather
 * than shown raw.
 *
 * SIZES ARE IN BIG BLINDS, not chips. The wire carries the amount a bet raises
 * *to*, in chips, because that is unambiguous without knowing the stakes. But
 * "raise to 4" means nothing until you also know the blinds are 1/2 — and every
 * poker player already thinks in blinds, so `r4` at 1/2 reads as a 2bb open.
 * The conversion needs the big blind, which is why these take it as an argument
 * rather than being a lookup table.
 */

export interface ActionLabel {
  /** What the button says: "raise to 2bb". */
  text: string;
  /** The token, kept for the tooltip — it is what appears in the path. */
  token: string;
}

const WORDS: Record<string, string> = {
  f: "fold",
  x: "check",
  c: "call",
  A: "all-in",
};

/** `4` chips at a big blind of `2` → `"2"`; `9` → `"4.5"`. */
function inBlinds(chips: number, bigBlind: number): string {
  if (!bigBlind) return String(chips);
  const bb = chips / bigBlind;
  // Whole numbers read better without a trailing ".0", and half-blind sizes are
  // common enough that one decimal is worth keeping.
  return Number.isInteger(bb) ? String(bb) : bb.toFixed(1);
}

export function describeAction(token: string, bigBlind: number): ActionLabel {
  const kind = token.slice(0, 1);
  const word = WORDS[kind];
  if (word) return { text: word, token };

  const chips = Number(token.slice(1));
  if (!Number.isFinite(chips)) return { text: token, token };

  const size = inBlinds(chips, bigBlind);
  // "to", because the amount is the total the pot is raised TO, not the
  // increment — the distinction that makes a 3-bet read as 9 rather than 5.
  if (kind === "r") return { text: `raise to ${size}bb`, token };
  if (kind === "b") return { text: `bet ${size}bb`, token };
  return { text: token, token };
}

export function describeActions(tokens: string[], bigBlind: number): ActionLabel[] {
  return tokens.map((token) => describeAction(token, bigBlind));
}
