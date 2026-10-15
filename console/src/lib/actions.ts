/**
 * Turning wire tokens into something a poker player reads.
 *
 * `f`, `c`, `r4`, `A` is the path vocabulary — deliberately terse, because it
 * has to survive being a URL and a stored bookmark. That is the right shape for
 * an identifier and the wrong one for a button, so it is translated here rather
 * than shown raw.
 *
 * SIZES ARE IN BIG BLINDS, and they are the server's `to`: the chips the
 * actor has in on the street after acting. The token's number is NOT that --
 * on a raise it is only the chips above the call -- so it is never read here.
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
export function inBlinds(chips: number, bigBlind: number): string {
  if (!bigBlind) return String(chips);
  const bb = chips / bigBlind;
  // Whole numbers read better without a trailing ".0", and half-blind sizes are
  // common enough that one decimal is worth keeping.
  return Number.isInteger(bb) ? String(bb) : bb.toFixed(1);
}

/** An action as the server sends it: the path token and its raise-to. */
export interface Sized {
  token: string;
  to: number;
}

export function describeAction({ token, to }: Sized, bigBlind: number): ActionLabel {
  const kind = token.slice(0, 1);
  const word = WORDS[kind];
  if (word) return { text: word, token };
  if (!Number.isFinite(to)) return { text: token, token };

  const size = inBlinds(to, bigBlind);
  if (kind === "r") return { text: `raise to ${size}bb`, token };
  if (kind === "b") return { text: `bet ${size}bb`, token };
  return { text: token, token };
}

/** Label `tokens` in their order, sized from the `edges` the node offered. */
export function describeActions(tokens: string[], edges: Sized[], bigBlind: number): ActionLabel[] {
  const to = new Map(edges.map((edge) => [edge.token, edge.to]));
  return tokens.map((token) =>
    describeAction({ token, to: to.get(token) ?? Number.NaN }, bigBlind),
  );
}
