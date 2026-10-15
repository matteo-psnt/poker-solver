/**
 * Comparing one spot across stack depths.
 *
 * Action MENUS differ between depths -- 3 bb has no 5.5 bb raise -- so two
 * depths are compared by KIND (fold, call or check, raise or bet, all in), the
 * one vocabulary every depth shares.
 */

export const KINDS = ["fold", "call", "raise", "allin"] as const;
export type Kind = (typeof KINDS)[number];
export type KindMix = Record<Kind, number>;

export const KIND_LABEL: Record<Kind, string> = {
  fold: "fold",
  call: "call or check",
  raise: "raise or bet",
  allin: "all in",
};

/** The kind of a wire token: `f`, `c`/`x`, `r…`/`b…`, `A`. */
export function kindOf(token: string): Kind {
  const head = token[0];
  if (head === "f") return "fold";
  if (head === "c" || head === "x") return "call";
  if (head === "A") return "allin";
  return "raise";
}

export function kindMix(strategy: readonly number[] | null, tokens: readonly string[]): KindMix {
  const mix: KindMix = { fold: 0, call: 0, raise: 0, allin: 0 };
  if (!strategy) return mix;
  tokens.forEach((token, i) => {
    mix[kindOf(token)] += strategy[i] ?? 0;
  });
  return mix;
}

/** Share of the strategy that differs between two mixes: 0 identical, 1 disjoint. */
export function totalVariation(a: KindMix, b: KindMix): number {
  return KINDS.reduce((sum, k) => sum + Math.abs(a[k] - b[k]), 0) / 2;
}
