/**
 * Cards as a person sees them, from the spelling the wire uses.
 *
 * A card arrives as `Card.__repr__` writes it — `"Ah"`, `"Td"`, `"2c"` — and
 * that spelling is the ONLY one `parse_board` reads back, so it stays canonical
 * everywhere: in the URL, in the picker, in a bookmark. This module never
 * changes it; it only says what a rank and a suit should LOOK like.
 *
 * Four colours, not two
 * ---------------------
 * A two-colour deck is right at a physical table, where you hold two cards and
 * can see both suits at a glance. It is wrong on a screen showing five board
 * cards beside two hole cards, where "is the flush draw live" is a question the
 * reader should answer by colour instead of by reading seven glyphs. Every
 * solver UI is four-colour for that reason, and the chart beside this one
 * already encodes meaning in colour, so the eye is doing that work anyway.
 *
 * The server remains the authority on whether a board is legal. `readBoard` is
 * lenient in exactly the way `parse_board` is — spaces and commas optional —
 * so that pasting a board someone sent you works; anything it cannot read it
 * drops, and the request that follows gets the refusal in the server's words.
 */

/** High to low, matching the chart's row order so the two read the same way. */
export const RANKS = ["A", "K", "Q", "J", "T", "9", "8", "7", "6", "5", "4", "3", "2"] as const;

/** Spades first, then the two reds split apart so adjacent rows differ. */
const SUITS = ["s", "h", "d", "c"] as const;

/** How many cards each street ADDS, and what to call the slot on screen. */
export const BOARD_SLOTS = [
  { street: "flop", cards: 3 },
  { street: "turn", cards: 1 },
  { street: "river", cards: 1 },
] as const;

export const BOARD_SIZE = 5;

const GLYPHS: Record<string, string> = { s: "♠", h: "♥", d: "♦", c: "♣" };

/**
 * One colour per suit, from the console's own accents where they exist.
 *
 * Spades read as the foreground rather than as a fifth colour: it is the suit a
 * reader falls back to, and giving it a hue of its own would leave nothing on
 * the card looking neutral.
 */
const COLOURS: Record<string, string> = {
  s: "var(--fg)",
  h: "#E0655C",
  d: "#5B9BD5",
  c: "#56B184",
};

export interface ParsedCard {
  /** The wire spelling, unchanged — `"Ah"`. */
  text: string;
  rank: string;
  suit: string;
  glyph: string;
  colour: string;
}

const RANK_SET = new Set<string>(RANKS);
const SUIT_SET = new Set<string>(SUITS);

/** `"Ah"` into its parts, or null if it is not a card this deck contains. */
export function parseCard(text: string): ParsedCard | null {
  if (text.length !== 2) return null;
  // Ranks are upper case on the wire and suits are lower, but a person typing
  // into the paste field will not be careful about it and the server is not
  // either — `Card.new` takes both.
  const rank = text.slice(0, 1).toUpperCase();
  const suit = text.slice(1, 2).toLowerCase();
  if (!RANK_SET.has(rank) || !SUIT_SET.has(suit)) return null;
  return {
    text: `${rank}${suit}`,
    rank,
    suit,
    glyph: GLYPHS[suit] ?? "?",
    colour: COLOURS[suit] ?? "var(--fg)",
  };
}

/**
 * A board string into the cards it names, dropping anything unreadable.
 *
 * Deliberately lenient and deliberately NOT authoritative: this exists so the
 * picker can show what a pasted board means, and so a half-typed one does not
 * blank the slots. A board this accepts can still be refused by the server —
 * for repeating a card, or for being too short for the line — and that refusal
 * is the one the page shows.
 */
export function readBoard(board: string): ParsedCard[] {
  const text = board.replace(/[\s,]/g, "");
  const cards: ParsedCard[] = [];
  // `+ 1 <` rather than `<`: a trailing half-card is someone still typing, not
  // a card, and treating it as one makes the last slot flicker on every keypress.
  for (let index = 0; index + 1 < text.length; index += 2) {
    const card = parseCard(text.slice(index, index + 2));
    if (card) cards.push(card);
  }
  return cards;
}

/** The cards back into the one spelling the server reads. */
export function writeBoard(cards: readonly ParsedCard[]): string {
  return cards.map((card) => card.text).join("");
}

/** Every card in the deck, in picker order: four rows of thirteen. */
export function deck(): ParsedCard[] {
  const cards: ParsedCard[] = [];
  for (const suit of SUITS) {
    for (const rank of RANKS) {
      const card = parseCard(`${rank}${suit}`);
      if (card) cards.push(card);
    }
  }
  return cards;
}
