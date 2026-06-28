import { useState } from "react";
import { PlayingCard } from "@/components/PlayingCard";
import { BOARD_SIZE, BOARD_SLOTS, deck, RANKS, readBoard, writeBoard } from "@/lib/cards";
import { cn } from "@/lib/utils";

/**
 * Choosing the runout, by clicking cards.
 *
 * Slots are grouped flop/turn/river so the board has a shape before it has cards.
 * The grouping is naming, not game logic -- the server decides what a line reaches.
 *
 * The slots are also the way in: you point at the empty turn to say "deal a turn".
 * One gesture covers dealing and re-dealing, because truncating to a slot at or
 * past the end is a no-op and truncating to a filled one is what changing that
 * card means. Clicking a filled slot clears IT AND EVERYTHING AFTER, because a
 * board is an ordered deal and shifting later cards forward would silently change
 * which street a card was dealt on.
 *
 * `live` marks the cards the line does not reach: replay consumes cards street by
 * street and IGNORES the surplus, which is what lets you hold one runout fixed
 * while walking back up the line. Left undrawn, a turn question is answered with a
 * river card on screen.
 *
 * No text field. `board` is a search param, so the URL is already the paste target.
 */
export function BoardPicker({
  board,
  onChange,
  live,
  forceOpen = false,
}: {
  board: string;
  onChange: (board: string) => void;
  /** How many cards the current line actually reaches, if it is answerable. */
  live?: number | null;
  /** The line needs cards it has not got — the deck opens itself and stays. */
  forceOpen?: boolean;
}) {
  const [open, setOpen] = useState(false);
  const cards = readBoard(board);
  const taken = new Set(cards.map((card) => card.text));
  const showDeck = open || forceOpen;

  const slots = Array.from({ length: BOARD_SIZE }, (_, index) => cards[index] ?? null);
  // Silent when there is nothing past `count` to drop — clicking an empty slot
  // is opening the deck, not editing the board, and writing the same string
  // back would be a router navigation per click and a history entry per open.
  const truncateTo = (count: number) => {
    if (count >= cards.length) return;
    onChange(writeBoard(cards.slice(0, count)));
  };

  return (
    <div className="space-y-2">
      <div className="flex flex-wrap items-end gap-4">
        <div className="flex items-end gap-2">
          {BOARD_SLOTS.map((group, groupIndex) => {
            const from = BOARD_SLOTS.slice(0, groupIndex).reduce(
              (total, earlier) => total + earlier.cards,
              0,
            );
            return (
              <div key={group.street} className="space-y-1">
                <div className="flex gap-1">
                  {Array.from({ length: group.cards }, (_, offset) => {
                    const index = from + offset;
                    const card = slots[index] ?? null;
                    return (
                      <button
                        key={index}
                        type="button"
                        // The slot IS the control. Clicking one truncates the
                        // board to it and opens the deck, so "deal a flop" and
                        // "change the turn" are the same gesture on the thing
                        // you are changing — no button standing in for it.
                        // Truncating to an index at or past the end is a no-op,
                        // which is what makes the empty slots pure openers.
                        onClick={() => {
                          truncateTo(index);
                          setOpen(true);
                        }}
                        title={
                          card
                            ? `${card.text} — click to pick it again, clearing what follows`
                            : `pick the ${group.street}`
                        }
                        // `flex`, so the card is a flex item rather than inline
                        // content and no baseline is consulted at all — belt
                        // and braces with `PlayingCard`'s own alignment.
                        className="flex rounded-[4px] hover:opacity-70"
                      >
                        <PlayingCard
                          card={card?.text ?? null}
                          // The board is the control this bar is FOR, so it is
                          // the largest thing on it. At `sm` it read as
                          // punctuation between two buttons.
                          size="md"
                          dimmed={live != null && index >= live}
                        />
                      </button>
                    );
                  })}
                </div>
                <div className="text-center text-[10px] tracking-wider text-[var(--fg-faint)] uppercase">
                  {group.street}
                </div>
              </div>
            );
          })}
        </div>

        <div className="flex items-center gap-3 pb-4">
          {cards.length > 0 && (
            <button
              type="button"
              onClick={() => onChange("")}
              className="text-[11px] text-[var(--fg-faint)] underline-offset-2 hover:text-[var(--fg)] hover:underline"
            >
              clear
            </button>
          )}
          {/* A dismiss for an open panel, not a way in — the slots are that.
              Absent while the line still needs cards, where closing the deck
              would hide the only thing that answers the question on screen. */}
          {open && !forceOpen && (
            <button
              type="button"
              onClick={() => setOpen(false)}
              className="text-[11px] text-[var(--fg-faint)] underline-offset-2 hover:text-[var(--fg)] hover:underline"
            >
              done
            </button>
          )}
        </div>
      </div>

      {live != null && cards.length > live && (
        // Not a warning — it is the feature. Said once, where the dimming is.
        <p className="text-[11px] text-[var(--fg-faint)]">
          This line reaches {live} of them; the rest are kept for when you walk further down it.
        </p>
      )}

      {showDeck && (
        <Deck
          taken={taken}
          full={cards.length >= BOARD_SIZE}
          // Closes itself on the fifth card: there is nothing left to pick, and
          // a deck of 52 disabled cards is a worse way to say so than going.
          onPick={(next) => {
            onChange(next);
            if (readBoard(next).length >= BOARD_SIZE) setOpen(false);
          }}
          // The CARDS read back out, not the raw param. Appending to the raw
          // one carried a half-typed paste along with it: with `"AsK"` in the
          // field, clicking 7c wrote `"AsK7c"`, which the server refuses as not
          // a whole number of cards. Clicking a card also canonicalises the URL.
          board={writeBoard(cards)}
        />
      )}
    </div>
  );
}

/** The 52, in four rows of thirteen. Clicking one deals it to the next slot. */
function Deck({
  taken,
  full,
  board,
  onPick,
}: {
  taken: ReadonlySet<string>;
  full: boolean;
  board: string;
  onPick: (board: string) => void;
}) {
  return (
    <div
      // Capped, not stretched. `1fr` columns in a panel this wide made each
      // card a 130px slab of empty space, so the deck outweighed the board it
      // is for. A card should look like a card at any window width.
      className="grid w-full max-w-[36rem] gap-px rounded border border-[var(--border)] bg-[var(--border)] p-px"
      style={{ gridTemplateColumns: `repeat(${RANKS.length}, minmax(0, 1fr))` }}
    >
      {deck().map((card) => {
        // A card already on the board and a full board are both refusals the
        // server would make — `parse_board` rejects a repeat — so they are made
        // unclickable here rather than sent and bounced.
        const used = taken.has(card.text);
        return (
          <button
            key={card.text}
            type="button"
            disabled={used || full}
            onClick={() => onPick(`${board}${card.text}`)}
            title={used ? `${card.text} is already on the board` : card.text}
            className={cn(
              "flex items-center justify-center gap-px bg-[#101014] py-1 font-mono text-[11px] leading-none",
              used || full
                ? "opacity-25"
                : "cursor-pointer hover:bg-[#1d1d24] focus-visible:outline focus-visible:outline-1 focus-visible:outline-[var(--fg)]",
            )}
            style={{ color: card.colour }}
          >
            <span>{card.rank}</span>
            <span className="text-[10px]">{card.glyph}</span>
          </button>
        );
      })}
    </div>
  );
}
