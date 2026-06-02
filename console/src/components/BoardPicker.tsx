import { PlayingCard } from "@/components/PlayingCard";
import { BOARD_SIZE, BOARD_SLOTS, RANKS, deck, readBoard, writeBoard } from "@/lib/cards";
import { cn } from "@/lib/utils";
import { useState } from "react";

/**
 * Choosing the runout, by clicking cards.
 *
 * This was a 36-pixel text field that took `"2c7d9h"`. That is the right thing
 * for the URL to carry and the wrong thing to ask a person for: you have to know
 * the spelling, you get no help with which cards are already out, and a typo
 * comes back as a 422 that greys the panel rather than as a card that will not
 * take. Postflop was effectively unreachable — which is most of the game.
 *
 * Three things make it work, and each is a fix for something specific:
 *
 *   SLOTS, grouped flop/turn/river, so the board has a shape before it has
 *   cards and "this line needs three" is answerable by looking. The grouping is
 *   naming, not game logic — the server decides what a line reaches.
 *
 *   `live`, so a board longer than the line uses is visibly longer. Replay
 *   consumes cards street by street and IGNORES the surplus, which is what lets
 *   you hold one runout fixed while walking back up the line. Left undrawn, that
 *   silently answers a turn question with a river card on screen.
 *
 * There is no text field. One was kept at first, on the reasoning that a board
 * someone sends you arrives as text — but the thing they send you is a LINK,
 * and `board` is a search param, so the URL already is the paste target. A
 * second way in bought nothing and put the spelling back on screen.
 *
 * Clicking a filled slot clears IT AND EVERYTHING AFTER, rather than leaving a
 * hole. A board is an ordered deal, so there is no such thing as a river with no
 * turn, and the alternative — shifting later cards forward — silently changes
 * which street a card was dealt on.
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
  const truncateTo = (count: number) => onChange(writeBoard(cards.slice(0, count)));

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
                        disabled={!card}
                        onClick={() => truncateTo(index)}
                        title={card ? `${card.text} — click to clear from here` : undefined}
                        className="rounded-[4px] enabled:hover:opacity-70 disabled:cursor-default"
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

        <div className="flex items-center gap-2 pb-4">
          <button
            type="button"
            onClick={() => setOpen((was) => !was)}
            className={cn(
              "rounded border px-2 py-1 text-[11px]",
              showDeck
                ? "border-[var(--fg-faint)] text-[var(--fg)]"
                : "border-[var(--border)] text-[var(--fg-muted)] hover:text-[var(--fg)]",
            )}
          >
            {showDeck ? "hide deck" : "pick cards"}
          </button>
          {cards.length > 0 && (
            <button
              type="button"
              onClick={() => onChange("")}
              className="text-[11px] text-[var(--fg-faint)] underline-offset-2 hover:text-[var(--fg)] hover:underline"
            >
              clear
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
          onPick={onChange}
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
