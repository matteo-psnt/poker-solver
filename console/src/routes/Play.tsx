import { useBlueprintRun, useDealHand, useLeaveHand, useSubmitAction } from "@/api/queries";
import type { Hand, HandEvent } from "@/api/types";
import { CardRow } from "@/components/PlayingCard";
import { describeAction } from "@/lib/actions";
import { BOARD_SIZE } from "@/lib/cards";
import { actionColours } from "@/lib/range";
import { cn } from "@/lib/utils";
import { useCallback, useEffect, useRef, useState } from "react";

/**
 * Sit down against the blueprint, one hand at a time.
 *
 * What this page is FOR is finding spots where the bot does something strange —
 * and the fastest way there is a human, who explores far off the distribution
 * self-play ever reached. Which is also why the untrained count is on screen
 * throughout rather than hidden in a detail view: off that distribution the bot
 * plays uniform-random because it has no strategy, and a player who cannot see
 * that will read it as a bad blueprint instead of an unvisited node.
 *
 * There is deliberately NO running win-rate. Head-to-head needs ~46k deals to
 * resolve ~100 mbb and the blueprint is under-trained, so a counter here would
 * invite exactly the conclusion the numbers cannot support. Per-hand chips are
 * shown because they are a fact about the hand; a total would be a claim about
 * the solver.
 *
 * Why it is a table now
 * ---------------------
 * It was three labelled rows of two-character cards — `board  As Kd 7c` — with
 * the pot and the stacks as one line of text above them. Everything on it was
 * legible and nothing on it was READABLE: whose turn it was, which seat had the
 * button, and whether the bot had acted since you last looked were all facts
 * you assembled by reading, on a page whose whole value is how many hands you
 * get through. Seats face each other, the board is between them, the cards are
 * cards, and the seat to act is the one that is lit.
 *
 * The keys matter for the same reason. Finding a strange spot is a numbers
 * game, and a hand costs two clicks and a mouse trip to a button whose position
 * moves with the size of the menu. Every action is a digit in menu order, the
 * common ones are also their own initial, and space deals the next hand.
 */
export function Play() {
  const [hand, setHand] = useState<Hand | null>(null);
  const [seat, setSeat] = useState(0);
  const run = useBlueprintRun();
  const deal = useDealHand();
  const act = useSubmitAction();
  const leave = useLeaveHand();

  const bigBlind = run.data?.big_blind ?? 0;
  const error = deal.error ?? act.error;
  const busy = deal.isPending || act.isPending;
  const canAct = Boolean(hand && !hand.over) && !busy;

  /**
   * Hand the session back when you walk away from it.
   *
   * `Blueprint.tsx` mounts this tab and the chart one at a time so that leaving
   * "ends it where it lives" — which was not true of anything, because the
   * endpoint existed and nothing called it. Sessions live on the blueprint
   * server in a store of 64 that drops the OLDEST to make room, so an abandoned
   * hand is not free: it evicts a hand someone is still playing.
   */
  const sessionRef = useRef<string | null>(null);
  sessionRef.current = hand?.session ?? null;
  const release = leave.mutate;
  useEffect(
    () => () => {
      const session = sessionRef.current;
      if (session) release({ session });
    },
    [release],
  );

  const onDeal = useCallback(() => {
    if (busy || !run.data) return;
    const previous = sessionRef.current;
    deal.mutate(
      { human_seat: seat },
      {
        onSuccess: (next) => {
          setHand(next);
          // Abandoning the old one explicitly rather than leaving it to be
          // evicted: the store is shared with anyone else at a keyboard.
          if (previous && previous !== next.session) release({ session: previous });
        },
      },
    );
  }, [busy, deal.mutate, release, run.data, seat]);

  const onAct = useCallback(
    (token: string) => {
      if (!hand || hand.over || busy) return;
      act.mutate({ session: hand.session, token }, { onSuccess: setHand });
    },
    [act.mutate, busy, hand],
  );

  useKeyboard(hand, canAct, onAct, onDeal);

  return (
    <div className="space-y-3">
      <div className="rounded-md border border-[var(--border)] bg-[var(--panel)]">
        <DealBar
          hand={hand}
          seat={seat}
          onSeat={setSeat}
          onDeal={onDeal}
          busy={busy}
          ready={Boolean(run.data)}
        />

        {run.error || error ? (
          <p className="border-b border-red-500/25 bg-red-500/5 px-3 py-2 font-mono text-[12px] text-red-400">
            {run.error
              ? "No blueprint server is reachable — nothing to play against."
              : String(error?.message)}
          </p>
        ) : null}

        {hand ? (
          <Table hand={hand} bigBlind={bigBlind} />
        ) : (
          <p className="px-3 py-16 text-center text-[13px] text-[var(--fg-faint)]">
            Deal a hand to sit down. Space deals; every action is a digit.
          </p>
        )}

        {hand && !hand.over ? (
          <Actions hand={hand} bigBlind={bigBlind} busy={busy} onAct={onAct} />
        ) : null}
        {hand?.over ? <Result hand={hand} bigBlind={bigBlind} onDeal={onDeal} busy={busy} /> : null}
      </div>

      {hand ? <History hand={hand} bigBlind={bigBlind} /> : null}
    </div>
  );
}

/** Sit down, and choose which seat to sit in. */
function DealBar({
  hand,
  seat,
  onSeat,
  onDeal,
  busy,
  ready,
}: {
  hand: Hand | null;
  seat: number;
  onSeat: (seat: number) => void;
  onDeal: () => void;
  busy: boolean;
  ready: boolean;
}) {
  return (
    <div className="flex flex-wrap items-center gap-3 border-b border-[var(--border)] px-3 py-2.5">
      <button
        type="button"
        disabled={busy || !ready}
        onClick={onDeal}
        className="flex items-center gap-2 rounded border border-[var(--fg-faint)] px-3 py-1.5 text-[12px] text-[var(--fg)] hover:bg-white/[0.06] disabled:opacity-40"
      >
        {busy ? "dealing…" : hand ? "next hand" : "deal"}
        <Key label="space" />
      </button>

      <label className="flex items-center gap-2 text-[11px] text-[var(--fg-muted)]">
        you sit in seat
        <select
          value={seat}
          onChange={(event) => onSeat(Number(event.target.value))}
          className="rounded border border-[var(--border)] bg-transparent px-2 py-1 text-[11px]"
        >
          <option value={0}>0</option>
          <option value={1}>1</option>
        </select>
      </label>

      {/* The button alternates on its own — the server does that so nobody sees
          one side of every spot and reads it as the whole strategy. Saying so
          here stops the alternation looking like a control that slipped. */}
      <span className="text-[11px] text-[var(--fg-faint)]">the button alternates each hand</span>
    </div>
  );
}

/**
 * The table: two seats facing each other, the board between them.
 *
 * The seat to act is lit, and that is the one piece of state the old page made
 * you work out — `to_act` was not on screen at all, so "is it me" was answered
 * by noticing whether there were buttons underneath.
 */
function Table({ hand, bigBlind }: { hand: Hand; bigBlind: number }) {
  const villain = 1 - hand.human_seat;
  const board = Array.from({ length: BOARD_SIZE }, (_, index) => hand.board[index] ?? null);

  return (
    <div
      className="border-b border-[var(--border)] px-3 py-6"
      style={{
        // A felt, kept nearly as dark as the panel: the console is a dark
        // instrument and a green table would be the loudest thing in it. Enough
        // to say "this region is the table", not enough to be decoration.
        backgroundImage:
          "radial-gradient(60% 70% at 50% 50%, rgba(45,105,80,.20) 0%, rgba(20,32,27,.12) 55%, transparent 80%)",
      }}
    >
      {/* Narrow and centred, NOT the panel's full width. Stretched, the two
          seats were 1700px bars with the name at one end and the cards at the
          other — two toolbars stacked, when the whole point is that they face
          each other across a board. A table has a size; a row does not. */}
      <div className="mx-auto max-w-[42rem] space-y-4">
        <Seat
          name="bot"
          seat={villain}
          button={hand.button}
          stack={hand.stacks[villain] ?? 0}
          bigBlind={bigBlind}
          cards={hand.bot_hole_cards}
          active={hand.to_act === villain}
          did={lastAction(hand, "bot")}
          street={hand.street}
        />

        <div className="flex flex-col items-center gap-2">
          {/* The street, named. The board implies it, but only if you count
              cards — and preflop it implies nothing at all, because there is
              no board to count. */}
          <span className="font-mono text-[10px] tracking-[0.2em] text-[var(--fg-faint)] uppercase">
            {hand.street}
          </span>
          <CardRow cards={board} size="md" live={hand.board.length} />
          <div className="flex items-baseline gap-2 font-mono text-[12px]">
            <span className="tracking-widest text-[var(--fg-faint)] uppercase">pot</span>
            <span className="text-[15px] tabular-nums text-[var(--fg)]">{hand.pot}</span>
            <span className="text-[var(--fg-muted)]">{inBlinds(hand.pot, bigBlind)}</span>
          </div>
        </div>

        <Seat
          name="you"
          seat={hand.human_seat}
          button={hand.button}
          stack={hand.stacks[hand.human_seat] ?? 0}
          bigBlind={bigBlind}
          cards={hand.hole_cards}
          active={hand.to_act === hand.human_seat}
          did={lastAction(hand, "human")}
          street={hand.street}
        />
      </div>
    </div>
  );
}

/**
 * The last thing a player did, whatever street they did it on.
 *
 * The bot's move is the thing you most need and the thing the table was worst
 * at: `POST /action` auto-plays it and returns, so the only record of it was a
 * line in the log two panels down. You could see the pot had grown and not what
 * had grown it.
 *
 * This was scoped to the CURRENT street at first, on the reasoning that a move
 * from a closed street is history and the board gaining a card announces it.
 * That reasoning fails in the most common case there is: you raise, the bot
 * calls, and the flop comes — so the street has advanced, nobody has acted on
 * it, and the answer to "what did the bot do" is blank at exactly the moment
 * you are asking. The street is drawn on the chip when it is not the current
 * one, which is what makes showing it honest rather than merely reassuring.
 */
function lastAction(hand: Hand, actor: "bot" | "human"): HandEvent | null {
  for (let index = hand.log.length - 1; index >= 0; index--) {
    const event = hand.log[index];
    if (event?.actor === actor) return event;
  }
  return null;
}

/**
 * One player: who they are, what they hold, what they just did, and what is
 * left behind it.
 *
 * `cards` is null while the server is withholding the bot's, which is not the
 * same as holding none — see `hand_payload`. Backs are drawn for it, so the
 * layout does not jump when they arrive at showdown either.
 */
function Seat({
  name,
  seat,
  button,
  stack,
  bigBlind,
  cards,
  active,
  did,
  street,
}: {
  name: string;
  seat: number;
  button: number;
  stack: number;
  bigBlind: number;
  cards: string[] | null;
  active: boolean;
  /** The last thing they did, drawn in front of them. */
  did: HandEvent | null;
  /** The street the hand is on now — see `Did`. */
  street: string;
}) {
  return (
    <div
      className={cn(
        "flex flex-wrap items-center gap-4 rounded-md border px-3 py-2.5 transition-colors",
        active ? "border-[var(--fg-faint)] bg-white/[0.04]" : "border-[var(--border)] bg-black/20",
      )}
    >
      <div className="flex min-w-0 items-center gap-2">
        <span
          className={cn(
            "size-1.5 shrink-0 rounded-full",
            active ? "bg-[#56B184] motion-safe:animate-pulse" : "bg-transparent",
          )}
        />
        <span className="font-mono text-[13px] tracking-wider text-[var(--fg)] uppercase">
          {name}
        </span>
        {/* Heads-up, so the button is the small blind and the other seat is the
            big one. Two labels rather than a seat number, because the spot is
            what a player is reading and `seat 1` is not one. */}
        <span className="rounded border border-[var(--border)] px-1.5 py-0.5 font-mono text-[10px] text-[var(--fg-muted)]">
          {seat === button ? "BTN" : "BB"}
        </span>
        {/* Beside the name and the lamp, not after the cards: it is a fact
            about WHO, and it read as a caption on the wrong thing out there. */}
        {active && <span className="text-[11px] text-[var(--fg-muted)]">to act</span>}
      </div>

      <div className="flex items-baseline gap-2 font-mono text-[12px]">
        <span className="tracking-widest text-[var(--fg-faint)] uppercase">stack</span>
        <span className="tabular-nums text-[var(--fg)]">{stack}</span>
        <span className="text-[var(--fg-muted)]">{inBlinds(stack, bigBlind)}</span>
      </div>

      {/* In FRONT of the player, between their stack and their cards, which is
          where the chips would be. Coloured by the same rule as the buttons and
          the chart, so a bet is the same red in all three. */}
      {did && <Did event={did} street={street} bigBlind={bigBlind} />}

      <div className="ml-auto">
        <CardRow cards={cards ?? [null, null]} size="lg" faceDown={cards === null} />
      </div>
    </div>
  );
}

/**
 * What a player just did, as a chip in front of them.
 *
 * Loud on purpose. The complaint it answers is that you could not tell the bot
 * had bet without reading the log — the pot number moved and nothing else did,
 * so a page whose value is how many hands you get through made you stop and
 * read on every one of them.
 */
function Did({
  event,
  street,
  bigBlind,
}: {
  event: HandEvent;
  /** The street the hand is on NOW, to tell a fresh move from a carried one. */
  street: string;
  bigBlind: number;
}) {
  const colour = actionColours([TOKEN_FOR[event.action] ?? event.action])[0];
  const stale = event.street !== street;
  return (
    <span
      // The one test hook on this page. The chip says the same words as the
      // action button above it and the log below it — by design, they must —
      // so a test cannot tell the three apart by their text.
      data-testid="did"
      className={cn(
        "rounded border px-2 py-1 font-mono text-[12px] text-[var(--fg)]",
        stale && "opacity-55",
      )}
      style={{
        borderColor: colour,
        backgroundColor: `color-mix(in srgb, ${colour} ${stale ? 12 : 26}%, transparent)`,
      }}
    >
      {label(event.action, event.amount, bigBlind)}
      {stale && (
        // Named, and dimmed, so a move that is still the last one taken cannot
        // be misread as a move taken on the street now showing.
        <span className="ml-2 text-[10px] text-[var(--fg-faint)] lowercase">{event.street}</span>
      )}
      {event.untrained && (
        // The caveat has to travel with the move, not sit in a total below:
        // an untrained decision is uniform-random, so this chip is not a read.
        <span className="ml-2 text-[10px] text-[#D9A44E]">untrained</span>
      )}
    </span>
  );
}

/**
 * The menu, in the server's order, each action on a digit.
 *
 * Coloured by the same rule the chart uses — fold blue, call green, raise red
 * shading with size — so the button you press here and the square you read
 * there are the same colour for the same action. That is the whole reason
 * `actionColours` is shared rather than re-picked per page.
 */
function Actions({
  hand,
  bigBlind,
  busy,
  onAct,
}: {
  hand: Hand;
  bigBlind: number;
  busy: boolean;
  onAct: (token: string) => void;
}) {
  const colours = actionColours(hand.legal.map((action) => action.token));
  return (
    <div className="flex flex-wrap items-center gap-2 border-b border-[var(--border)] px-3 py-3">
      {hand.legal.map((action, index) => (
        <button
          key={action.token}
          type="button"
          disabled={busy}
          onClick={() => onAct(action.token)}
          title={`token: ${action.token}`}
          className="flex items-center gap-2 rounded border px-3 py-2 text-[13px] text-[var(--fg)] hover:brightness-125 disabled:opacity-40"
          style={{
            borderColor: colours[index],
            // A tint rather than a fill: five saturated buttons in a row is a
            // toolbar, and the one thing that must stay readable is the size.
            backgroundColor: `color-mix(in srgb, ${colours[index]} 22%, transparent)`,
          }}
        >
          {describeAction(action.token, bigBlind).text}
          <Key label={String(index + 1)} />
        </button>
      ))}
      {busy && <span className="text-[11px] text-[var(--fg-faint)]">waiting on the bot…</span>}
    </div>
  );
}

/** How the hand ended, and what it cost. Per-hand only — never a total. */
function Result({
  hand,
  bigBlind,
  onDeal,
  busy,
}: {
  hand: Hand;
  bigBlind: number;
  onDeal: () => void;
  busy: boolean;
}) {
  const payoff = hand.payoff ?? 0;
  const flat = payoff === 0;
  return (
    <div className="flex flex-wrap items-center gap-4 border-b border-[var(--border)] px-3 py-3">
      <span
        className={cn(
          "font-mono text-[15px]",
          flat ? "text-[var(--fg-muted)]" : payoff > 0 ? "text-[#56B184]" : "text-[#E0655C]",
        )}
      >
        {flat ? "chopped" : payoff > 0 ? "you win" : "you lose"}{" "}
        <span className="tabular-nums">
          {payoff > 0 ? "+" : ""}
          {payoff}
        </span>{" "}
        <span className="text-[12px] opacity-70">{inBlinds(payoff, bigBlind)}</span>
      </span>
      <span className="text-[12px] text-[var(--fg-faint)]">
        {hand.showdown ? "at showdown" : "on a fold"}
      </span>
      <button
        type="button"
        disabled={busy}
        onClick={onDeal}
        className="ml-auto flex items-center gap-2 rounded border border-[var(--fg-faint)] px-3 py-1.5 text-[12px] text-[var(--fg)] hover:bg-white/[0.06] disabled:opacity-40"
      >
        next hand
        <Key label="space" />
      </button>
    </div>
  );
}

/**
 * What the bot did, street by street, and how untrained it was while doing it.
 *
 * The untrained count leads because it is the caveat on everything below it: a
 * decision at an infoset training never visited is uniform-random, and a mix
 * that reads as a strategy when it is a coin flip is the one way this page can
 * actively mislead.
 */
function History({ hand, bigBlind }: { hand: Hand; bigBlind: number }) {
  const streets = groupByStreet(hand.log);
  return (
    <div className="rounded-md border border-[var(--border)] bg-[var(--panel)]">
      <header className="flex flex-wrap items-center gap-3 border-b border-[var(--border)] px-3 py-2">
        <h2 className="font-mono text-[11px] tracking-widest text-[var(--fg-muted)] uppercase">
          hand history
        </h2>
        <span
          className={cn(
            "font-mono text-[11px]",
            hand.bot_untrained_decisions > 0 ? "text-[#D9A44E]" : "text-[var(--fg-faint)]",
          )}
        >
          {hand.bot_untrained_decisions} of {hand.bot_decisions} bot decisions were at infosets
          training never visited
          {hand.bot_untrained_decisions > 0 && " — those were uniform-random, not strategy"}
        </span>
      </header>

      <div className="divide-y divide-[var(--border)]">
        {streets.map(({ street, events }) => (
          <div key={street} className="flex gap-3 px-3 py-2">
            <span className="w-14 shrink-0 pt-0.5 font-mono text-[10px] tracking-wider text-[var(--fg-faint)] uppercase">
              {street}
            </span>
            <ol className="min-w-0 flex-1 space-y-1.5">
              {events.map(({ event, index }) => (
                <Event key={index} event={event} bigBlind={bigBlind} />
              ))}
            </ol>
          </div>
        ))}
        {hand.log.length === 0 && (
          <p className="px-3 py-3 text-[12px] text-[var(--fg-faint)]">Nothing has happened yet.</p>
        )}
      </div>
    </div>
  );
}

function Event({ event, bigBlind }: { event: HandEvent; bigBlind: number }) {
  return (
    <li className="space-y-1 font-mono text-[11px]">
      <div className="flex flex-wrap items-center gap-2">
        <span
          className={cn(
            "w-8 shrink-0",
            event.actor === "bot" ? "text-[var(--fg)]" : "text-[var(--fg-muted)]",
          )}
        >
          {event.actor === "bot" ? "bot" : "you"}
        </span>
        <span className="text-[var(--fg)]">{label(event.action, event.amount, bigBlind)}</span>
        {event.untrained && (
          <span className="rounded border border-[#D9A44E] px-1 text-[10px] text-[#D9A44E]">
            untrained
          </span>
        )}
      </div>
      {/* The mix arrives only once the hand is over — mid-hand it would be a
          look at the opponent's strategy. It names action TYPES, so a menu with
          two raise sizes shows "raise" twice, in menu order: the sizes are not
          on the wire and are not invented here. */}
      {event.mix && <Mix mix={event.mix} />}
    </li>
  );
}

/** The distribution the bot sampled from, as a bar and as numbers. */
function Mix({ mix }: { mix: [string, number][] }) {
  const colours = actionColours(mix.map(([name]) => TOKEN_FOR[name] ?? name));
  return (
    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 pl-10 text-[10px]">
      <span className="flex h-1.5 w-24 overflow-hidden rounded-[2px] bg-[var(--border)]">
        {mix.map(([name, weight], index) => (
          <span
            key={`${index}-${name}`}
            style={{ width: `${weight * 100}%`, backgroundColor: colours[index] }}
          />
        ))}
      </span>
      {mix.map(([name, weight], index) => (
        <span key={`${index}-${name}`} className="text-[var(--fg-faint)]">
          {name} <span className="tabular-nums text-[var(--fg-muted)]">{pct(weight)}</span>
        </span>
      ))}
    </div>
  );
}

/**
 * Action TYPE names back to the token vocabulary `actionColours` keys on.
 *
 * The log speaks types and the chart speaks tokens, and the colours have to
 * agree across them or the same action is red in one panel and green in the
 * other. Sizes do not survive the trip — a type name carries no amount — so
 * every raise shades identically here, which is honest: nothing on the wire
 * says which of them this was.
 */
const TOKEN_FOR: Record<string, string> = {
  fold: "f",
  check: "x",
  call: "c",
  bet: "b",
  raise: "r",
  "all-in": "A",
};

/** Consecutive events sharing a street, keeping each event's original index. */
function groupByStreet(
  log: HandEvent[],
): { street: string; events: { event: HandEvent; index: number }[] }[] {
  const groups: { street: string; events: { event: HandEvent; index: number }[] }[] = [];
  log.forEach((event, index) => {
    const last = groups.at(-1);
    if (last && last.street === event.street) last.events.push({ event, index });
    else groups.push({ street: event.street, events: [{ event, index }] });
  });
  return groups;
}

/**
 * Every action on a digit, the common ones also on their initial.
 *
 * Ignored while a text field has focus, so typing into one is typing and not a
 * fold. Bound on the window rather than a container because the table is not
 * focusable and requiring a click into it first is the exact friction this is
 * here to remove.
 */
function useKeyboard(
  hand: Hand | null,
  canAct: boolean,
  onAct: (token: string) => void,
  onDeal: () => void,
) {
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey) return;
      const target = event.target as HTMLElement | null;
      if (target && /^(INPUT|SELECT|TEXTAREA)$/.test(target.tagName)) return;

      if (event.key === " ") {
        event.preventDefault();
        if (!hand || hand.over) onDeal();
        return;
      }
      if (!canAct || !hand) return;

      const digit = Number(event.key);
      const byDigit = Number.isInteger(digit) && digit > 0 ? hand.legal[digit - 1] : undefined;
      const byLetter = hand.legal.find(
        (action) => INITIALS[event.key.toLowerCase()] === action.type,
      );
      const chosen = byDigit ?? byLetter;
      if (chosen) {
        event.preventDefault();
        onAct(chosen.token);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [canAct, hand, onAct, onDeal]);
}

/** `k` for check rather than `x`: the button says "check", and the token is a URL thing. */
const INITIALS: Record<string, string> = {
  f: "fold",
  k: "check",
  c: "call",
  a: "all-in",
};

function Key({ label }: { label: string }) {
  return (
    <kbd className="rounded border border-[var(--border)] bg-black/40 px-1 font-mono text-[9px] text-[var(--fg-faint)]">
      {label}
    </kbd>
  );
}

function pct(weight: number): string {
  return `${(weight * 100).toFixed(0)}%`;
}

/** Chips as blinds, for the second line under every chip figure. */
function inBlinds(chips: number, bigBlind: number): string {
  if (!bigBlind) return "";
  const bb = chips / bigBlind;
  return `${Number.isInteger(bb) ? bb : bb.toFixed(1)}bb`;
}

/**
 * The hand log speaks TYPES ("raise") and chip amounts, not path tokens, so it
 * cannot use `describeAction` directly — but it must read the same way, or the
 * same bet is "raise to 2bb" on a button and "raise 4" one panel below.
 */
function label(type: string, amount: number, bigBlind: number): string {
  if (type === "fold" || type === "check") return type;
  if (type === "call") return "call";
  if (type === "all-in" || type === "all_in") return "all-in";
  const size =
    bigBlind && Number.isInteger(amount / bigBlind)
      ? String(amount / bigBlind)
      : (amount / (bigBlind || 1)).toFixed(1);
  if (type === "raise") return `raise to ${size}bb`;
  if (type === "bet") return `bet ${size}bb`;
  return `${type} ${size}bb`;
}
