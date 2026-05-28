import { useBlueprintRun, useDealHand, useSubmitAction } from "@/api/queries";
import type { Hand, HandEvent } from "@/api/types";
import { BoxControl } from "@/components/BoxControl";
import { Panel } from "@/components/Panel";
import { describeAction } from "@/lib/actions";
import { cn } from "@/lib/utils";
import { useState } from "react";

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
 */
export function Play() {
  const [hand, setHand] = useState<Hand | null>(null);
  const [seat, setSeat] = useState(0);
  const run = useBlueprintRun();
  const deal = useDealHand();
  const act = useSubmitAction();

  const bigBlind = run.data?.big_blind ?? 0;
  const error = deal.error ?? act.error;
  const busy = deal.isPending || act.isPending;

  return (
    <div className="space-y-3 p-3">
      <BoxControl />
      <Panel
        title="table"
        error={
          run.error
            ? "No blueprint server is reachable — nothing to play against."
            : error
              ? String(error.message)
              : null
        }
      >
        <div className="space-y-3 px-3 py-3">
          <div className="flex flex-wrap items-center gap-3">
            <button
              type="button"
              disabled={busy || !run.data}
              onClick={() => deal.mutate({ human_seat: seat }, { onSuccess: setHand })}
              className="rounded border border-[var(--border)] px-3 py-1.5 font-mono text-[12px] text-[var(--fg)] hover:bg-white/[0.06] disabled:opacity-40"
            >
              {hand ? "next hand" : "deal"}
            </button>
            <label className="flex items-center gap-2 font-mono text-[11px] text-[var(--fg-muted)]">
              seat
              <select
                value={seat}
                onChange={(event) => setSeat(Number(event.target.value))}
                className="rounded border border-[var(--border)] bg-transparent px-2 py-1 text-[11px]"
              >
                <option value={0}>0</option>
                <option value={1}>1</option>
              </select>
            </label>
            {hand && (
              <span className="font-mono text-[11px] text-[var(--fg-faint)]">
                button seat {hand.button} · {hand.street}
              </span>
            )}
          </div>

          {hand ? <Table hand={hand} /> : null}

          {hand && !hand.over ? (
            <div className="flex flex-wrap items-center gap-2">
              {hand.legal.map((action) => (
                <button
                  key={action.token}
                  type="button"
                  disabled={busy}
                  onClick={() =>
                    act.mutate(
                      { session: hand.session, token: action.token },
                      { onSuccess: setHand },
                    )
                  }
                  className="rounded border border-[var(--border)] px-3 py-1.5 font-mono text-[12px] text-[var(--fg)] hover:bg-white/[0.06] disabled:opacity-40"
                >
                  {describeAction(action.token, bigBlind).text}
                </button>
              ))}
            </div>
          ) : null}

          {hand?.over ? <Result hand={hand} /> : null}
        </div>
      </Panel>

      {hand ? (
        <Panel title="what the bot did">
          <div className="space-y-2 px-3 py-3">
            <div className="font-mono text-[11px] text-[var(--fg-muted)]">
              {hand.bot_untrained_decisions} of {hand.bot_decisions} bot decisions were at infosets
              training never visited
              {hand.bot_untrained_decisions > 0 && " — those were uniform-random, not strategy"}
            </div>
            <ol className="space-y-1">
              {hand.log.map((event, index) => (
                // Position, not content: a line repeats the same action freely.
                <Event
                  key={`${index}-${event.actor}-${event.action}`}
                  event={event}
                  bigBlind={bigBlind}
                />
              ))}
            </ol>
          </div>
        </Panel>
      ) : null}
    </div>
  );
}

function Table({ hand }: { hand: Hand }) {
  return (
    <div className="space-y-2 rounded border border-[var(--border)] p-3">
      <div className="flex flex-wrap items-center gap-6 font-mono text-[11px] text-[var(--fg-muted)]">
        <span>
          pot <span className="tabular-nums text-[var(--fg)]">{hand.pot}</span>
        </span>
        <span>
          stacks <span className="tabular-nums text-[var(--fg)]">{hand.stacks.join(" / ")}</span>
        </span>
      </div>

      <Row label="board" cards={hand.board} empty="—" />
      <Row label="you" cards={hand.hole_cards} empty="—" />
      {/* Withheld by the server until the hand ends; shown as backs so the
          layout does not jump when they arrive. */}
      <Row label="bot" cards={hand.bot_hole_cards ?? ["??", "??"]} empty="—" />
    </div>
  );
}

function Row({ label, cards, empty }: { label: string; cards: string[]; empty: string }) {
  return (
    <div className="flex items-center gap-3">
      <span className="w-10 font-mono text-[11px] text-[var(--fg-faint)]">{label}</span>
      <div className="flex gap-1">
        {cards.length === 0 ? (
          <span className="font-mono text-[12px] text-[var(--fg-faint)]">{empty}</span>
        ) : (
          cards.map((card, index) => <Card key={`${index}-${card}`} card={card} />)
        )}
      </div>
    </div>
  );
}

const RED = new Set(["h", "d"]);

function Card({ card }: { card: string }) {
  const hidden = card === "??";
  const suit = card.slice(1, 2);
  return (
    <span
      className={cn(
        "inline-flex h-7 w-6 items-center justify-center rounded-[3px] border font-mono text-[12px]",
        hidden
          ? "border-[var(--border)] bg-white/[0.03] text-[var(--fg-faint)]"
          : "border-[var(--border)] bg-white/[0.06]",
        !hidden && RED.has(suit) ? "text-[#E0655C]" : !hidden ? "text-[var(--fg)]" : "",
      )}
    >
      {hidden ? "·" : card}
    </span>
  );
}

function Result({ hand }: { hand: Hand }) {
  const won = (hand.payoff ?? 0) > 0;
  const flat = (hand.payoff ?? 0) === 0;
  return (
    <div className="flex flex-wrap items-center gap-3 font-mono text-[12px]">
      <span
        className={cn(flat ? "text-[var(--fg-muted)]" : won ? "text-[#56B184]" : "text-[#E0655C]")}
      >
        {flat ? "chopped" : won ? "you win" : "you lose"}{" "}
        <span className="tabular-nums">
          {(hand.payoff ?? 0) > 0 ? "+" : ""}
          {hand.payoff}
        </span>
      </span>
      <span className="text-[var(--fg-faint)]">{hand.showdown ? "at showdown" : "on a fold"}</span>
    </div>
  );
}

function Event({ event, bigBlind }: { event: HandEvent; bigBlind: number }) {
  return (
    <li className="space-y-0.5 border-b border-[var(--border)] pb-1 last:border-0 font-mono text-[11px]">
      <div className="flex items-center gap-2">
        <span className="w-12 text-[var(--fg-faint)]">{event.street}</span>
        <span className={event.actor === "bot" ? "text-[var(--fg)]" : "text-[var(--fg-muted)]"}>
          {event.actor}
        </span>
        <span className="text-[var(--fg)]">{label(event.action, event.amount, bigBlind)}</span>
        {event.untrained && (
          <span className="rounded border border-[#D9A44E] px-1 text-[10px] text-[#D9A44E]">
            untrained
          </span>
        )}
      </div>
      {/* The mix arrives only once the hand is over — mid-hand it would be a
          look at the opponent's strategy. */}
      {event.mix && (
        <div className="flex flex-wrap gap-3 pl-14 text-[var(--fg-faint)]">
          {event.mix.map(([name, weight]) => (
            <span key={name}>
              {name} <span className="tabular-nums">{(weight * 100).toFixed(0)}%</span>
            </span>
          ))}
        </div>
      )}
    </li>
  );
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
