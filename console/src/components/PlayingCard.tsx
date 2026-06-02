import { type ParsedCard, parseCard } from "@/lib/cards";
import { cn } from "@/lib/utils";

/**
 * One card, at the three sizes the console needs.
 *
 * A card was two mono characters in a bordered box — `Ah` — which is fine in a
 * terminal and poor on a screen: the suit was the same colour as the rank, so
 * reading a board meant reading it letter by letter, and the hole cards looked
 * like the board looked like the picker. Here rank and suit are separate marks
 * and the suit carries the colour, which is what makes a board legible in one
 * glance rather than five.
 *
 * `sm` is the picker and the log, `md` the board, `lg` the cards in front of a
 * player. The size ladder is deliberate: on a table the eye should land on the
 * hole cards first and the board second, and nothing else on the page competes.
 */
export function PlayingCard({
  card,
  size = "md",
  faceDown = false,
  dimmed = false,
  className,
}: {
  /** The wire spelling, or null for an empty slot. */
  card: string | null;
  size?: "sm" | "md" | "lg";
  faceDown?: boolean;
  /** Present but not in play — a board card the current line never reaches. */
  dimmed?: boolean;
  className?: string;
}) {
  const parsed = card ? parseCard(card) : null;
  const box = SIZES[size];

  if (faceDown) return <FaceDown className={cn(box.frame, ALIGN, className)} />;
  if (!parsed) return <Empty className={cn(box.frame, ALIGN, className)} />;

  return (
    <span
      className={cn(
        "inline-flex flex-col items-center justify-center border border-[var(--border)] bg-[#17171c] leading-none",
        ALIGN,
        box.frame,
        dimmed && "opacity-30",
        className,
      )}
      style={{ color: parsed.colour }}
      title={parsed.text}
    >
      <span className={cn("font-mono font-medium", box.rank)}>{parsed.rank}</span>
      <span className={box.suit}>{parsed.glyph}</span>
    </span>
  );
}

/** The same frame with nothing in it, so a row of slots does not reflow. */
function Empty({ className }: { className?: string }) {
  return (
    <span
      className={cn(
        "inline-flex items-center justify-center border border-dashed border-[var(--border)] text-[var(--fg-faint)]",
        className,
      )}
    />
  );
}

/**
 * A card the server is withholding.
 *
 * Drawn as a back rather than as an empty slot, because those mean different
 * things: the bot HAS two cards and you are not being shown them until the hand
 * is over. An empty frame would read as a seat with no cards in it.
 */
function FaceDown({ className }: { className?: string }) {
  return (
    <span
      className={cn("inline-block border border-[var(--border)]", className)}
      style={{
        backgroundImage:
          "repeating-linear-gradient(45deg, rgba(255,255,255,.07) 0 2px, transparent 2px 5px)",
        backgroundColor: "#141419",
      }}
      title="face down until the hand is over"
    />
  );
}

/**
 * Why every state carries the same `vertical-align`.
 *
 * These are `inline-flex`, so in an inline context they sit on the text
 * baseline — and an EMPTY one has no in-flow line box, so CSS takes its
 * baseline from its bottom margin edge, while a filled one takes it from the
 * rank inside it. Identical frames, two different baselines: picking a card
 * made it drop a few pixels below the placeholder it replaced, on the one
 * screen where you are staring straight at the slot you just clicked.
 *
 * `middle` rather than a baseline of any kind, because it is computed from the
 * box, not from its contents — so an empty slot and a full one cannot disagree.
 * A no-op wherever the parent is already a flex container (`CardRow`), which is
 * why this never showed there.
 */
const ALIGN = "align-middle";

const SIZES = {
  sm: { frame: "h-7 w-5 rounded-[3px] gap-px", rank: "text-[11px]", suit: "text-[9px]" },
  md: { frame: "h-12 w-9 rounded-[4px] gap-0.5", rank: "text-[17px]", suit: "text-[13px]" },
  lg: { frame: "h-16 w-12 rounded-[5px] gap-1", rank: "text-[23px]", suit: "text-[17px]" },
} as const;

/** A board or a holding, laid out. Kept here so every row of cards matches. */
export function CardRow({
  cards,
  size = "md",
  faceDown = false,
  /** How many of `cards` the current line actually reaches; the rest dim. */
  live,
  className,
}: {
  cards: readonly (string | ParsedCard | null)[];
  size?: "sm" | "md" | "lg";
  faceDown?: boolean;
  live?: number;
  className?: string;
}) {
  return (
    <div className={cn("flex items-center gap-1", className)}>
      {cards.map((card, index) => {
        const text = typeof card === "string" || card === null ? card : card.text;
        return (
          <PlayingCard
            // Position, not value: two slots can hold the same placeholder null,
            // and a board legitimately repeats no card but a picker mid-edit can.
            key={`${index}-${text ?? "empty"}`}
            card={text}
            size={size}
            faceDown={faceDown}
            dimmed={live !== undefined && index >= live}
          />
        );
      })}
    </div>
  );
}
