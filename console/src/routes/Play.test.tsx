import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createEvent, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Play } from "./Play";

/**
 * The two things a table has to get right that a screenshot cannot show.
 *
 * The KEYS, because the page's whole value is how many hands you get through
 * and the buttons move as the menu changes size — so a digit that folds when it
 * should call is worse than no shortcut at all.
 *
 * The GOODBYE, because sessions live on the blueprint server in a store of 64
 * that evicts the oldest. `Blueprint.tsx` claimed unmounting "ends it where it
 * lives" while nothing called the endpoint, and nothing on screen would ever
 * have shown that: the cost lands on whoever else is playing.
 */
const RUN = {
  op: "blueprint-run",
  run: "run-production-025433",
  starting_stack: 200,
  small_blind: 1,
  big_blind: 2,
  combos: 1326,
  loading: null,
  can_switch: true,
};

const HAND = {
  op: "hand",
  session: "sess-one",
  over: false,
  street: "Preflop",
  board: [],
  pot: 3,
  stacks: [198, 199],
  human_seat: 0,
  button: 0,
  to_act: 0,
  hole_cards: ["Ah", "Kd"],
  bot_hole_cards: null,
  legal: [
    { token: "f", type: "fold", amount: 0 },
    { token: "c", type: "call", amount: 2 },
    { token: "r6", type: "raise", amount: 6 },
    // The wire spelling is `all_in` (`str(ActionType.ALL_IN)` is `name.lower()`).
    // No fixture carried one, which is why three lookups keyed on the DISPLAY
    // spelling `all-in` went unnoticed.
    { token: "A", type: "all_in", amount: 198 },
  ],
  payoff: null,
  showdown: false,
  bot_decisions: 1,
  bot_untrained_decisions: 1,
  log: [
    {
      seat: 1,
      actor: "bot",
      action: "raise",
      amount: 6,
      street: "Preflop",
      untrained: true,
      mix: null,
    },
  ],
};

let sent: { url: string; method: string; body: unknown }[] = [];

beforeEach(() => {
  sent = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string, init?: RequestInit) => {
      const method = init?.method ?? "GET";
      sent.push({
        url: String(url),
        method,
        body: init?.body ? JSON.parse(String(init.body)) : null,
      });
      if (String(url).startsWith("/api/blueprint/run")) {
        return new Response(JSON.stringify(RUN), { status: 200 });
      }
      if (String(url).endsWith("/action")) {
        return new Response(JSON.stringify({ ...HAND, session: "sess-one", pot: 5 }), {
          status: 200,
        });
      }
      if (String(url) === "/api/blueprint/play") {
        return new Response(JSON.stringify(HAND), { status: 200 });
      }
      return new Response(JSON.stringify({ session: "sess-one", dropped: true }), { status: 200 });
    }),
  );
});
afterEach(() => vi.unstubAllGlobals());

function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <Play />
    </QueryClientProvider>,
  );
}

/**
 * Deal, once the page can. The button renders immediately and is DISABLED
 * until `/api/blueprint/run` answers — sizes are meaningless without the
 * stakes — so clicking on sight clicks a dead control.
 */
async function dealOn(rendered: ReturnType<typeof mount>) {
  const button = (await screen.findByRole("button", { name: /deal/ })) as HTMLButtonElement;
  await waitFor(() => expect(button.disabled).toBe(false));
  fireEvent.click(button);
  await waitFor(() => expect(screen.getByText("to act")).toBeTruthy());
  return rendered;
}

const deal = () => dealOn(mount());

const acted = () => sent.filter((call) => call.url.endsWith("/action"));

describe("the table", () => {
  it("shows the hand once it is dealt", async () => {
    await deal();
    expect(screen.getByText("pot")).toBeTruthy();
    // Both seats, both labelled by position rather than by seat number.
    expect(screen.getByText("BTN")).toBeTruthy();
    expect(screen.getByText("BB")).toBeTruthy();
  });

  it("keeps the untrained count on screen, not behind a detail view", async () => {
    await deal();
    expect(screen.getByText(/uniform-random, not strategy/)).toBeTruthy();
  });

  it("offers every legal action, sized in blinds", async () => {
    await deal();
    const menu = screen.getAllByRole("button").map((button) => button.textContent);
    expect(menu).toContain("fold1");
    expect(menu).toContain("call2");
    // `r6` at a big blind of 2. The log below says the same words for the same
    // bet, which is the point of `describeAction` and `label` agreeing.
    expect(menu).toContain("raise to 3bb3");
  });
});

/**
 * What the bot just did, at the table.
 *
 * `POST /action` auto-plays the bot and returns, so its move never had a moment
 * of its own on screen: the pot number moved and nothing else did, and the only
 * record was a line two panels down. Reading the log on every hand is exactly
 * the cost this page cannot afford.
 */
describe("what just happened", () => {
  it("draws the bot's move in front of it", async () => {
    await deal();
    const chips = screen.getAllByTestId("did").map((chip) => chip.textContent);
    // `r6` at a big blind of 2, and the caveat travels with the move.
    expect(chips).toEqual(["raise to 3bbuntrained"]);
  });

  it("names the street, which the board only implies — and preflop not at all", async () => {
    await deal();
    // Twice: once at the table, once as the log's street heading. One would
    // mean the table lost its label, which is the whole point preflop, where
    // there is no board to infer it from.
    expect(screen.getAllByText("Preflop")).toHaveLength(2);
  });

  it("still answers the question after the street has turned, and names the street", async () => {
    // You raise, the bot calls, the flop comes. Scoped to the current street
    // this went blank at exactly the moment you were asking what the bot did —
    // which is the most common shape a hand has.
    const onFlop = { ...HAND, street: "Flop", board: ["As", "Kd", "7c"] };
    vi.stubGlobal(
      "fetch",
      vi.fn(async (url: string) =>
        String(url).startsWith("/api/blueprint/run")
          ? new Response(JSON.stringify(RUN), { status: 200 })
          : new Response(JSON.stringify(onFlop), { status: 200 }),
      ),
    );
    await deal();
    const chips = screen.getAllByTestId("did").map((chip) => chip.textContent);
    // Carried, but labelled with the street it was taken on, so it cannot read
    // as something the bot did on the flop now showing.
    expect(chips).toEqual(["raise to 3bbPreflopuntrained"]);
  });
});

describe("the keys", () => {
  it("sends the action a digit names, in menu order", async () => {
    await deal();
    fireEvent.keyDown(window, { key: "2" });
    await waitFor(() => expect(acted()).toHaveLength(1));
    expect(acted()[0]?.body).toEqual({ token: "c" });
  });

  it("takes an initial for the common ones", async () => {
    await deal();
    fireEvent.keyDown(window, { key: "f" });
    await waitFor(() => expect(acted()).toHaveLength(1));
    expect(acted()[0]?.body).toEqual({ token: "f" });
  });

  it("shoves on `a`, whose wire word is `all_in` and not `all-in`", async () => {
    // `f`/`k`/`c` worked only because those words are identical in both
    // spellings. Facing a shove, `a` did nothing at all.
    await deal();
    fireEvent.keyDown(window, { key: "a" });
    await waitFor(() => expect(acted()).toHaveLength(1));
    expect(acted()[0]?.body).toEqual({ token: "A" });
  });

  it("leaves space alone on a focused button, so it still activates it", async () => {
    // `preventDefault()` used to run unconditionally, cancelling the native
    // activation, and then declined to deal because a hand was live -- so
    // tabbing onto an action and pressing space did nothing whatsoever.
    await deal();
    const [button] = screen.getAllByRole("button");
    if (!button) throw new Error("no button to focus");
    const event = createEvent.keyDown(button, { key: " " });
    fireEvent(button, event);
    expect(event.defaultPrevented).toBe(false);
  });

  it("ignores a digit with no action behind it", async () => {
    await deal();
    fireEvent.keyDown(window, { key: "9" });
    expect(acted()).toHaveLength(0);
  });

  it("does not fold you for typing into a field", async () => {
    await deal();
    // The seat select is a real control on this page; typing in one must be
    // typing, and `f` there must not be a fold.
    fireEvent.keyDown(screen.getByRole("combobox"), { key: "f" });
    expect(acted()).toHaveLength(0);
  });
});

describe("leaving", () => {
  it("hands the session back when the tab unmounts", async () => {
    const { unmount } = await dealOn(mount());
    unmount();
    await waitFor(() =>
      expect(
        sent.some(
          (call) => call.method === "DELETE" && call.url === "/api/blueprint/play/sess-one",
        ),
      ).toBe(true),
    );
  });
});
