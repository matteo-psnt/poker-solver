import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createMemoryHistory, createRouter, RouterProvider } from "@tanstack/react-router";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { routeTree } from "@/routes/tree";

/**
 * A line that has outrun its board is an ANSWER, not a refusal.
 *
 * It was a 422 handed to `Panel`'s error slot, so the ordinary act of stepping
 * into a postflop line turned the page red and said **unavailable**. The server
 * now says what it is waiting for and sends the line leading up to the question,
 * which is what lets the street appear as a column with empty cards in it.
 */
const PENDING = {
  op: "solver-node",
  path: "c/x",
  terminal: false,
  board: [],
  grid: null,
  children: [],
  button: 0,
  pot: 0,
  stack: null,
  pending: { street: "flop", needed: 3, have: 0 },
  line: [
    {
      kind: "spot",
      street: "preflop",
      actor: 0,
      chosen: "c",
      pot: 3,
      stack: 199,
      options: [
        { token: "f", type: "fold", amount: 0 },
        { token: "c", type: "call", amount: 1 },
      ],
    },
    {
      kind: "spot",
      street: "preflop",
      actor: 1,
      chosen: "x",
      pot: 4,
      stack: 198,
      options: [
        { token: "x", type: "check", amount: 0 },
        { token: "b4", type: "bet", amount: 4 },
      ],
    },
  ],
};

const RUN = {
  op: "blueprint-run",
  run: "run-production-025433",
  starting_stack: 200,
  small_blind: 1,
  big_blind: 2,
  combos: 1326,
};

/**
 * Six combos across three classes, which is enough for every state the grid
 * draws: AA trained, KK allocated but never visited, QQ blocked by the board.
 * Two of those look identical if anything collapses them, which is the failure
 * `RangeGrid`'s docstring calls the worst this page could commit.
 */
const COMBOS = ["AsAd", "AhAc", "KsKd", "KhKc", "QsQd", "QhQc"];

const NODE = {
  op: "solver-node",
  path: "c/x",
  terminal: false,
  board: ["As", "Kd", "7c"],
  grid: {
    street: "Flop",
    board: ["As", "Kd", "7c"],
    actor: 0,
    actions: ["f", "c", "r6"],
    combo_buckets: [0, 0, 1, 1, -1, -1],
    blocked: 2,
    trained_buckets: 1,
    buckets: {
      "0": { trained: true, strategy: [0.2, 0.5, 0.3] },
      "1": { trained: false, strategy: null },
    },
  },
  children: [
    { token: "f", type: "fold", amount: 0 },
    { token: "c", type: "call", amount: 2 },
  ],
  button: 0,
  pot: 4,
  stack: 198,
  pending: null,
  // The line as columns: two preflop decisions, then the flop it dealt. The
  // options are the whole menu at each past spot, which is what makes stepping
  // back into one a click rather than a round trip.
  line: [...PENDING.line, { kind: "dealt", street: "flop", cards: ["As", "Kd", "7c"], pot: 4 }],
};

/** What `/api/blueprint/node` answers. Set per-describe. */
let node: () => Response;

function answer(url: string): Response {
  if (url.startsWith("/api/blueprint/node")) return node();
  if (url.startsWith("/api/blueprint/run")) {
    return new Response(JSON.stringify(RUN), { status: 200 });
  }
  if (url.startsWith("/api/blueprint/combos")) {
    return new Response(JSON.stringify({ op: "combos", combos: COMBOS }), { status: 200 });
  }
  return new Response("{}", { status: 200 });
}

beforeEach(() => {
  node = () => new Response(JSON.stringify(PENDING), { status: 200 });
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => answer(String(url))),
  );
});
afterEach(() => vi.unstubAllGlobals());

function mountAt(path: string) {
  const router = createRouter({
    routeTree,
    history: createMemoryHistory({ initialEntries: [path] }),
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
}

describe("a line that has outrun its board", () => {
  const POSTFLOP = "/blueprint?tab=chart&path=c%2Fx";

  it("says what it is waiting for, in the street's own name", async () => {
    mountAt(POSTFLOP);
    await waitFor(() => expect(screen.getByText(/Deal 3 more cards/)).toBeTruthy());
  });

  it("does not file it as a panel fault", async () => {
    mountAt(POSTFLOP);
    await waitFor(() => expect(screen.getByText(/Deal 3 more cards/)).toBeTruthy());
    // `Panel` prefixes its error slot with "unavailable:" and turns the header
    // rule red. That is what this used to do to every postflop spot.
    expect(screen.queryByText(/unavailable/i)).toBeNull();
  });

  it("opens the deck, so the answer is one click away", async () => {
    mountAt(POSTFLOP);
    await waitFor(() => expect(screen.getByText(/Deal 3 more cards/)).toBeTruthy());
    expect(screen.getByTitle("As")).toBeTruthy();
    expect(screen.getByTitle("2c")).toBeTruthy();
  });

  it("puts the street in the line as a column of its own", async () => {
    mountAt(POSTFLOP);
    await waitFor(() => expect(screen.getByText(/Deal 3 more cards/)).toBeTruthy());
    // Named where it happens, between the preflop action and whatever follows —
    // not in a bar above the page that is there whether the line reaches it or
    // not. Two of them: the column header and the deck's own slot label.
    expect(screen.getAllByText("flop").length).toBeGreaterThan(0);
  });

  it("greys the panel for a line that does not exist", async () => {
    // A bookmark that outlived its action model IS a fault, and unlike a short
    // board there is nothing the page can ask for that would fix it.
    const stale = "'r5' is not available here. On offer: f, c, r6.";
    node = () => new Response(JSON.stringify({ error: stale }), { status: 422 });
    mountAt("/blueprint?tab=chart&path=r5");
    await waitFor(() => expect(screen.getByText(/unavailable/i)).toBeTruthy());
  });

  it("greys the panel for a mistyped board, which no deal can fix", async () => {
    const typo = "'Ax' is not a card.";
    node = () => new Response(JSON.stringify({ error: typo }), { status: 422 });
    mountAt("/blueprint?tab=chart&board=Ax");
    await waitFor(() => expect(screen.getByText(/unavailable/i)).toBeTruthy());
  });
});

describe("the line, as columns", () => {
  const SPOT = "/blueprint?tab=chart&path=c%2Fx&board=AsKd7c";

  beforeEach(() => {
    node = () => new Response(JSON.stringify(NODE), { status: 200 });
  });

  it("draws one column per thing that happened, the seats named not numbered", async () => {
    mountAt(SPOT);
    await waitFor(() => expect(screen.getByText("whole range")).toBeTruthy());
    // Seat 0 holds the button, which the server says rather than the client
    // assuming: BTN acted, BB acted, the flop came, BTN is to act.
    expect(screen.getAllByText("BTN").length).toBe(2);
    expect(screen.getAllByText("BB").length).toBe(1);
  });

  it("shows a past spot's whole menu, and touching it changes nothing", async () => {
    mountAt(SPOT);
    await waitFor(() => expect(screen.getByText("whole range")).toBeTruthy());

    // `fold` was on offer at the first spot and was not taken. It is drawn --
    // the menu is what makes a column worth reading -- and clicking it asks for
    // the SPOT, not for a line in which the fold happened. Looking is not
    // editing: you step back into a spot first, and take an action from there.
    fireEvent.click(screen.getAllByText("fold")[0] as HTMLElement);
    await waitFor(() =>
      expect(
        (globalThis.fetch as unknown as { mock: { calls: string[][] } }).mock.calls.some(
          ([url]) => String(url).includes("node") && String(url).includes("path=&"),
        ),
      ).toBe(true),
    );
    expect(
      (globalThis.fetch as unknown as { mock: { calls: string[][] } }).mock.calls.some(
        ([url]) => String(url).includes("node") && String(url).includes("path=f&"),
      ),
    ).toBe(false);
  });

  it("goes BACK to a spot when you click the column for it", async () => {
    mountAt(SPOT);
    await waitFor(() => expect(screen.getByText("whole range")).toBeTruthy());

    // The chart as it stood when that spot was being decided: the line BEFORE
    // its token, which for the first column is the preflop root.
    fireEvent.click(screen.getAllByTitle("read the chart at this spot")[0] as HTMLElement);
    await waitFor(() =>
      expect(
        (globalThis.fetch as unknown as { mock: { calls: string[][] } }).mock.calls.some(
          ([url]) => String(url).includes("node") && String(url).includes("path=&"),
        ),
      ).toBe(true),
    );
  });

  it("shows the flop's cards where the flop happened", async () => {
    mountAt(SPOT);
    await waitFor(() => expect(screen.getByText("whole range")).toBeTruthy());
    // The board is in the line, not in a bar above it -- so a preflop spot has
    // no flop column at all. See the root case below.
    expect(screen.getAllByTitle("As").length).toBeGreaterThan(0);
  });

  it("has no street column at all at the preflop root", async () => {
    node = () =>
      new Response(JSON.stringify({ ...NODE, path: "", line: [], pending: null }), { status: 200 });
    mountAt("/blueprint?tab=chart");
    await waitFor(() => expect(screen.getByText("whole range")).toBeTruthy());
    // The headline behaviour: you are not asked for a flop until the line
    // reaches one. Nothing on the page names a street before then.
    expect(screen.queryByText("flop")).toBeNull();
  });
});

/**
 * The grid itself, which nothing rendered until this existed.
 *
 * Every case above goes down the 422 branch, so `aggregate → RangeGrid →
 * Summary → Coverage → HandDetail` — half of it new — never executed, and a
 * throw anywhere in there is a blank page rather than a greyed panel.
 */
describe("the chart, drawn", () => {
  const SPOT = "/blueprint?tab=chart&path=c%2Fx&board=AsKd7c";

  beforeEach(() => {
    node = () => new Response(JSON.stringify(NODE), { status: 200 });
  });

  it("names the spot from the grid rather than from the path", async () => {
    mountAt(SPOT);
    await waitFor(() => expect(screen.getByText(/Flop · BTN to act/)).toBeTruthy());
  });

  it("totals the whole range, weighted by combos and net of the untrained", async () => {
    mountAt(SPOT);
    await waitFor(() => expect(screen.getByText("whole range")).toBeTruthy());
    // AA is the only trained class: 2 combos at [0.2, 0.5, 0.3]. KK is
    // allocated and unvisited, so it is reported beside the total, not in it.
    expect(screen.getByText("20.0%")).toBeTruthy();
    expect(screen.getByText("50.0%")).toBeTruthy();
    expect(screen.getByText("30.0%")).toBeTruthy();
    expect(screen.getByText(/over 2 combos; 2 untrained and left out/)).toBeTruthy();
  });

  it("labels the menu in blinds, so the rail reads as poker", async () => {
    mountAt(SPOT);
    await waitFor(() => expect(screen.getByText("whole range")).toBeTruthy());
    // `r6` at a big blind of 2, the same words the play table's buttons use.
    expect(screen.getByText("raise to 3bb")).toBeTruthy();
  });

  it("says how finely the spot can be answered at all", async () => {
    mountAt(SPOT);
    await waitFor(() => expect(screen.getByText("trained buckets")).toBeTruthy());
    // 6 combos in the fixture, 2 blocked, 2 buckets. Without this line a grid
    // where every hand looks alike reads as a broken chart rather than as an
    // abstraction that cannot tell those hands apart.
    expect(screen.getByText(/4 combos over 2 buckets/)).toBeTruthy();
    expect(screen.getByText(/at most 2 different mixes/)).toBeTruthy();
  });

  it("says it plainly when the abstraction has collapsed the spot entirely", async () => {
    // One bucket is the case this line exists for -- every hand on screen is
    // drawn alike because the solver cannot tell any of them apart. It is also
    // the case that read "1 buckets ... 1 different mixes".
    node = () =>
      new Response(
        JSON.stringify({
          ...NODE,
          grid: {
            ...NODE.grid,
            combo_buckets: [0, 0, 0, 0, -1, -1],
            trained_buckets: 1,
            buckets: { "0": { trained: true, strategy: [0.2, 0.5, 0.3] } },
          },
        }),
        { status: 200 },
      );
    mountAt(SPOT);
    await waitFor(() => expect(screen.getByText("trained buckets")).toBeTruthy());
    expect(screen.getByText(/over 1 bucket — every hand here shares one mix/)).toBeTruthy();
  });

  it("reports how much of the spot was learned", async () => {
    mountAt(SPOT);
    await waitFor(() => expect(screen.getByText("trained buckets")).toBeTruthy());
    expect(screen.getByText("1/2")).toBeTruthy();
    expect(screen.getByText("2 combos")).toBeTruthy();
  });

  it("pins a hand into the rail when you click its square", async () => {
    mountAt(SPOT);
    await waitFor(() => expect(screen.getByText("whole range")).toBeTruthy());
    expect(screen.getByText(/Hover a hand, or click to pin it/)).toBeTruthy();

    const square = screen.getByText("AA").closest("button");
    fireEvent.click(square as HTMLButtonElement);
    // The rail describes it and says so — the fix for reading a number meaning
    // holding the mouse still on one of 169 squares.
    await waitFor(() => expect(screen.getByText("pinned")).toBeTruthy());
    expect(screen.queryByText(/Hover a hand, or click to pin it/)).toBeNull();
  });

  it("drops the pin when the spot changes, rather than contradicting the grid", async () => {
    // A pin is a snapshot of one cell in one spot. Carried into another line it
    // kept describing the old strategy and combo count beside a grid drawn from
    // the new one -- and a board containing an ace draws AA as BLOCKED while the
    // rail still called it pinned.
    mountAt(SPOT);
    await waitFor(() => expect(screen.getByText("whole range")).toBeTruthy());
    fireEvent.click(screen.getByText("AA").closest("button") as HTMLButtonElement);
    await waitFor(() => expect(screen.getByText("pinned")).toBeTruthy());

    // Step to a different spot: same page, different chart. The first column's
    // taken action is the way back now.
    fireEvent.click(screen.getAllByTitle("read the chart at this spot")[0] as HTMLElement);
    await waitFor(() => expect(screen.getByText(/Hover a hand, or click to pin it/)).toBeTruthy());
    expect(screen.queryByText("pinned")).toBeNull();
  });
});
