import { routeTree } from "@/routes/tree";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider, createMemoryHistory, createRouter } from "@tanstack/react-router";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * The refusal that made postflop unusable.
 *
 * Replay needs the runout — *"This line reaches Flop and needs 3 board cards,
 * but 0 were given."* — and that 422 was handed to `Panel`'s error slot, so the
 * ordinary act of stepping into a postflop line turned the page red and said
 * **unavailable**. It is not a fault; it is the page asking for a board, and it
 * is the reason the deck exists.
 *
 * Asserted against a REAL 422 through the client rather than a mocked hook,
 * because what is under test is the round trip: `ApiError` has to carry the
 * status and the server's sentence for `shortBoard` to recognise it at all.
 */
const NEEDS_BOARD = "This line reaches Flop and needs 3 board cards, but 0 were given.";

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
      "0": { trained: true, strategy: [0.2, 0.5, 0.3], reach_count: 40 },
      "1": { trained: false, strategy: null, reach_count: 0 },
    },
  },
  children: [
    { token: "f", type: "fold", amount: 0 },
    { token: "c", type: "call", amount: 2 },
  ],
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
  // The page also mounts the box control and the run picker above the tabs.
  // Neither is under test, but both have to be answerable or the render that
  // is under test never happens.
  if (url.startsWith("/api/runs")) {
    return new Response(JSON.stringify({ op: "runs", runs: [] }), { status: 200 });
  }
  return new Response("{}", { status: 200 });
}

beforeEach(() => {
  node = () => new Response(JSON.stringify({ error: NEEDS_BOARD }), { status: 422 });
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

describe("a line that needs a board it has not got", () => {
  const POSTFLOP = "/blueprint?tab=chart&path=c%2Fx";

  it("shows the server's own sentence, which names the street and the count", async () => {
    mountAt(POSTFLOP);
    await waitFor(() => expect(screen.getByText(NEEDS_BOARD)).toBeTruthy());
  });

  it("does not file it as a panel fault", async () => {
    mountAt(POSTFLOP);
    await waitFor(() => expect(screen.getByText(NEEDS_BOARD)).toBeTruthy());
    // `Panel` prefixes its error slot with "unavailable:" and turns the header
    // rule red. That is what this line used to do to every postflop spot.
    expect(screen.queryByText(/unavailable/i)).toBeNull();
  });

  it("opens the deck, so the answer is one click away", async () => {
    mountAt(POSTFLOP);
    await waitFor(() => expect(screen.getByText(NEEDS_BOARD)).toBeTruthy());
    expect(screen.getByTitle("As")).toBeTruthy();
    expect(screen.getByTitle("2c")).toBeTruthy();
  });

  it("still suppresses the empty menu, which is a consequence and not a fact", async () => {
    mountAt(POSTFLOP);
    await waitFor(() => expect(screen.getByText(NEEDS_BOARD)).toBeTruthy());
    // There are actions from here; the server simply cannot say what they are
    // until it has a board. "no actions from here" would be a lie.
    expect(screen.queryByText(/no actions from here/)).toBeNull();
  });

  it("treats a mistyped board the same way, not as a fault", async () => {
    // Reachable through the paste field, which is the text format this page was
    // fixed for. Every board refusal names a card; no path refusal does.
    const typo = "'Ax' is not a card.";
    node = () => new Response(JSON.stringify({ error: typo }), { status: 422 });
    mountAt("/blueprint?tab=chart&board=Ax");
    await waitFor(() => expect(screen.getByText(typo)).toBeTruthy());
    expect(screen.queryByText(/unavailable/i)).toBeNull();
  });

  it("still greys the panel for a line that does not exist", async () => {
    // A bookmark that outlived its action model IS a fault, and naming a token
    // rather than a card is what tells the two apart.
    const stale = "'r5' is not available here. On offer: f, c, r6.";
    node = () => new Response(JSON.stringify({ error: stale }), { status: 422 });
    mountAt("/blueprint?tab=chart&path=r5");
    await waitFor(() => expect(screen.getByText(/unavailable/i)).toBeTruthy());
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
    await waitFor(() => expect(screen.getByText(/Flop · seat 0 to act/)).toBeTruthy());
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

    // Step into a different line: same page, different spot.
    fireEvent.click(screen.getByText("preflop"));
    await waitFor(() => expect(screen.getByText(/Hover a hand, or click to pin it/)).toBeTruthy());
    expect(screen.queryByText("pinned")).toBeNull();
  });
});
