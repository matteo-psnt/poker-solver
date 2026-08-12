import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * The time unit `Cost` hands its chart — a contract between two modules, and
 * the one thing the uPlot-to-Recharts port could silently get wrong.
 *
 * uPlot's `time: true` scale reads Unix SECONDS unless told `ms: 1`, so `Cost`
 * divided by 1000 to feed it. `Date` and Recharts want MILLISECONDS. Carrying
 * that division across would have put every tick and tooltip in January 1970 —
 * visibly wrong on screen, and invisible to tsc, which sees `number` either way.
 *
 * Asserted by capturing the props rather than by reading rendered axis ticks:
 * Recharts draws through `ResponsiveContainer`, which measures its parent, and
 * jsdom has no layout — so the chart renders empty however the observer is
 * stubbed. The rendered tick would have been a test of jsdom's layout engine.
 * What matters is the number crossing the boundary, and that is exact.
 */
const captured: { times: number[]; values: number[] }[] = [];

vi.mock("@/components/StepChart", () => ({
  StepChart: (props: { times: number[]; values: number[]; label: string }) => {
    captured.push({ times: props.times, values: props.values });
    return <div data-testid="step-chart" />;
  },
}));

const { Cost } = await import("./Cost");

const AT = "2026-08-12T09:00:00+00:00";
const LAST = "2026-08-12T15:00:00+00:00";

const COST = {
  op: "cost",
  hours: 0,
  task_hours: 12.5,
  tasks: 3,
  unended: 0,
  peak_concurrency: 2,
  first_at: AT,
  last_at: LAST,
  rate_per_node_hour: null,
  dollars: null,
  billed: null,
  billed_reason: null,
  series: [
    { at: AT, running: 1 },
    { at: LAST, running: 0 },
  ],
};

beforeEach(() => {
  captured.length = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(COST), { status: 200 })),
  );
});

afterEach(() => vi.unstubAllGlobals());

function mount() {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <Cost />
    </QueryClientProvider>,
  );
}

describe("the concurrency chart", () => {
  it("is handed epoch MILLISECONDS, not the seconds uPlot wanted", async () => {
    mount();
    await waitFor(() => expect(screen.getByTestId("step-chart")).toBeTruthy());

    const { times } = captured.at(-1) ?? { times: [] };
    expect(times).toEqual([Date.parse(AT), Date.parse(LAST)]);
    // The failure this exists for, stated as itself: seconds land in 1970.
    expect(new Date(times[0] ?? 0).getUTCFullYear()).toBe(2026);
  });

  it("passes the running counts through unchanged", async () => {
    mount();
    await waitFor(() => expect(screen.getByTestId("step-chart")).toBeTruthy());
    // Including the zero: a pool that emptied is a real observation, not a gap.
    expect(captured.at(-1)?.values).toEqual([1, 0]);
  });

  it("renders the node-hours the area under that curve integrates to", async () => {
    mount();
    await waitFor(() => expect(screen.getByText(/12\.5/)).toBeTruthy());
  });
});
