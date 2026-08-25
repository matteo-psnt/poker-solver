import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { createMemoryHistory, createRouter, RouterProvider } from "@tanstack/react-router";
import { render, screen, waitFor } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { count, mbb, percent } from "@/lib/format";

/**
 * What `RunDetail` hands Recharts, and what it does with what Recharts hands
 * back.
 *
 * Recharts is mocked rather than rendered, for the reason `Cost.test.tsx` gives:
 * it draws through `ResponsiveContainer`, which measures its parent, and jsdom
 * has no layout — so the chart is empty however the observer is stubbed, and an
 * assertion on a rendered tick would be a test of jsdom's layout engine.
 *
 * The tooltip formatters are the point. Recharts 3 widened their parameters
 * from the datum's own type to `ValueType`/`ReactNode`, which is a TYPE change
 * tsc checks and a RUNTIME contract nothing did: a tooltip only renders on
 * hover, so a formatter handed something it cannot coerce would have shipped.
 * These call them the way the library does.
 */
type TooltipCapture = {
  formatter: (...args: never[]) => unknown;
  labelFormatter: (...args: never[]) => unknown;
};

/**
 * One entry per chart, paired with its own tooltip. Paired rather than indexed
 * because both panels mount a `LineChart` and the order they render in is not
 * this test's business: a tooltip is found by the data its chart was given.
 */
const charts: { keys: string[]; tooltip?: TooltipCapture }[] = [];

const chartFor = (key: string) => {
  const found = charts.find((chart) => chart.keys.includes(key));
  if (!found?.tooltip) throw new Error(`no chart mounted a tooltip for ${key}`);
  return found.tooltip;
};

vi.mock("recharts", () => {
  const wrap =
    (testId: string) =>
    ({ children }: { children?: ReactNode }) => <div data-testid={testId}>{children}</div>;
  return {
    ResponsiveContainer: wrap("responsive-container"),
    LineChart: ({ children, data }: { children?: ReactNode; data?: object[] }) => {
      charts.push({ keys: Object.keys(data?.[0] ?? {}) });
      return <div data-testid="line-chart">{children}</div>;
    },
    CartesianGrid: () => null,
    Legend: () => null,
    Line: () => null,
    XAxis: () => null,
    YAxis: () => null,
    Tooltip: (props: TooltipCapture) => {
      // React renders a parent before its children, so the chart still open is
      // this tooltip's own.
      const chart = charts.at(-1);
      if (chart) chart.tooltip = props;
      return null;
    },
  };
});

const { routeTree } = await import("./tree");

const RUN_ID = "run-production-025433-1095";

const VIEW = {
  at: "2026-08-24T12:00:00+00:00",
  parts: {
    run: {
      payload: {
        op: "runinfo",
        run_id: RUN_ID,
        status: "completed",
        iterations: 30_000_000,
      },
      error: null,
    },
    progress: {
      payload: {
        op: "progress",
        rows: [
          { iteration: 10_000_000, coverage: 0.5, visits_per_infoset: 12.3 },
          { iteration: 30_000_000, coverage: 1.0, visits_per_infoset: 34.5 },
        ],
        coverage_plateau_iteration: null,
      },
      error: null,
    },
    curve: {
      payload: {
        op: "curve",
        tier: "4/2/2",
        points: [
          { iteration: 10_000_000, exploitability_mbb: 1381.0, coverage: 0.5 },
          { iteration: 30_000_000, exploitability_mbb: 1037.0, coverage: 1.0 },
        ],
        missing_iterations: [],
      },
      error: null,
    },
  },
};

beforeEach(() => {
  charts.length = 0;
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(JSON.stringify(VIEW), { status: 200 })),
  );
});
afterEach(() => vi.unstubAllGlobals());

function mountRun() {
  const router = createRouter({
    routeTree,
    history: createMemoryHistory({ initialEntries: [`/runs/${RUN_ID}`] }),
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
}

describe("the exploitability curve", () => {
  it("renders a chart once the run has scored rungs", async () => {
    mountRun();

    await waitFor(() => expect(screen.getAllByTestId("line-chart").length).toBeGreaterThan(0));
  });

  it("formats a tooltip value the way Recharts 3 calls it", async () => {
    mountRun();
    await waitFor(() => expect(charts.length).toBeGreaterThan(0));

    // Recharts passes `(value, name, item, index, payload)`, and the value
    // arrives as `ValueType | undefined` rather than the number in the datum.
    const { formatter, labelFormatter } = chartFor("exploitability_mbb");

    expect(formatter(...([1037.0, "exploitability_mbb", {}, 0, []] as never[]))).toBe(mbb(1037.0));
    expect(labelFormatter(...([30_000_000, []] as never[]))).toBe(`${count(30_000_000)} it`);
  });

  it("coerces a stringified value rather than rendering NaN", async () => {
    mountRun();
    await waitFor(() => expect(charts.length).toBeGreaterThan(0));

    // `ValueType` admits `string`; the coercion is what stops that reaching
    // `toFixed` and printing NaN into a chart nobody would think to distrust.
    const { formatter } = chartFor("exploitability_mbb");

    expect(formatter(...(["1037" as unknown as number, "x", {}, 0, []] as never[]))).toBe(
      mbb(1037.0),
    );
  });
});

describe("the coverage chart's formatter", () => {
  it("splits on the series name, which Recharts hands over as the second argument", async () => {
    mountRun();
    await waitFor(() => expect(charts.length).toBeGreaterThan(1));

    const coverage = chartFor("visits_per_infoset");

    expect(coverage.formatter(...([0.5, "coverage", {}, 0, []] as never[]))).toBe(percent(0.5));
    expect(coverage.formatter(...([12.34, "rungs", {}, 0, []] as never[]))).toBe("12.3");
  });
});
