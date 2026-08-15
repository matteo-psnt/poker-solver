import { routeTree } from "@/routes/tree";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider, createMemoryHistory, createRouter } from "@tanstack/react-router";
import { render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * A running task's log has to keep arriving.
 *
 * `useLog` was `refetchInterval: false` for every task, on the grounds that "a
 * published log for a finished task does not change". True of a finished one,
 * and the whole bug for a running one: the node republishes the log on a timer
 * and the console never asked again, so an open log sat frozen at whatever it
 * held when the page mounted.
 *
 * Asserted against the query cache rather than by advancing timers: what is
 * under test is the DECISION — poll or do not — and reading it back is exact,
 * where driving a 15s interval through fake timers tests React's scheduler.
 */
const LINES = { op: "logs", task: "task-a", lines: ["fetching warm-start prior"] };

function row(overrides: Record<string, unknown>) {
  return {
    task_id: "task-a",
    attempt: 1,
    job_id: "poker-20260815",
    run_id: "run-a",
    op: "train-static",
    what: "train-static run-a",
    cause: "running",
    cause_source: "batch",
    workers: 16,
    units: 0,
    started_at: "2026-08-15T09:00:00+00:00",
    ended_at: null,
    ...overrides,
  };
}

let rows: ReturnType<typeof row>[];

function answer(url: string): Response {
  if (url.startsWith("/api/logs/")) {
    return new Response(JSON.stringify(LINES), { status: 200 });
  }
  if (url.startsWith("/api/tasks")) {
    return new Response(JSON.stringify({ op: "tasks", rows, source_rows: rows.length }), {
      status: 200,
    });
  }
  return new Response("{}", { status: 200 });
}

beforeEach(() => {
  rows = [row({})];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (url: string) => answer(String(url))),
  );
});
afterEach(() => vi.unstubAllGlobals());

function mount() {
  const router = createRouter({
    routeTree,
    history: createMemoryHistory({ initialEntries: ["/tasks/task-a"] }),
  });
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  render(
    <QueryClientProvider client={client}>
      <RouterProvider router={router} />
    </QueryClientProvider>,
  );
  return client;
}

const logInterval = (client: QueryClient) =>
  client
    .getQueryCache()
    .findAll({ queryKey: ["log"] })
    .at(0)?.observers[0]?.options.refetchInterval;

describe("a task that is still running", () => {
  it("keeps asking for the log", async () => {
    const client = mount();
    await waitFor(() => expect(screen.getByText(/fetching warm-start prior/)).toBeTruthy());
    // The defect, stated as itself: this was `false` for every task.
    expect(logInterval(client)).toBeGreaterThan(0);
  });

  it("says so, so the panel is not read as a finished one", async () => {
    mount();
    await waitFor(() => expect(screen.getByText(/Published log · live/)).toBeTruthy());
  });
});

describe("a task that has ended", () => {
  beforeEach(() => {
    rows = [row({ ended_at: "2026-08-15T11:00:00+00:00", cause: "success" })];
  });

  it("does not poll a log that cannot change", async () => {
    const client = mount();
    await waitFor(() => expect(screen.getByText(/fetching warm-start prior/)).toBeTruthy());
    expect(logInterval(client)).toBe(false);
  });

  it("drops the live marker", async () => {
    mount();
    await waitFor(() => expect(screen.getByText(/fetching warm-start prior/)).toBeTruthy());
    expect(screen.queryByText(/· live/)).toBeNull();
  });
});
