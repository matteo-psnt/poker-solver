import { routeTree } from "@/routes/tree";
import { createMemoryHistory, createRouter } from "@tanstack/react-router";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

/**
 * The promise that no bookmark breaks.
 *
 * Fourteen destinations became six, so eight paths moved or merged. Every one of
 * them is a URL someone may have bookmarked, pasted into a message, or written
 * into a doc — and a redirect is exactly the kind of thing that is written once,
 * believed forever, and never exercised. A 404 for `/charts` would be a worse
 * outcome than the tidier route table is a better one.
 *
 * Routed against a memory history rather than rendered: what is under test is
 * where a path LANDS, not what it draws. Nothing here fetches.
 */
function at(path: string) {
  const router = createRouter({
    routeTree,
    history: createMemoryHistory({ initialEntries: [path] }),
  });
  return router.load().then(() => ({
    pathname: router.state.location.pathname,
    search: router.state.location.search as Record<string, unknown>,
    // What the COMPONENT is handed, which is not the same thing: a zod
    // `.default()` is applied when the route validates its search, and never
    // written back into the URL. Asserting on the location alone would say a
    // bare `/blueprint` has no tab, when the page opens on the chart.
    validated: (router.state.matches.at(-1)?.search ?? {}) as Record<string, unknown>,
  }));
}

beforeEach(() => {
  // The pages mount queries; nothing here should reach the network even so.
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response("{}", { status: 200 })),
  );
});
afterEach(() => vi.unstubAllGlobals());

describe("the six destinations", () => {
  it.each(["/", "/runs", "/experiments", "/tasks", "/blueprint", "/operate"])(
    "%s resolves to itself",
    async (path) => {
      expect((await at(path)).pathname).toBe(path);
    },
  );

  it("still routes a run and a task by id", async () => {
    expect((await at("/runs/run-production-025433")).pathname).toBe("/runs/run-production-025433");
    expect((await at("/tasks/run-a-000000-1")).pathname).toBe("/tasks/run-a-000000-1");
  });
});

describe("the paths that moved", () => {
  it("sends the chart and the play table to the blueprint page, on their tabs", async () => {
    // Two halves of one question — what did it learn — that used to be two
    // destinations with the box control belonging to neither.
    expect(await at("/charts")).toMatchObject({ pathname: "/blueprint", search: { tab: "chart" } });
    expect(await at("/play")).toMatchObject({ pathname: "/blueprint", search: { tab: "play" } });
  });

  it.each([
    ["/dispatch", "dispatch"],
    ["/share", "share"],
    ["/cost", "cost"],
    ["/activity", "activity"],
  ])("sends %s to the operate page on its tab", async (from, tab) => {
    expect(await at(from)).toMatchObject({ pathname: "/operate", search: { tab } });
  });

  it("sends the old evals list to the run list", async () => {
    // The evaluations still exist; they are filed under the run that earned
    // them, which is where the question "is THIS run any good" is asked.
    expect((await at("/evals")).pathname).toBe("/runs");
  });
});

describe("the search params that had to survive", () => {
  /**
   * The chart's spot is in the URL because the page it replaced held it in
   * `useState` — so its own docstring's claim that a bookmarked spot stays the
   * spot it was, was untrue of anything on it. Folding the chart into a tab is
   * exactly the kind of move that would quietly undo that.
   */
  it("keeps a shared chart spot intact through the new route", async () => {
    const landed = await at("/blueprint?tab=chart&path=r/c&board=AhKd7s&average=false");
    expect(landed).toMatchObject({
      pathname: "/blueprint",
      search: { tab: "chart", path: "r/c", board: "AhKd7s", average: false },
    });
  });

  it("defaults the tab rather than rejecting a bare /blueprint", async () => {
    // The default is not written into the URL — it is what the route hands the
    // component, so that is what has to be asserted.
    expect((await at("/blueprint")).validated).toMatchObject({ tab: "chart" });
  });

  it("defaults the operate page to dispatch", async () => {
    expect((await at("/operate")).validated).toMatchObject({ tab: "dispatch" });
  });

  it("keeps the task cause filter", async () => {
    expect((await at("/tasks?cause=oom")).search).toMatchObject({ cause: "oom" });
  });
});
