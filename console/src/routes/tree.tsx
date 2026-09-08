/**
 * The route table, and what each destination IS.
 *
 * Separate from `main.tsx` so it can be imported without bootstrapping the app:
 * the redirects below are the promise that no bookmark breaks, and a promise
 * nothing exercises is one that quietly stops being true. `routes.test.tsx`
 * mounts this tree against a memory history and follows every old path.
 */

import { createRootRoute, createRoute, redirect } from "@tanstack/react-router";
import { z } from "zod";
import { Shell } from "@/components/Shell";
import { Blueprint } from "@/routes/Blueprint";
import { Cost } from "@/routes/Cost";
import { Overview } from "@/routes/Overview";
import { RunDetail } from "@/routes/RunDetail";
import { Runs } from "@/routes/Runs";
import { TaskLog } from "@/routes/TaskLog";
import { Tasks } from "@/routes/Tasks";

const rootRoute = createRootRoute({ component: Shell });

/**
 * Six destinations, one per SUBJECT.
 *
 * A console is organised by what you are looking AT; the command line is organised
 * by what you can DO, because argv offers no other structure. One page per command
 * reached fourteen destinations, and answering *is this run any good* crossed
 * three of them -- Runs, RunDetail, Evals, Cost. That is one subject, so it is one
 * page.
 *
 * The test each survivor passed: is this a place you GO, or a follow-up you arrive
 * at? Tasks stays, because "why did that die" is a question people start with --
 * the run log cannot record a death, and the task account is the only thing that
 * can. An eval is not: it belongs to the run it scored.
 */
const routes = [
  createRoute({ getParentRoute: () => rootRoute, path: "/", component: Overview }),

  createRoute({
    getParentRoute: () => rootRoute,
    path: "/tasks",
    component: Tasks,
    // Validated, so a hand-edited URL cannot put arbitrary state into the page.
    validateSearch: z.object({
      cause: z.string().optional(),
      // The whole log is 6,031 attempts and 5.3 MB; the page rendered every
      // one into the DOM. A window is the default and `all` is a link, so the
      // rare question that needs the lot is still one click and still shareable.
      all: z.boolean().optional(),
    }),
  }),
  createRoute({ getParentRoute: () => rootRoute, path: "/tasks/$taskId", component: TaskLog }),

  createRoute({ getParentRoute: () => rootRoute, path: "/runs", component: Runs }),
  createRoute({ getParentRoute: () => rootRoute, path: "/runs/$runId", component: RunDetail }),

  /**
   * The blueprint, read as a grid or played against.
   *
   * `path`/`board`/`average` came with the chart and had to: they are search
   * params precisely because the page they replaced held them in `useState`, so
   * its own claim that a bookmarked spot survives was untrue of anything on it.
   * `tab` joins them for the same reason.
   */
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/blueprint",
    component: Blueprint,
    validateSearch: z.object({
      tab: z.enum(["chart", "play"]).default("chart"),
      path: z.string().default(""),
      board: z.string().default(""),
      average: z.boolean().default(true),
    }),
  }),

  /**
   * What the work cost, against what Azure actually billed.
   *
   * No search params: there is one thing on this page. It was a tab of an
   * `Operate` page beside dispatching and publishing, and those are gone -- a
   * console is for reading, and queueing work is what the command line is for.
   */
  createRoute({ getParentRoute: () => rootRoute, path: "/cost", component: Cost }),
];

/**
 * The old paths, kept alive.
 *
 * Eight routes moved or merged. A 404 for a URL someone bookmarked — or that a
 * doc or a chat message points at — is a worse outcome than the tidier route
 * table is a better one, and these cost one line each. `/evals` lands on the run
 * list rather than nowhere: the evaluations exist, they are just filed under the
 * run that earned them now.
 */
const MOVED: Record<string, { to: string; search?: Record<string, unknown> }> = {
  "/charts": { to: "/blueprint", search: { tab: "chart" } },
  "/play": { to: "/blueprint", search: { tab: "play" } },
  "/operate": { to: "/cost" },
  "/evals": { to: "/runs" },
};

const redirects = Object.entries(MOVED).map(([from, target]) =>
  createRoute({
    getParentRoute: () => rootRoute,
    path: from,
    beforeLoad: () => {
      throw redirect({ to: target.to, search: target.search ?? {}, replace: true });
    },
  }),
);

export const routeTree = rootRoute.addChildren([...routes, ...redirects]);
