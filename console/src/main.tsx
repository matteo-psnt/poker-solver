import { isTransient } from "@/api/client";
import { Shell } from "@/components/Shell";
import { Cost } from "@/routes/Cost";
import { Dispatch } from "@/routes/Dispatch";
import { Evals } from "@/routes/Evals";
import { Experiments } from "@/routes/Experiments";
import { Overview } from "@/routes/Overview";
import { Play } from "@/routes/Play";
import { RunDetail } from "@/routes/RunDetail";
import { Runs } from "@/routes/Runs";
import { Share } from "@/routes/Share";
import { Solver } from "@/routes/Solver";
import { TaskLog } from "@/routes/TaskLog";
import { Tasks } from "@/routes/Tasks";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider, createRootRoute, createRoute, createRouter } from "@tanstack/react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { z } from "zod";
import "./index.css";

const rootRoute = createRootRoute({ component: Shell });

const routes = [
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/",
    component: Overview,
  }),
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/tasks",
    component: Tasks,
    // Validated, so a hand-edited URL cannot put arbitrary state into the page.
    validateSearch: z.object({ cause: z.string().optional() }),
  }),
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/tasks/$taskId",
    component: TaskLog,
  }),
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/runs",
    component: Runs,
  }),
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/runs/$runId",
    component: RunDetail,
  }),
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/solver",
    component: Solver,
  }),
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/play",
    component: Play,
  }),
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/evals",
    component: Evals,
  }),
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/cost",
    component: Cost,
  }),
  // The write surfaces. No search params: a dispatch form's state is what the
  // operator is about to DO, and putting that in the URL makes a half-filled
  // submission bookmarkable and shareable — which is the wrong thing to be able
  // to hand someone.
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/dispatch",
    component: Dispatch,
  }),
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/share",
    component: Share,
  }),
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/experiments",
    component: Experiments,
  }),
];

const router = createRouter({ routeTree: rootRoute.addChildren(routes) });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      // A cloud read that failed is unlikely to succeed on an immediate retry,
      // and the panel already shows the reason. Retrying would only delay it.
      // The exception is the server not being there YET — see `isTransient`.
      // `console-dev` starts Vite and `serve` together and Vite wins by several
      // seconds, so at boot this is the NORMAL state, not an incident.
      retry: (failureCount, error) => isTransient(error) && failureCount < 5,
      // Backs off to ~15s of trying, which covers a cold `serve` (the Azure SDK
      // import alone is most of it). Capped rather than growing: a page left
      // open through a real outage should keep asking at a sane cadence, not
      // drift to minutes between attempts.
      retryDelay: (attempt) => Math.min(500 * 2 ** attempt, 5_000),
      // Keep showing the last good answer while refetching, so a panel never
      // blanks between polls.
      placeholderData: (previous: unknown) => previous,
      refetchOnWindowFocus: true,
    },
  },
});

const root = document.getElementById("root");
if (!root) throw new Error("no #root");

createRoot(root).render(
  <StrictMode>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </StrictMode>,
);
