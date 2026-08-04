import { Shell } from "@/components/Shell";
import { Cost } from "@/routes/Cost";
import { Evals } from "@/routes/Evals";
import { LegLog } from "@/routes/LegLog";
import { Legs } from "@/routes/Legs";
import { Overview } from "@/routes/Overview";
import { RunDetail } from "@/routes/RunDetail";
import { Runs } from "@/routes/Runs";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider, createRootRoute, createRoute, createRouter } from "@tanstack/react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { z } from "zod";
import "./index.css";

const rootRoute = createRootRoute({ component: Shell });

const routes = [
  createRoute({ getParentRoute: () => rootRoute, path: "/", component: Overview }),
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/legs",
    component: Legs,
    // Validated, so a hand-edited URL cannot put arbitrary state into the page.
    validateSearch: z.object({ cause: z.string().optional() }),
  }),
  createRoute({ getParentRoute: () => rootRoute, path: "/legs/$taskId", component: LegLog }),
  createRoute({ getParentRoute: () => rootRoute, path: "/runs", component: Runs }),
  createRoute({ getParentRoute: () => rootRoute, path: "/runs/$runId", component: RunDetail }),
  createRoute({ getParentRoute: () => rootRoute, path: "/evals", component: Evals }),
  createRoute({ getParentRoute: () => rootRoute, path: "/cost", component: Cost }),
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
      retry: false,
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
