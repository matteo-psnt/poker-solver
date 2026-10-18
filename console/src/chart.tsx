/**
 * The public chart: the console's Blueprint grid alone, served by the blueprint
 * host at `/blueprint` beside the play page.
 *
 * A second entry rather than the console behind a filter, so the public build
 * carries none of the operator's pages. The route keeps the console's path and
 * search so a spot link reads the same on both, and Caddy maps the console's
 * `/api/blueprint/*` reads onto the server's own `/api/*`.
 */
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import {
  createRootRoute,
  createRoute,
  createRouter,
  Outlet,
  RouterProvider,
} from "@tanstack/react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { blueprintSearch } from "@/routes/blueprintSearch";
import { Charts } from "@/routes/Charts";
import "./index.css";
import "./chart.css";

function Page() {
  return (
    <div className="min-h-full">
      <div className="bar">
        <div className="site">
          <b>Blueprint</b>
          <nav>
            <a href="/">Play</a>
            <a className="on" href="/chart">
              Strategy
            </a>
          </nav>
          <span className="blurb">
            the trained strategy at every depth the bot plays, before purification
          </span>
        </div>
        <div className="end">
          <a href="https://github.com/matteo-psnt/poker-solver">source</a>
        </div>
      </div>
      <main className="mx-auto max-w-[1400px] p-4">
        <Outlet />
      </main>
    </div>
  );
}

const rootRoute = createRootRoute({ component: Page });
const routeTree = rootRoute.addChildren([
  createRoute({
    getParentRoute: () => rootRoute,
    path: "/blueprint",
    component: Charts,
    validateSearch: blueprintSearch,
  }),
]);

const router = createRouter({ routeTree });
const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, placeholderData: (previous: unknown) => previous } },
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
