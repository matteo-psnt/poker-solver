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

function Page() {
  return (
    <div className="min-h-full">
      <header className="flex items-center gap-4 border-b border-[var(--border)] px-4 py-2 font-mono text-[11px] text-[var(--fg-muted)]">
        <span className="tracking-widest text-[var(--fg-faint)]">BLUEPRINT</span>
        <span>the trained strategy, before purification</span>
        <a className="ml-auto text-[var(--fg)] hover:underline" href="/">
          play against it →
        </a>
        <a
          className="text-[var(--fg)] hover:underline"
          href="https://github.com/matteo-psnt/poker-solver"
        >
          source
        </a>
      </header>
      <main className="max-w-[1400px] p-4">
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
