import { isTransient } from "@/api/client";
import { routeTree } from "@/routes/tree";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider, createRouter } from "@tanstack/react-router";
import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import "./index.css";

const router = createRouter({ routeTree });

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
