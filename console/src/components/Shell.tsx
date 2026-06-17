import { useNow } from "@/api/queries";
import type { Phase } from "@/api/types";
import { count } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Link, Outlet, useRouterState } from "@tanstack/react-router";
import { Activity, Grid3x3, LayoutDashboard, Rocket, ScrollText } from "lucide-react";

/**
 * Six destinations, one per SUBJECT -- see `routes/tree.tsx` for the test each one
 * passed. Ungrouped, because six items do not need headings.
 */
const NAV = [
  { to: "/", label: "Now", icon: LayoutDashboard },
  { to: "/runs", label: "Runs", icon: Activity },
  { to: "/tasks", label: "Tasks", icon: ScrollText },
  { to: "/blueprint", label: "Blueprint", icon: Grid3x3 },
  { to: "/operate", label: "Operate", icon: Rocket },
] as const;

/**
 * Left rail plus a status bar that is present on every page.
 *
 * Nodes, burn rate and live task count are the things worth knowing wherever
 * you are, so they are never a click away. They read the Now view, which the
 * Overview polls too -- TanStack Query dedupes them, so this costs nothing.
 */
/** Phases that mean a task is occupying a node or waiting for one. */
const IN_FLIGHT = new Set<Phase>(["running", "starting", "queued"]);

export function Shell() {
  const path = useRouterState({ select: (s) => s.location.pathname });
  // The same composed view the Now page polls, not the two commands on their
  // own: one memo, one cadence, and served stale-while-revalidate -- so the bar
  // never waits on Azure, and a page other than Now costs the same sweep
  // rather than two extra Batch reads of its own.
  const view = useNow();
  // `?.` at every step: this is the SHELL, and a 200 whose body lacks a part
  // must take down nothing rather than every page at once.
  const pool = { data: view.data?.parts?.pool?.payload };
  const jobs = { data: view.data?.parts?.jobs?.payload };

  const nodes = pool.data?.current_dedicated_nodes ?? null;
  // `?.` past `jobs` as well as past `data`: this is the app SHELL, so a 200
  // whose body lacks the field takes down every page at once rather than one
  // panel. Same one-character gap as `RunPicker`'s `current` had.
  const live =
    jobs.data?.jobs?.reduce(
      (total, job) =>
        total +
        // `phase`, not a fourth copy of Batch's enum spelling. This file had a
        // private `state()` helper duplicating `shortState` under another name,
        // which is why it survived that deletion: nothing grepping for the
        // callers of `shortState` could see it.
        job.tasks.filter((t) => IN_FLIGHT.has(t.phase)).length,
      0,
    ) ?? null;

  return (
    <div className="flex min-h-full">
      <nav className="w-[168px] shrink-0 border-r border-[var(--border)] p-3">
        <div className="mb-4 px-2 font-mono text-[11px] tracking-widest text-[var(--fg-faint)]">
          POKER-SOLVER
        </div>
        <ul className="space-y-0.5">
          {NAV.map(({ to, label, icon: Icon }) => {
            const active = to === "/" ? path === "/" : path.startsWith(to);
            return (
              <li key={to}>
                <Link
                  to={to}
                  className={cn(
                    "flex items-center gap-2 rounded px-2 py-1.5 text-[12px]",
                    active
                      ? "bg-white/[0.07] text-[var(--fg)]"
                      : "text-[var(--fg-muted)] hover:bg-white/[0.04] hover:text-[var(--fg)]",
                  )}
                >
                  <Icon className="size-3.5" />
                  {label}
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className="min-w-0 flex-1">
        <header className="flex items-center gap-4 border-b border-[var(--border)] px-4 py-2 font-mono text-[11px] text-[var(--fg-muted)]">
          {/* An idle pool at rest is CORRECT and cheap. It is deliberately not
              styled as an alarm. */}
          <span>
            pool <span className="tnum text-[var(--fg)]">{count(nodes)}</span>
            {pool.data?.target_dedicated_nodes != null && (
              <span className="tnum">/{pool.data.target_dedicated_nodes}</span>
            )}
          </span>
          {/* What it costs NOW, not the per-node list price: the rate is the
              same every day and the node count is the thing that moves. */}
          {pool.data?.burn_per_hour != null && (
            <span title={pool.data.hourly_cost ?? undefined}>
              · <span className="tnum text-[var(--fg)]">${pool.data.burn_per_hour.toFixed(2)}</span>
              /hr
            </span>
          )}
          <span>
            · <span className="tnum text-[var(--fg)]">{count(live)}</span> running
          </span>
        </header>
        <main className="max-w-[1400px] p-4">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
