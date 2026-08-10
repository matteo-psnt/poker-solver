import { useJobs, usePool } from "@/api/queries";
import { count } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Link, Outlet, useRouterState } from "@tanstack/react-router";
import {
  Activity,
  DollarSign,
  FlaskConical,
  Gauge,
  Grid3x3,
  LayoutDashboard,
  Rocket,
  ScrollText,
  Spade,
  Split,
  UploadCloud,
} from "lucide-react";

/**
 * The nav, grouped the way `poker-solver --help` is grouped.
 *
 * A flat list was fine at seven items and stopped being fine at ten, and the
 * obvious fixes — alphabetical, or by how often a page is opened — would both
 * be a second organising idea for a tool that already has one. The command
 * line's `COMMANDS` tuple is ordered in groups on purpose (open a surface,
 * dispatch and account for work, read the record), so the console uses the
 * same three. Someone who knows one surface then knows where to look on the
 * other, and a new page has an obvious home rather than a position.
 *
 * The fourth CLI group — run on a node — has no entry and cannot get one:
 * those commands are compute invoked BY the node wrapper, and a button here
 * would train on whichever laptop is serving the console.
 */
const NAV = [
  {
    group: "surface",
    items: [
      { to: "/", label: "Overview", icon: LayoutDashboard },
      { to: "/solver", label: "Solver", icon: Grid3x3 },
      { to: "/play", label: "Play", icon: Spade },
    ],
  },
  {
    group: "work",
    items: [
      { to: "/dispatch", label: "Dispatch", icon: Rocket },
      { to: "/tasks", label: "Tasks", icon: ScrollText },
      { to: "/share", label: "Share", icon: UploadCloud },
    ],
  },
  {
    group: "record",
    items: [
      { to: "/runs", label: "Runs", icon: Activity },
      { to: "/experiments", label: "Experiments", icon: Split },
      { to: "/evals", label: "Evals", icon: FlaskConical },
      { to: "/cost", label: "Cost", icon: DollarSign },
      // What the TOOL cost, beside what the compute cost. Different subjects,
      // same shape of question, and neither belongs under "work".
      { to: "/activity", label: "Activity", icon: Gauge },
    ],
  },
] as const;

/**
 * Left rail plus a status bar that is present on every page.
 *
 * Nodes, burn rate and live task count are the things worth knowing wherever
 * you are, so they are never a click away. They reuse the same queries the
 * Overview panels use -- TanStack Query dedupes them, so this costs nothing.
 */
export function Shell() {
  const path = useRouterState({ select: (s) => s.location.pathname });
  const pool = usePool();
  const jobs = useJobs(10);

  const nodes = pool.data?.current_dedicated_nodes ?? null;
  const live =
    jobs.data?.jobs.reduce(
      (total, job) =>
        total +
        job.tasks.filter((t) => ["running", "active", "preparing"].includes(state(t.state))).length,
      0,
    ) ?? null;

  return (
    <div className="flex min-h-full">
      <nav className="w-[168px] shrink-0 border-r border-[var(--border)] p-3">
        <div className="mb-4 px-2 font-mono text-[11px] tracking-widest text-[var(--fg-faint)]">
          POKER-SOLVER
        </div>
        {NAV.map(({ group, items }) => (
          <div key={group} className="mb-3">
            <div className="mb-1 px-2 text-[10px] tracking-widest text-[var(--fg-faint)] uppercase">
              {group}
            </div>
            <ul className="space-y-0.5">
              {items.map(({ to, label, icon: Icon }) => {
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
          </div>
        ))}
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
          {pool.data?.hourly_cost && <span>· {pool.data.hourly_cost}</span>}
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

const state = (raw: string | null) => (raw ?? "").split(".").pop()?.toLowerCase() ?? "";
