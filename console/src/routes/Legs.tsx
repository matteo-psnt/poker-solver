import { useLegs } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { StatusBadge, displayName, toneFor } from "@/components/StatusBadge";
import { Table, Td, Th } from "@/components/Table";
import { errorOf } from "@/lib/error";
import { clock, legLabel, runLabel, since, span } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Link, getRouteApi, useNavigate } from "@tanstack/react-router";
import { useMemo } from "react";

const route = getRouteApi("/legs");

/**
 * The densest view in the console, and the one opened when something died.
 *
 * Filters live in the URL rather than component state: a filtered view is then
 * a link, which is what someone actually wants to share when asking why a leg
 * failed.
 */
export function Legs() {
  const { cause } = route.useSearch();
  const navigate = useNavigate({ from: "/legs" });
  const legs = useLegs(0);
  // One `now` for the whole table, so every open-ended duration in a render is
  // measured against the same instant rather than drifting down the rows.
  const now = Date.now();

  const causes = useMemo(() => {
    const seen = new Set<string>();
    for (const row of legs.data?.rows ?? []) if (row.cause) seen.add(row.cause);
    return [...seen].sort();
  }, [legs.data]);

  const rows = useMemo(() => {
    const all = [...(legs.data?.rows ?? [])].reverse();
    return cause ? all.filter((r) => r.cause === cause) : all;
  }, [legs.data, cause]);

  return (
    <Panel
      title={`Legs${cause ? ` · ${cause}` : ""}`}
      updatedAt={legs.dataUpdatedAt}
      staleAfterMs={120_000}
      error={errorOf(legs.error)}
      loading={legs.isLoading}
      empty={legs.data && rows.length === 0 ? "No legs match." : null}
      onRefresh={() => legs.refetch()}
      refreshing={legs.isFetching}
    >
      <div className="flex flex-wrap gap-1.5 border-b border-[var(--border)] px-3 py-2">
        <Chip active={!cause} onClick={() => navigate({ search: {} })}>
          all
        </Chip>
        {causes.map((c) => (
          <Chip key={c} active={cause === c} onClick={() => navigate({ search: { cause: c } })}>
            {displayName(c)}
          </Chip>
        ))}
      </div>
      {rows.length > 0 && (
        <Table>
          <thead>
            <tr>
              <Th>task</Th>
              <Th right>#</Th>
              <Th>what</Th>
              <Th>run</Th>
              <Th>cause</Th>
              <Th right>started</Th>
              <Th right>took</Th>
              <Th right>ended</Th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => {
              const tone = toneFor(row.cause);
              return (
                <tr
                  key={`${row.task_id}-${row.attempt}`}
                  // The ROW is the unit being scanned, so a bad one is tinted
                  // rather than only badged.
                  className={cn(
                    tone === "bad" && "bg-red-500/[0.04]",
                    tone === "warn" && "bg-amber-500/[0.04]",
                  )}
                >
                  <Td mono>
                    <Link
                      to="/legs/$taskId"
                      params={{ taskId: row.task_id }}
                      title={row.task_id}
                      className="hover:underline"
                    >
                      {legLabel(row.task_id)}
                    </Link>
                  </Td>
                  <Td right className="text-[var(--fg-faint)]">
                    {row.attempt ?? "—"}
                  </Td>
                  <Td className="text-[var(--fg-muted)]">{row.what || row.op || "—"}</Td>
                  {/* Not every leg has a run: a `vector-sweep` is a measurement
                      that produces none, so this is blank rather than broken. */}
                  <Td mono className="text-[var(--fg-muted)]">
                    {row.run_id ? (
                      <Link
                        to="/runs/$runId"
                        params={{ runId: row.run_id }}
                        title={row.run_id}
                        className="hover:underline"
                      >
                        {runLabel(row.run_id)}
                      </Link>
                    ) : (
                      "—"
                    )}
                  </Td>
                  <Td>
                    {/* Tone from the WIRE value, label from the display name:
                        `toneFor` keys off what the leg log recorded, so passing
                        it a renamed word would silently mute every badge. */}
                    {/* The exit code is IN the badge now, not beside it. The
                        number was only ever read through its meaning — 124 is
                        the guard's deadline, 137 the OOM killer, -9 a
                        cancellation — and `displayName` already says that in
                        words. The raw code stays in the tooltip for the rare
                        case where an unmapped one turns up. */}
                    <StatusBadge
                      state={displayName(row.cause)}
                      tone={tone}
                      title={`recorded as "${row.cause}"${
                        row.exit_code == null ? "" : `, exit ${row.exit_code}`
                      }`}
                    />
                  </Td>
                  <Td right className="text-[var(--fg-faint)]" title={row.started_at ?? undefined}>
                    {clock(row.started_at)}
                  </Td>
                  {/* Open-ended for a leg still running, so "running 2h" and
                      "took 2h" stay distinguishable from "unknown". */}
                  <Td right className="tnum text-[var(--fg-muted)]">
                    {span(row.started_at, row.ended_at, now)}
                  </Td>
                  <Td right className="text-[var(--fg-faint)]" title={row.ended_at ?? undefined}>
                    {row.ended_at ? since(row.ended_at) : "—"}
                  </Td>
                </tr>
              );
            })}
          </tbody>
        </Table>
      )}
    </Panel>
  );
}

function Chip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded px-2 py-0.5 font-mono text-[11px] ring-1 ring-inset",
        active
          ? "bg-white/10 text-[var(--fg)] ring-white/20"
          : "text-[var(--fg-muted)] ring-[var(--border)] hover:text-[var(--fg)]",
      )}
    >
      {children}
    </button>
  );
}
