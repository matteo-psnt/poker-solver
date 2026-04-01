import { useLegs } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { StatusBadge, toneFor } from "@/components/StatusBadge";
import { Table, Td, Th } from "@/components/Table";
import { errorOf } from "@/lib/error";
import { since } from "@/lib/format";
import { cn } from "@/lib/utils";
import { getRouteApi, useNavigate } from "@tanstack/react-router";
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
            {c}
          </Chip>
        ))}
      </div>
      {rows.length > 0 && (
        <Table>
          <thead>
            <tr>
              <Th>task</Th>
              <Th right>#</Th>
              <Th>op</Th>
              <Th>run</Th>
              <Th>cause</Th>
              <Th right>exit</Th>
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
                  <Td mono>{row.task_id}</Td>
                  <Td right className="text-[var(--fg-faint)]">
                    {row.attempt ?? "—"}
                  </Td>
                  <Td className="text-[var(--fg-muted)]">{row.op ?? "—"}</Td>
                  <Td mono className="text-[var(--fg-muted)]">
                    {row.run_id || "—"}
                  </Td>
                  <Td>
                    <StatusBadge state={row.cause} />
                  </Td>
                  <Td right>{row.exit_code ?? "—"}</Td>
                  <Td right className="text-[var(--fg-faint)]">
                    {since(row.ended_at)}
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
