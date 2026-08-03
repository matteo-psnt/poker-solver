import { useRuns } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { StatusBadge } from "@/components/StatusBadge";
import { Table, Td, Th } from "@/components/Table";
import { errorOf } from "@/lib/error";
import { count } from "@/lib/format";
import { Link } from "@tanstack/react-router";

export function Runs() {
  const runs = useRuns();
  return (
    <Panel
      title="Runs"
      updatedAt={runs.dataUpdatedAt}
      staleAfterMs={120_000}
      error={errorOf(runs.error)}
      loading={runs.isLoading}
      empty={runs.data && runs.data.runs.length === 0 ? "No published runs." : null}
      onRefresh={() => runs.refetch()}
      refreshing={runs.isFetching}
    >
      {runs.data && runs.data.runs.length > 0 && (
        <Table>
          <thead>
            <tr>
              <Th>run</Th>
              <Th>config</Th>
              <Th right>iterations</Th>
              <Th right>infosets</Th>
              <Th>status</Th>
            </tr>
          </thead>
          <tbody>
            {runs.data.runs.map((run) => (
              <tr key={run.name}>
                <Td mono>
                  <Link to="/runs/$runId" params={{ runId: run.name }} className="hover:underline">
                    {run.name}
                  </Link>
                  {!run.loadable && run.blocker && (
                    <span className="ml-2 text-[11px] text-amber-400">⚠ {run.blocker}</span>
                  )}
                </Td>
                <Td className="text-[var(--fg-muted)]">{run.config_name ?? "—"}</Td>
                <Td right>{count(run.iterations)}</Td>
                <Td right className="text-[var(--fg-muted)]">
                  {count(run.num_infosets)}
                </Td>
                <Td>
                  <StatusBadge state={run.status} />
                </Td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}
    </Panel>
  );
}
