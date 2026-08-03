import { useEvals } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { Table, Td, Th } from "@/components/Table";
import { errorOf } from "@/lib/error";
import { mbb } from "@/lib/format";

const str = (value: unknown) => (value == null ? "—" : String(value));
const num = (value: unknown) => (typeof value === "number" ? value : null);

export function Evals() {
  const evals = useEvals();
  return (
    <Panel
      title="Evaluations"
      updatedAt={evals.dataUpdatedAt}
      staleAfterMs={120_000}
      error={errorOf(evals.error)}
      loading={evals.isLoading}
      empty={
        evals.data && evals.data.rows.length === 0
          ? "No evaluations in the current layout. Older ones are legacy files on the share."
          : null
      }
      onRefresh={() => evals.refetch()}
      refreshing={evals.isFetching}
    >
      {evals.data && evals.data.rows.length > 0 && (
        <Table>
          <thead>
            <tr>
              <Th>run</Th>
              <Th>commit</Th>
              <Th>scorer</Th>
              <Th>opponent</Th>
              <Th right>seed</Th>
              <Th right>hands</Th>
              <Th right>mbb/g</Th>
            </tr>
          </thead>
          <tbody>
            {evals.data.rows.map((row, index) => (
              <tr key={`${row.run_id}-${index}`}>
                <Td mono>{row.run_id ?? "—"}</Td>
                <Td mono className="text-[var(--fg-faint)]">
                  {row.eval_git_commit?.slice(0, 7) ?? "—"}
                </Td>
                <Td>{str(row.knobs.scorer)}</Td>
                <Td>{str(row.knobs.opponent)}</Td>
                <Td right className="text-[var(--fg-muted)]">
                  {str(row.knobs.base_seed)}
                </Td>
                <Td right>{str(row.results.num_hands)}</Td>
                <Td right>
                  {mbb(num(row.results.exploitability_mbb), num(row.results.std_error_mbb))}
                </Td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}
    </Panel>
  );
}
