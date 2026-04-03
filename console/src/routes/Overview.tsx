import { useJobs, useLegs, usePool } from "@/api/queries";
import { Panel } from "@/components/Panel";
import {
  StatusBadge,
  exitMeaning,
  shortState,
  taskOutcome,
  taskTone,
} from "@/components/StatusBadge";
import { Table, Td, Th } from "@/components/Table";
import { errorOf } from "@/lib/error";
import { count, since } from "@/lib/format";
import { Link } from "@tanstack/react-router";

/** The page the console exists for: what is happening, and did anything die. */
export function Overview() {
  const pool = usePool();
  const jobs = useJobs(10);
  const legs = useLegs(10);

  return (
    <div className="space-y-3">
      <Panel
        title="Pool"
        updatedAt={pool.dataUpdatedAt}
        staleAfterMs={30_000}
        error={errorOf(pool.error)}
        loading={pool.isLoading}
        onRefresh={() => pool.refetch()}
        refreshing={pool.isFetching}
      >
        {pool.data && (
          <div className="grid grid-cols-2 gap-x-6 gap-y-2 p-3 sm:grid-cols-4">
            <Stat label="pool" value={pool.data.pool_id} mono />
            <Stat
              label="nodes"
              value={`${count(pool.data.current_dedicated_nodes)} / ${count(
                pool.data.target_dedicated_nodes,
              )}`}
            />
            <Stat label="allocation" value={pool.data.allocation_state?.split(".").pop() ?? "—"} />
            <Stat label="vm size" value={pool.data.vm_size ?? "—"} mono />
            {pool.data.resize_errors.map((e) => (
              <p key={e.code} className="col-span-full font-mono text-[12px] text-red-400">
                {e.code}: {e.message}
              </p>
            ))}
          </div>
        )}
      </Panel>

      {/* Batch and the leg log answer DIFFERENT questions; the panels sit
          together because neither is sufficient alone. */}
      <Panel
        title="Batch"
        updatedAt={jobs.dataUpdatedAt}
        staleAfterMs={30_000}
        error={errorOf(jobs.error)}
        loading={jobs.isLoading}
        empty={jobs.data && jobs.data.jobs.length === 0 ? "Nothing running." : null}
        onRefresh={() => jobs.refetch()}
        refreshing={jobs.isFetching}
      >
        {jobs.data && jobs.data.jobs.length > 0 && (
          <Table>
            <thead>
              <tr>
                <Th>task</Th>
                <Th>job</Th>
                <Th>state</Th>
                <Th right>exit</Th>
              </tr>
            </thead>
            <tbody>
              {jobs.data.jobs.flatMap((job) =>
                job.tasks.map((task) => {
                  const state = shortState(task.state);
                  const meaning = exitMeaning(task.exit_code);
                  return (
                    <tr key={`${job.job}/${task.task}`}>
                      <Td mono>{task.task}</Td>
                      <Td mono className="text-[var(--fg-muted)]">
                        {job.job}
                      </Td>
                      <Td>
                        {/* Coloured on state AND exit code. Batch's `completed`
                            means finished, not succeeded — badging it green made
                            a cancelled task look like a clean one. */}
                        <StatusBadge
                          state={taskOutcome(task.state, task.exit_code)}
                          tone={taskTone(task.state, task.exit_code)}
                          title={`Batch reports state "${state}"`}
                        />
                      </Td>
                      <Td right title={meaning ?? undefined}>
                        {task.exit_code ?? "—"}
                        {meaning && task.exit_code !== 0 && (
                          <span className="ml-1 text-[var(--fg-faint)]">ⓘ</span>
                        )}
                      </Td>
                    </tr>
                  );
                }),
              )}
            </tbody>
          </Table>
        )}
      </Panel>

      <Panel
        title="Legs"
        updatedAt={legs.dataUpdatedAt}
        staleAfterMs={120_000}
        error={errorOf(legs.error)}
        loading={legs.isLoading}
        empty={legs.data && legs.data.rows.length === 0 ? "No leg records." : null}
        onRefresh={() => legs.refetch()}
        refreshing={legs.isFetching}
      >
        {legs.data && legs.data.rows.length > 0 && (
          <Table>
            <thead>
              <tr>
                <Th>task</Th>
                <Th>op</Th>
                <Th>cause</Th>
                <Th right>exit</Th>
                <Th right>ended</Th>
              </tr>
            </thead>
            <tbody>
              {[...legs.data.rows].reverse().map((row) => (
                <tr key={`${row.task_id}-${row.attempt}`}>
                  <Td mono>
                    <Link
                      to="/legs/$taskId"
                      params={{ taskId: row.task_id }}
                      className="hover:underline"
                    >
                      {row.task_id}
                    </Link>
                  </Td>
                  <Td className="text-[var(--fg-muted)]">{row.op ?? "—"}</Td>
                  <Td>
                    <StatusBadge state={row.cause} />
                  </Td>
                  <Td right>{row.exit_code ?? "—"}</Td>
                  <Td right className="text-[var(--fg-faint)]">
                    {since(row.ended_at)}
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        )}
      </Panel>
    </div>
  );
}

function Stat({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <div className="text-[11px] text-[var(--fg-faint)] uppercase tracking-wider">{label}</div>
      <div className={mono ? "font-mono" : "tnum"}>{value}</div>
    </div>
  );
}
