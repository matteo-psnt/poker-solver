import { Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { useNow } from "@/api/queries";
import type { NodePhase, PoolView, TaskRow } from "@/api/types";
import { Panel } from "@/components/Panel";
import { StatusBadge } from "@/components/StatusBadge";
import { Table, Td, Th } from "@/components/Table";
import { useClock } from "@/lib/clock";
import { clock, count, duration, instant, runLabel, since, taskLabel } from "@/lib/format";
import { groupFinished, medianSeconds } from "@/lib/groups";
import { elapsed, nodeLabel, type PoolShape, poolShape, type Slot } from "@/lib/pool";
import { cn } from "@/lib/utils";

/** How many grouped rows the finished list shows. */
const GROUPS = 16;

/**
 * The page the console exists for: what is running, and what just ended.
 *
 * Three panels, in the order a reader asks: the WORK (one row per task, named
 * by its run and what it does, not by its node), what ENDED (grouped, so a
 * 13-seed gate is one row), and the MACHINES, one line per pool.
 *
 * One request for the whole screen, served stale-while-revalidate; panels still
 * fail independently because each part carries its own `error`.
 */
export function Overview() {
  const view = useNow();
  const parts = view.data?.parts;
  const pool = parts?.pool;
  const poolData = pool?.payload;
  const jobs = parts?.jobs;
  const jobsData = jobs?.payload;
  const tasks = parts?.tasks;
  const tasksData = tasks?.payload;
  const composedAt = instant(view.data?.at) ?? view.dataUpdatedAt;
  const now = useClock(15_000);
  const [onlyTrouble, setOnlyTrouble] = useState(false);

  const shape = useMemo(
    () =>
      poolShape(
        jobsData ?? undefined,
        poolData?.pools.flatMap((p) => p.nodes),
      ),
    [jobsData, poolData],
  );
  // Batch knows which node holds a task; only the task log knows its run and
  // how far along it is. The newest attempt per task wins.
  const rowByTask = useMemo(() => {
    const byTask = new Map<string, TaskRow>();
    for (const row of tasksData?.rows ?? []) byTask.set(row.task_id, row);
    return byTask;
  }, [tasksData]);
  const poolOfNode = useMemo(() => {
    const byNode = new Map<string, string>();
    for (const p of poolData?.pools ?? []) for (const n of p.nodes) byNode.set(n.id, p.pool_id);
    return byNode;
  }, [poolData]);
  const groups = useMemo(() => {
    const all = groupFinished(tasksData?.rows ?? []);
    return onlyTrouble ? all.filter((g) => g.rows[0]?.cause !== "completed") : all;
  }, [tasksData, onlyTrouble]);

  const panel = {
    updatedAt: composedAt,
    loading: view.isLoading,
    onRefresh: () => view.refresh(),
    refreshing: view.isFetching,
  };
  const working = shape.slots.filter((slot) => slot.task);

  return (
    <div className="space-y-3">
      <Panel
        title={`Running — ${workSummary(shape)}`}
        staleAfterMs={30_000}
        error={pool?.error ?? jobs?.error ?? null}
        empty={
          jobsData && poolData && working.length === 0 && shape.queue.length === 0
            ? "Nothing running and nothing queued."
            : null
        }
        {...panel}
      >
        {(working.length > 0 || shape.queue.length > 0) && (
          <Table>
            <thead>
              <tr>
                <Th>run</Th>
                <Th>work</Th>
                <Th right>attempt</Th>
                <Th>progress</Th>
                <Th right>left</Th>
                <Th right>running for</Th>
                <Th>machine</Th>
              </tr>
            </thead>
            <tbody>
              {working.map((slot) => (
                <WorkRow
                  key={slot.task?.task}
                  slot={slot}
                  row={slot.task ? rowByTask.get(slot.task.task) : undefined}
                  pool={slot.node ? poolOfNode.get(slot.node.id) : undefined}
                  now={now}
                />
              ))}
              {shape.queue.map((task, index) => {
                const row = rowByTask.get(task.task);
                return (
                  <tr key={`${task.job}/${task.task}`}>
                    <Td>
                      <RunLink runId={row?.run_id} />
                    </Td>
                    <Td className="text-[var(--fg-muted)]">{row?.what || taskLabel(task.task)}</Td>
                    <Td right className="text-[var(--fg-faint)]">
                      —
                    </Td>
                    <Td className="text-[var(--fg-faint)]">
                      queued, #{index + 1} in line
                      {index >= shape.queue.length - shape.starved && (
                        <span className="ml-2 text-amber-400">waits for the pool to grow</span>
                      )}
                    </Td>
                    <Td right className="text-[var(--fg-faint)]">
                      —
                    </Td>
                    <Td right className="text-[var(--fg-faint)]">
                      waiting {elapsed(task.created, now) || "—"}
                    </Td>
                    <Td className="text-[var(--fg-faint)]">—</Td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        )}
      </Panel>

      <Panel
        title="Finished recently"
        aside={
          <span className="flex gap-1 text-[11px]">
            {(
              [
                ["all", false],
                ["not completed", true],
              ] as const
            ).map(([label, value]) => (
              <button
                key={label}
                type="button"
                onClick={() => setOnlyTrouble(value)}
                className={cn(
                  "rounded px-2 py-0.5",
                  onlyTrouble === value
                    ? "bg-white/[0.08] text-[var(--fg)]"
                    : "text-[var(--fg-muted)] hover:text-[var(--fg)]",
                )}
              >
                {label}
              </button>
            ))}
          </span>
        }
        staleAfterMs={120_000}
        error={tasks?.error ?? null}
        empty={
          tasksData && groups.length === 0
            ? onlyTrouble
              ? "Everything recent completed."
              : "No task records."
            : null
        }
        {...panel}
      >
        {groups.length > 0 && (
          <Table>
            <thead>
              <tr>
                <Th>ended</Th>
                <Th>run</Th>
                <Th>work</Th>
                <Th>outcome</Th>
                <Th right>each took</Th>
              </tr>
            </thead>
            <tbody>
              {groups.slice(0, GROUPS).map((group) => {
                const first = group.rows[0] as TaskRow;
                const many = group.rows.length > 1;
                const took = medianSeconds(group.rows);
                return (
                  <tr key={`${first.task_id}-${first.attempt}`}>
                    <Td title={first.ended_at ?? undefined}>
                      <span className="text-[var(--fg)]">
                        {many
                          ? `${clock(new Date(group.first).toISOString())}–${clock(new Date(group.last).toISOString())}`
                          : clock(first.ended_at)}
                      </span>
                      <span className="ml-2 text-[var(--fg-faint)]">
                        {since(new Date(group.last).toISOString())}
                      </span>
                    </Td>
                    <Td>
                      <RunLink runId={first.run_id} />
                    </Td>
                    <Td>
                      <Link
                        to="/tasks/$taskId"
                        params={{ taskId: first.task_id }}
                        className="hover:underline"
                        title={first.task_id}
                      >
                        {first.what || first.op || taskLabel(first.task_id)}
                      </Link>
                      {many && (
                        <span className="ml-2 rounded bg-white/[0.07] px-1.5 py-px font-mono text-[11px] text-[var(--fg-muted)]">
                          ×{group.rows.length}
                        </span>
                      )}
                    </Td>
                    <Td>
                      <StatusBadge state={first.cause} />
                    </Td>
                    <Td right className="text-[var(--fg-muted)]">
                      {took == null ? "—" : `${many ? "median " : ""}${duration(took)}`}
                    </Td>
                  </tr>
                );
              })}
            </tbody>
          </Table>
        )}
      </Panel>

      <Panel
        title={poolsTitle(poolData?.pools)}
        staleAfterMs={30_000}
        error={pool?.error ?? null}
        {...panel}
      >
        {poolData?.pools.map((p) => (
          <PoolLine key={p.pool_id} pool={p} now={now} />
        ))}
      </Panel>
    </div>
  );
}

/** `2 running · 1 queued` — what the work is doing, not what the nodes are. */
function workSummary(shape: PoolShape): string {
  const running = shape.slots.filter((s) => s.task).length;
  const parts = [`${running} running`];
  if (shape.queue.length > 0) parts.push(`${shape.queue.length} queued`);
  return parts.join(" · ");
}

/** A run named the way the run list names it, linking to its page; a dash when the task has no run. */
function RunLink({ runId }: { runId: string | null | undefined }) {
  if (!runId) return <span className="text-[var(--fg-faint)]">—</span>;
  return (
    <Link to="/runs/$runId" params={{ runId }} className="font-mono hover:underline" title={runId}>
      {runLabel(runId)}
    </Link>
  );
}

function WorkRow({
  slot,
  row,
  pool,
  now,
}: {
  slot: Slot;
  row?: TaskRow;
  pool?: string;
  now: number;
}) {
  const task = slot.task;
  if (!task) return null;
  const started = task.start_time ?? null;
  return (
    <tr>
      <Td>
        <RunLink runId={row?.run_id} />
      </Td>
      <Td>
        <Link
          to="/tasks/$taskId"
          params={{ taskId: task.task }}
          className="hover:underline"
          title={task.task}
        >
          {row?.what || taskLabel(task.task)}
        </Link>
      </Td>
      <Td right className="text-[var(--fg-muted)]">
        {row?.attempt ?? "—"}
      </Td>
      <Td>
        <Progress row={row} />
      </Td>
      <Td right className="text-[var(--fg-muted)]">
        {row?.eta_seconds != null ? duration(row.eta_seconds) : "—"}
      </Td>
      <Td right className="text-[var(--fg-muted)]" title={started ?? undefined}>
        {elapsed(started, now) || "—"}
        {started && <span className="ml-2 text-[var(--fg-faint)]">since {clock(started)}</span>}
      </Td>
      <Td className="text-[var(--fg-muted)]" title={slot.node?.id}>
        {pool ?? "—"}
        {slot.node && (
          <span className="ml-2 font-mono text-[11px] text-[var(--fg-faint)]">
            {nodeLabel(slot.node.id)}
          </span>
        )}
      </Td>
    </tr>
  );
}

/**
 * How far along, and how much of that THIS attempt did.
 *
 * A resumed task starts part-way: the faint segment is where the attempt picked
 * up (`progress.base`), the bright one what it has added since. Absent for a
 * kind that cannot report, drawn as a dash rather than an empty bar.
 */
function Progress({ row }: { row?: TaskRow }) {
  const of = row?.progress;
  if (!of || of.total <= 0) return <span className="text-[var(--fg-faint)]">—</span>;
  const clamp = (v: number) => Math.max(0, Math.min(1, v / of.total));
  const done = clamp(of.done);
  const base = Math.min(done, clamp(of.base ?? 0));
  return (
    <span
      className="flex items-center gap-2"
      title={`${count(of.done)} / ${count(of.total)} ${of.unit}${of.base ? `, this attempt from ${count(of.base)}` : ""}`}
    >
      <span className="relative h-1.5 w-28 overflow-hidden rounded-full bg-[var(--border)]">
        <span
          className="absolute inset-y-0 left-0 bg-emerald-500/30"
          style={{ width: `${base * 100}%` }}
        />
        <span
          className="absolute inset-y-0 bg-emerald-500/80"
          style={{ left: `${base * 100}%`, width: `${(done - base) * 100}%` }}
        />
      </span>
      <span className="tnum text-[11px]">
        {count(of.done)} / {count(of.total)}
      </span>
    </span>
  );
}

/** `Pools` — or what is wrong with one, which is the only reason to look. */
function poolsTitle(pools: PoolView[] | undefined): string {
  const stuck = (pools ?? []).filter((p) => p.resize_errors.length > 0 || p.autoscale?.error);
  if (stuck.length === 0) return "Machines";
  return `Machines — ${stuck.map((p) => p.pool_id).join(", ")} cannot grow`;
}

const DOT: Record<NodePhase, string> = {
  busy: "bg-emerald-500",
  booting: "bg-amber-400 motion-safe:animate-pulse",
  idle: "ring-1 ring-[var(--fg-faint)]/50",
  down: "bg-red-500",
  leaving: "bg-zinc-500",
  unknown: "bg-zinc-500",
};

/**
 * One line per pool: its name and machine, each node as a dot with its state,
 * capacity and cost. A sleeping pool is dimmed and says so. Resize and
 * autoscale errors still print in full, because they are why a pool will not
 * grow.
 */
function PoolLine({ pool, now }: { pool: PoolView; now: number }) {
  const asleep = !pool.current_dedicated_nodes && !pool.target_dedicated_nodes;
  const error = pool.autoscale?.error;
  return (
    <div className="border-b border-[var(--border)]/60 px-3 py-2 last:border-b-0">
      <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
        <span className="w-[190px] shrink-0">
          <span
            className={cn(
              "font-mono text-[12px]",
              asleep ? "text-[var(--fg-muted)]" : "font-medium text-[var(--fg)]",
            )}
          >
            {pool.pool_id}
          </span>
          {pool.vm_size && (
            <span className="ml-2 font-mono text-[10px] text-[var(--fg-faint)]">
              {pool.vm_size.replace(/^standard_/i, "")}
            </span>
          )}
        </span>
        <span className="flex min-w-0 flex-1 flex-wrap items-center gap-3 text-[12px]">
          {asleep ? (
            <span className="text-[var(--fg-faint)]">asleep</span>
          ) : (
            pool.nodes.map((node) => (
              <span key={node.id} className="flex items-center gap-1.5" title={node.id}>
                <span className={cn("size-1.5 rounded-full", DOT[node.phase])} />
                <span className="text-[var(--fg-muted)]">
                  {node.phase}
                  {node.phase !== "busy" &&
                    elapsed(node.since, now) &&
                    ` ${elapsed(node.since, now)}`}
                </span>
              </span>
            ))
          )}
          {pool.target_dedicated_nodes !== pool.current_dedicated_nodes && (
            <span className="text-amber-400">growing to {count(pool.target_dedicated_nodes)}</span>
          )}
        </span>
        <span className="tnum w-[130px] text-right text-[12px] text-[var(--fg-muted)]">
          {pool.vcpus ?? 0}
          {pool.max_vcpus != null && ` / ${pool.max_vcpus}`} vCPU
        </span>
        <span className="tnum w-[70px] text-right text-[12px] text-[var(--fg)]">
          {pool.burn_per_hour ? `$${pool.burn_per_hour.toFixed(2)}/hr` : "—"}
        </span>
      </div>
      {pool.resize_errors.map((e, i) => (
        <p key={i} className="mt-1 font-mono text-[12px] text-[#E0655C]">
          {e.code}: {e.message}
        </p>
      ))}
      {error && (
        <p className="mt-1 font-mono text-[12px] text-[#E0655C]">
          autoscale: {error.code}
          {error.message && ` — ${error.message}`}
        </p>
      )}
    </div>
  );
}
