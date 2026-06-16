import { useNow } from "@/api/queries";
import type { NodePhase, NodeStatus, Pool, TaskRow } from "@/api/types";
import { Panel } from "@/components/Panel";
import { StatusBadge } from "@/components/StatusBadge";
import { Table, Td, Th } from "@/components/Table";
import { clock, count, duration, instant, since, span, taskLabel } from "@/lib/format";
import { type PoolShape, type Slot, elapsed, nodeLabel, poolShape } from "@/lib/pool";
import { cn } from "@/lib/utils";
import { Link } from "@tanstack/react-router";
import { useMemo } from "react";

/**
 * The page the console exists for: what is happening, and did anything die.
 *
 * ONE request for the whole screen, served stale-while-revalidate: a poll
 * never waits on a cloud sweep, and the age badge reads the payload's own `at`
 * -- the moment the server composed it -- rather than the moment this tab
 * fetched it, which after a stale serve would be a lie.
 *
 * Panels still fail INDEPENDENTLY: each part carries its own `error`, so an
 * expired `az login` greys the two Batch panels and leaves the rest.
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

  // Recomputed each render rather than ticked: the panel already re-renders on
  // the poll, and a second timer would redraw it between polls to move one
  // "2h 14m" by a minute.
  const now = Date.now();
  const shape = useMemo(
    () => poolShape(jobsData ?? undefined, poolData?.nodes),
    [jobsData, poolData],
  );
  // Batch knows which task holds a node; only the task log knows how far along
  // it is. Neither can draw the bar alone.
  const progress = useMemo(() => {
    const byTask = new Map<string, TaskRow>();
    for (const row of tasksData?.rows ?? []) {
      if (row.progress || row.eta_seconds != null) byTask.set(row.task_id, row);
    }
    return byTask;
  }, [tasksData]);

  const panel = {
    updatedAt: composedAt,
    loading: view.isLoading,
    onRefresh: () => view.refresh(),
    refreshing: view.isFetching,
  };

  return (
    <div className="space-y-3">
      <Panel title="Pool" staleAfterMs={30_000} error={pool?.error ?? null} {...panel}>
        {poolData && <PoolFacts pool={poolData} now={now} />}
      </Panel>

      {/* Batch and the task log answer DIFFERENT questions; they share a table
          because neither is sufficient alone: Batch says which node holds
          which task, the task log says how far along it is. */}
      <Panel
        title={`Nodes — ${phaseSummary(shape)}`}
        staleAfterMs={30_000}
        error={pool?.error ?? jobs?.error ?? null}
        empty={
          jobsData && poolData && shape.slots.length === 0 && shape.queue.length === 0
            ? "Idle — the pool has no nodes and nothing is queued."
            : null
        }
        {...panel}
      >
        {(shape.slots.length > 0 || shape.queue.length > 0) && (
          <Nodes shape={shape} now={now} progress={progress} />
        )}
      </Panel>

      <Panel
        title="Recent tasks"
        staleAfterMs={120_000}
        error={tasks?.error ?? null}
        empty={tasksData && tasksData.rows.length === 0 ? "No task records." : null}
        {...panel}
      >
        {tasksData && tasksData.rows.length > 0 && (
          <Table>
            <thead>
              <tr>
                <Th>task</Th>
                <Th>what</Th>
                <Th>cause</Th>
                <Th right>took</Th>
                <Th right>ended</Th>
              </tr>
            </thead>
            <tbody>
              {[...(tasksData?.rows ?? [])].reverse().map((row) => (
                <tr key={`${row.task_id}-${row.attempt}`}>
                  <Td mono title={row.task_id}>
                    <Link
                      to="/tasks/$taskId"
                      params={{ taskId: row.task_id }}
                      className="hover:underline"
                    >
                      {taskLabel(row.task_id)}
                    </Link>
                  </Td>
                  <Td className="text-[var(--fg-muted)]">{row.what || row.op || "—"}</Td>
                  <Td>
                    <StatusBadge state={row.cause} />
                  </Td>
                  <Td right className="text-[var(--fg-muted)]">
                    {span(row.started_at, row.ended_at, now)}
                  </Td>
                  <Td right className="text-[var(--fg-faint)]" title={row.ended_at ?? undefined}>
                    {row.ended_at ? since(row.ended_at) : "—"}
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

/** `6 busy · 1 booting` — only the phases that are present, busiest first. */
function phaseSummary(shape: PoolShape): string {
  const order: NodePhase[] = ["busy", "booting", "idle", "down", "leaving", "unknown"];
  const parts = order
    .filter((phase) => shape.byPhase[phase])
    .map((phase) => `${shape.byPhase[phase]} ${phase}`);
  if (shape.queue.length > 0) parts.push(`${shape.queue.length} queued`);
  return parts.join(" · ") || "none";
}

/**
 * The pool as facts, then what the autoscaler last decided about it.
 *
 * Two rows with a reason: the first is what IS (nodes, allocation, size,
 * burn), the second is what the formula WANTS and when it last said so. A pool
 * that will not grow is diagnosed by reading the two together, and the formula
 * variables are named rather than dumped -- `$TargetDedicatedNodes=7` is a
 * sentence in the formula's language, not the reader's.
 */
function PoolFacts({ pool, now }: { pool: Pool; now: number }) {
  const allocation = pool.allocation_state ?? "—";
  const allocationFor = elapsed(pool.allocation_since, now);
  return (
    <div className="divide-y divide-[var(--border)]/60">
      <div className="grid grid-cols-2 gap-x-6 gap-y-2 p-3 sm:grid-cols-4">
        <Stat
          label="nodes"
          value={`${count(pool.current_dedicated_nodes)} of ${count(
            pool.target_dedicated_nodes,
          )} wanted`}
        />
        {/* `resizing for 4m` is the fact that matters; `steady` needs no clock. */}
        <Stat
          label="allocation"
          value={
            allocation === "steady" || !allocationFor
              ? allocation
              : `${allocation} for ${allocationFor}`
          }
          title={pool.allocation_since ?? undefined}
        />
        <Stat label="vm size" value={pool.vm_size ?? "—"} mono />
        <Stat
          label="burn"
          value={pool.burn_per_hour != null ? `$${pool.burn_per_hour.toFixed(2)}/hr` : "—"}
          title={
            pool.hourly_cost
              ? `${pool.hourly_cost} × ${count(pool.current_dedicated_nodes)}`
              : undefined
          }
        />
        {pool.resize_errors.map((e) => (
          <p key={e.code} className="col-span-full font-mono text-[12px] text-red-400">
            {e.code}: {e.message}
          </p>
        ))}
      </div>
      <Autoscale pool={pool} now={now} />
    </div>
  );
}

/**
 * What the deployed formula last decided, as a sentence.
 *
 * Read off the pool's own last run, so it is what Batch actually acted on and
 * cannot be a stale copy of the formula. `error` is a FIELD, not a failed
 * request: Batch evaluated the formula and reported that it did not compute,
 * which is the answer to "why is the pool not growing" rather than the absence
 * of one.
 */
function Autoscale({ pool, now }: { pool: Pool; now: number }) {
  const run = pool.autoscale;
  if (!run) {
    return (
      <p className="px-3 py-2 text-[12px] text-[var(--fg-faint)]">
        autoscale — the pool has not evaluated its formula yet.
      </p>
    );
  }
  const v = run.variables;
  const wants = v.$TargetDedicatedNodes;
  const pending = v.pending;
  const ceiling = v.maxNodes;
  const evaluatedAt = instant(run.evaluated_at);
  const nextIn =
    evaluatedAt != null && run.interval_seconds != null
      ? Math.max(0, evaluatedAt + run.interval_seconds * 1000 - now) / 1000
      : null;
  const named = wants != null || pending != null || ceiling != null;
  return (
    <div className="px-3 py-2 text-[12px]">
      <span className="mr-3 text-[11px] text-[var(--fg-faint)] uppercase tracking-wider">
        autoscale
      </span>
      {named ? (
        <span className="tnum">
          wants <strong className="text-[var(--fg)]">{wants ?? "?"}</strong> nodes
          {pending != null && (
            <>
              {" "}
              for <strong className="text-[var(--fg)]">{pending}</strong> pending task
              {pending === "1" ? "" : "s"}
            </>
          )}
          {ceiling != null && <span className="text-[var(--fg-muted)]"> · ceiling {ceiling}</span>}
        </span>
      ) : (
        // A formula using other names still gets read, just not paraphrased.
        <span className="font-mono text-[11px] text-[var(--fg-muted)]">
          {Object.entries(v)
            .map(([name, value]) => `${name} = ${value}`)
            .join(" · ") || "evaluated to no variables"}
        </span>
      )}
      <span className="tnum ml-3 text-[var(--fg-faint)]" title={run.evaluated_at ?? undefined}>
        evaluated {run.evaluated_at ? since(run.evaluated_at) : "—"}
        {nextIn != null && ` · next in ~${duration(nextIn)}`}
      </span>
      {run.error && (
        <div className="mt-1 font-mono text-[12px] text-[#E0655C]">
          <p>{run.error.code}</p>
          {run.error.message && <p className="opacity-80">{run.error.message}</p>}
          {Object.entries(run.error.values).map(([name, value]) => (
            <p key={name} className="opacity-70">
              {name}: {value}
            </p>
          ))}
        </div>
      )}
    </div>
  );
}

/**
 * One row per node, then the queue beneath it.
 *
 * A table with headers rather than a row of unlabelled numbers: `1% · 5h 56m
 * left · 1h 02m · since 16:28` was four different measurements with nothing to
 * say which was which. Each is a column now, and a node holding no task says
 * what it IS doing instead -- `booting 2m`, `idle 4m`, `down` -- which is the
 * half of the picture a task list cannot show.
 */
function Nodes({
  shape,
  now,
  progress,
}: {
  shape: PoolShape;
  now: number;
  progress: Map<string, TaskRow>;
}) {
  return (
    <Table>
      <thead>
        <tr>
          <Th>node</Th>
          <Th>task</Th>
          <Th>progress</Th>
          <Th right>left</Th>
          <Th right>running for</Th>
          <Th right>started</Th>
        </tr>
      </thead>
      <tbody>
        {shape.slots.map((slot, index) => (
          <NodeRow
            key={slot.node?.id ?? slot.task?.task ?? index}
            slot={slot}
            now={now}
            row={slot.task ? progress.get(slot.task.task) : undefined}
          />
        ))}
        {shape.queue.length > 0 && (
          <tr>
            <td
              colSpan={6}
              className="border-b border-[var(--border)] px-3 py-1.5 text-[11px] text-[var(--fg-faint)] uppercase tracking-wider"
            >
              queue
              <span className="ml-3 normal-case tracking-normal">
                {shape.starved > 0 ? (
                  <span
                    className="text-amber-400"
                    title="more waiting than there are nodes idle or booting — these wait for the pool to grow, not for a task to end"
                  >
                    {shape.queue.length} waiting · {shape.starved} beyond capacity
                  </span>
                ) : (
                  `${shape.queue.length} waiting`
                )}
              </span>
            </td>
          </tr>
        )}
        {shape.queue.map((task, index) => (
          <tr key={`${task.job}/${task.task}`}>
            {/* Position, not a badge: "3rd in line" is the question, and a
                column of identical `queued` badges cannot answer it. */}
            <Td className="text-[var(--fg-faint)]">
              <span className="font-mono text-[11px]">#{index + 1}</span> in line
            </Td>
            <Td mono className="text-[var(--fg-muted)]" title={task.task}>
              {taskLabel(task.task)}
            </Td>
            <Td className="text-[var(--fg-faint)]">—</Td>
            <Td right className="text-[var(--fg-faint)]">
              —
            </Td>
            <Td right className="text-[var(--fg-faint)]">
              waiting {elapsed(task.created, now) || "—"}
            </Td>
            <Td right className="text-[var(--fg-faint)]" title={task.created ?? undefined}>
              {clock(task.created)}
            </Td>
          </tr>
        ))}
      </tbody>
    </Table>
  );
}

const DOT: Record<NodePhase, string> = {
  busy: "bg-emerald-500",
  booting: "bg-amber-400 motion-safe:animate-pulse",
  idle: "ring-1 ring-[var(--fg-faint)]/50",
  down: "bg-red-500",
  leaving: "bg-zinc-500",
  unknown: "bg-zinc-500",
};

function NodeRow({ slot, now, row }: { slot: Slot; now: number; row?: TaskRow }) {
  const { node, task } = slot;
  const phase: NodePhase = node?.phase ?? "busy";
  const started = task?.start_time ?? null;
  return (
    <tr className={cn(phase === "down" && "bg-red-500/[0.04]")}>
      <Td>
        <span className="flex items-center gap-2">
          <span className={cn("size-1.5 shrink-0 rounded-full", DOT[phase])} />
          {node ? (
            <NodeState node={node} now={now} />
          ) : (
            <span className="text-[var(--fg-faint)]">unlisted node</span>
          )}
        </span>
      </Td>
      <Td mono>
        {task ? (
          <Link
            to="/tasks/$taskId"
            params={{ taskId: task.task }}
            className="hover:underline"
            title={task.task}
          >
            {taskLabel(task.task)}
          </Link>
        ) : (
          <span className="text-[var(--fg-faint)]">—</span>
        )}
      </Td>
      <Td>
        <Bar row={row} />
      </Td>
      <Td right className="text-[var(--fg-muted)]">
        {row?.eta_seconds != null ? duration(row.eta_seconds) : "—"}
      </Td>
      <Td right className="text-[var(--fg-muted)]">
        {task ? elapsed(started, now) || "—" : "—"}
      </Td>
      <Td right className="text-[var(--fg-faint)]" title={started ?? undefined}>
        {task ? clock(started) : "—"}
      </Td>
    </tr>
  );
}

/**
 * The node itself: its tail id, and -- when it holds no task -- what it is
 * doing and for how long, which is the whole answer to "why is my queued task
 * not running". A busy node needs no clock of its own; its task has one.
 */
function NodeState({ node, now }: { node: NodeStatus; now: number }) {
  const how = elapsed(node.since, now);
  const word =
    node.phase === "busy" ? "" : node.phase === "idle" ? "idle" : (node.state ?? node.phase);
  const problem =
    node.errors[0] ?? (node.start_task_state === "failed" ? "start task failed" : null);
  return (
    <span className="flex min-w-0 flex-col">
      <span className="flex items-baseline gap-2">
        <span className="font-mono text-[12px]" title={node.id}>
          {nodeLabel(node.id)}
        </span>
        {word && (
          <span
            className={cn(
              "text-[11px]",
              node.phase === "down" ? "text-red-400" : "text-[var(--fg-faint)]",
            )}
            title={node.since ?? undefined}
          >
            {word}
            {how && ` ${how}`}
          </span>
        )}
      </span>
      {problem && (
        <span
          className="truncate font-mono text-[11px] text-red-400"
          title={node.errors.join("\n")}
        >
          {problem}
        </span>
      )}
    </span>
  );
}

/**
 * How far along a running task is, when its kind can say.
 *
 * Absent for a kind that cannot answer -- scoring one rung is opaque from
 * outside it -- and absent is drawn as a dash rather than as an empty bar,
 * because an empty bar reads as a task that is stuck rather than one that
 * simply does not report. The unit is in the tooltip; the `left` column beside
 * it carries the estimate.
 */
function Bar({ row }: { row?: TaskRow }) {
  const of = row?.progress;
  if (!of || of.total <= 0) return <span className="text-[var(--fg-faint)]">—</span>;
  const fraction = Math.max(0, Math.min(1, of.done / of.total));
  return (
    <span
      className="flex items-center gap-1.5"
      title={`${count(of.done)} / ${count(of.total)} ${of.unit}`}
    >
      <span className="h-1 w-16 overflow-hidden rounded-full bg-[var(--border)]">
        <span
          className="block h-full rounded-full bg-emerald-500/70"
          style={{ width: `${fraction * 100}%` }}
        />
      </span>
      <span className="tnum text-[11px]">{Math.round(fraction * 100)}%</span>
    </span>
  );
}

function Stat({
  label,
  value,
  mono,
  title,
}: {
  label: string;
  value: string;
  mono?: boolean;
  title?: string;
}) {
  return (
    <div title={title}>
      <div className="text-[11px] text-[var(--fg-faint)] uppercase tracking-wider">{label}</div>
      <div className={mono ? "font-mono" : "tnum"}>{value}</div>
    </div>
  );
}
