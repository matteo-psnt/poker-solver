import { useJobs, usePool, useTasks } from "@/api/queries";
import type { TaskRow } from "@/api/schemas";
import { Panel } from "@/components/Panel";
import { StatusBadge, exitMeaning, taskOutcome, taskTone } from "@/components/StatusBadge";
import { Table, Td, Th } from "@/components/Table";
import { errorOf } from "@/lib/error";
import { clock, count, duration, since, span, taskLabel } from "@/lib/format";
import { type PoolShape, type Task, elapsed, poolShape } from "@/lib/pool";
import { cn } from "@/lib/utils";
import { Link } from "@tanstack/react-router";
import { useMemo } from "react";

/** The page the console exists for: what is happening, and did anything die. */
export function Overview() {
  const pool = usePool();
  const jobs = useJobs(10);
  const tasks = useTasks(10);

  // Recomputed each render rather than ticked: the panel already re-renders on
  // the 15s poll, and a second timer would redraw it between polls to move one
  // "2h 14m" by a minute.
  const now = Date.now();
  const shape = useMemo(
    () => poolShape(jobs.data, pool.data?.target_dedicated_nodes),
    [jobs.data, pool.data],
  );
  const busy = shape.slots.filter((slot) => slot.task !== null).length;
  // Batch knows which task holds a node; only the task log knows how far along
  // it is. Neither can draw the bar alone.
  const progress = useMemo(() => {
    const byTask = new Map<string, TaskRow>();
    for (const row of tasks.data?.rows ?? []) {
      if (row.progress || row.eta_seconds != null) byTask.set(row.task_id, row);
    }
    return byTask;
  }, [tasks.data]);

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
            {/* ALLOCATED against wanted -- machines that exist. The panel below
                counts how many of them hold a task, which is a different fact
                that used to be written the same way and read as a contradiction. */}
            <Stat
              label="nodes"
              value={`${count(pool.data.current_dedicated_nodes)} of ${count(
                pool.data.target_dedicated_nodes,
              )} wanted`}
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

      {/* Batch and the task log answer DIFFERENT questions; the panels sit
          together because neither is sufficient alone. */}
      <Panel
        title={`Tasks — ${busy} running${
          shape.queue.length > 0 ? `, ${shape.queue.length} queued` : ""
        }`}
        updatedAt={jobs.dataUpdatedAt}
        staleAfterMs={30_000}
        error={errorOf(jobs.error)}
        loading={jobs.isLoading}
        empty={jobs.data && shape.slots.length === 0 && shape.queue.length === 0 ? "Idle." : null}
        onRefresh={() => jobs.refetch()}
        refreshing={jobs.isFetching}
      >
        {jobs.data && (shape.slots.length > 0 || shape.queue.length > 0) && (
          <>
            <Pipeline shape={shape} busy={busy} now={now} progress={progress} />
            <History history={shape.history} />
          </>
        )}
      </Panel>

      <Panel
        title="Tasks"
        updatedAt={tasks.dataUpdatedAt}
        staleAfterMs={120_000}
        error={errorOf(tasks.error)}
        loading={tasks.isLoading}
        empty={tasks.data && tasks.data.rows.length === 0 ? "No task records." : null}
        onRefresh={() => tasks.refetch()}
        refreshing={tasks.isFetching}
      >
        {tasks.data && tasks.data.rows.length > 0 && (
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
              {[...tasks.data.rows].reverse().map((row) => (
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

/**
 * Every row on this panel shares one grid, so the columns line up down the
 * whole thing — marker, what it is, when. That alignment IS the relation: the
 * queue reads as the continuation of the nodes rather than a second widget
 * that happens to sit underneath.
 */
const ROW = "grid grid-cols-[1.5rem_1fr_auto] items-center gap-3 px-3 py-1.5";

/**
 * The pool as one pipeline: what is on a node, then what is waiting for one.
 *
 * Drawn as rows rather than cards. The rest of this console is dense bordered
 * rows at one type size, and tinted boxes in the middle of it read as a
 * different application — while also making occupied and idle slots different
 * heights, so the grid jumped as the pool filled.
 *
 * Occupancy still has to be visible at a glance, since a list of two running
 * tasks looks identical whether the pool holds two nodes or eight. A dot per
 * slot carries that at a fraction of the weight: a column of hollow dots is an
 * idle pool without a single filled rectangle.
 */
function Pipeline({
  shape,
  busy,
  now,
  progress,
}: {
  shape: PoolShape;
  busy: number;
  now: number;
  progress: Map<string, TaskRow>;
}) {
  return (
    <div>
      <SectionRow label="nodes" aside={`${busy} of ${shape.slots.length} busy`} />
      {shape.slots.map((slot) => (
        <div key={slot.index} className={cn(ROW, "border-b border-[var(--border)]/60")}>
          <Dot filled={slot.task !== null} />
          {slot.task ? (
            <>
              <span className="truncate font-mono text-[12px]" title={slot.task.task}>
                {taskLabel(slot.task.task)}
              </span>
              {/* Elapsed answers "is this stuck"; the wall clock is what
                  correlates a slot with a line in a log or a task row. */}
              <span className="tnum flex shrink-0 items-center gap-2 text-[11px] text-[var(--fg-faint)]">
                <Bar row={progress.get(slot.task.task)} />
                {elapsed(slot.task.start_time, now) || "—"}
                {slot.task.start_time && ` · since ${clock(slot.task.start_time)}`}
              </span>
            </>
          ) : (
            <>
              <span className="text-[12px] text-[var(--fg-faint)]">idle</span>
              <span />
            </>
          )}
        </div>
      ))}

      {shape.queue.length > 0 && (
        <>
          <SectionRow
            label="queue"
            aside={
              shape.starved > 0 ? (
                <span
                  className="text-amber-400"
                  title="more waiting than there are free nodes — these wait for the pool to grow, not for a task to end"
                >
                  {shape.queue.length} waiting · {shape.starved} beyond capacity
                </span>
              ) : (
                `${shape.queue.length} waiting`
              )
            }
          />
          {shape.queue.map((task, index) => (
            <div
              key={`${task.job}/${task.task}`}
              className={cn(ROW, "border-b border-[var(--border)]/60")}
            >
              {/* Position, not a badge: "3rd in line" is the question, and a
                  column of identical `queued` badges cannot answer it. */}
              <span className="text-right font-mono text-[11px] text-[var(--fg-faint)]">
                {index + 1}
              </span>
              <span
                className="truncate font-mono text-[12px] text-[var(--fg-muted)]"
                title={task.task}
              >
                {taskLabel(task.task)}
              </span>
              <span className="tnum shrink-0 text-[11px] text-[var(--fg-faint)]">
                waiting {elapsed(task.created, now) || "—"}
              </span>
            </div>
          ))}
        </>
      )}
    </div>
  );
}

/**
 * How far along a running task is, when its kind can say.
 *
 * Absent for a kind that cannot answer -- scoring one rung is opaque from
 * outside it -- and absent is drawn as nothing rather than as an empty bar,
 * because an empty bar reads as a task that is stuck rather than one that
 * simply does not report.
 *
 * The phrase is in the tooltip rather than beside it: the unit is what makes
 * the number mean something, but a row of them would crowd out the task name.
 */
function Bar({ row }: { row?: TaskRow }) {
  const of = row?.progress;
  const eta = row?.eta_seconds;
  if (!of || of.total <= 0) {
    // An ETA without a bar is still worth saying: a kind can know roughly how
    // long its work takes without being able to report where it is inside one.
    return eta != null ? <span className="tnum">{duration(eta)} left</span> : null;
  }
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
      <span className="tnum">{Math.round(fraction * 100)}%</span>
      {eta != null && <span className="tnum">· {duration(eta)} left</span>}
    </span>
  );
}

/** Filled = a node is committed. A column of hollow dots is an idle pool. */
function Dot({ filled }: { filled: boolean }) {
  return (
    <span className="flex justify-center">
      <span
        className={cn(
          "size-1.5 rounded-full",
          filled ? "bg-emerald-500" : "ring-1 ring-[var(--fg-faint)]/50",
        )}
      />
    </span>
  );
}

/** A heading on the same grid as the rows it introduces, so nothing shifts. */
function SectionRow({ label, aside }: { label: string; aside?: React.ReactNode }) {
  return (
    <div
      className={cn(
        ROW,
        "border-b border-[var(--border)] text-[11px] text-[var(--fg-faint)] uppercase tracking-wider",
      )}
    >
      <span />
      <span>{label}</span>
      <span className="normal-case tracking-normal">{aside}</span>
    </div>
  );
}

/** Finished work, newest first — the half that used to be mixed in with the rest. */
function History({ history }: { history: Task[] }) {
  if (history.length === 0) return null;
  return (
    <div>
      <SectionRow label="recently finished" />
      <Table>
        <tbody>
          {history.slice(0, 6).map((task) => (
            <tr key={`${task.job}/${task.task}`}>
              <Td mono title={task.task}>
                {taskLabel(task.task)}
              </Td>
              <Td>
                {/* Coloured on state AND exit code, and the code's MEANING is
                    the badge — 124 is the guard's deadline, 137 the OOM killer.
                    The number itself sat in its own column saying less than the
                    word beside it, so it is a tooltip now. */}
                <StatusBadge
                  state={taskOutcome(task.state, task.exit_code)}
                  tone={taskTone(task.state, task.exit_code)}
                  title={[
                    `Batch reports state "${task.state}"`,
                    task.exit_code == null ? null : `exit ${task.exit_code}`,
                    exitMeaning(task.exit_code),
                  ]
                    .filter(Boolean)
                    .join(" — ")}
                />
              </Td>
              <Td right className="tnum text-[var(--fg-muted)]">
                {span(task.start_time, task.end_time)}
              </Td>
              <Td right className="text-[var(--fg-faint)]" title={task.end_time ?? undefined}>
                {since(task.end_time)}
              </Td>
            </tr>
          ))}
        </tbody>
      </Table>
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
