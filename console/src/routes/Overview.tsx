import { useNow } from "@/api/queries";
import type { TaskRow } from "@/api/types";
import { Panel } from "@/components/Panel";
import { StatusBadge } from "@/components/StatusBadge";
import { Table, Td, Th } from "@/components/Table";
import { clock, count, duration, since, span, taskLabel } from "@/lib/format";
import { type PoolShape, elapsed, poolShape } from "@/lib/pool";
import { cn } from "@/lib/utils";
import { Link } from "@tanstack/react-router";
import { useMemo } from "react";

/**
 * The page the console exists for: what is happening, and did anything die.
 *
 * ONE request for the whole screen. It used to be four — pool, jobs, tasks and
 * autoscale each with its own cache slot and its own poll interval — so the four
 * panels were never the same age as each other and the freshest one made the
 * page look live while another was 45 seconds behind. Now every number here was
 * read in the same sweep, which is what makes a single age badge honest.
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
  const autoscale = parts?.autoscale;
  const autoscaleData = autoscale?.payload;

  // Recomputed each render rather than ticked: the panel already re-renders on
  // the 15s poll, and a second timer would redraw it between polls to move one
  // "2h 14m" by a minute.
  const now = Date.now();
  const shape = useMemo(
    () => poolShape(jobsData ?? undefined, poolData?.target_dedicated_nodes),
    [jobsData, poolData],
  );
  const busy = shape.slots.filter((slot) => slot.task !== null).length;
  // Batch knows which task holds a node; only the task log knows how far along
  // it is. Neither can draw the bar alone.
  const progress = useMemo(() => {
    const byTask = new Map<string, TaskRow>();
    for (const row of tasksData?.rows ?? []) {
      if (row.progress || row.eta_seconds != null) byTask.set(row.task_id, row);
    }
    return byTask;
  }, [tasksData]);

  return (
    <div className="space-y-3">
      <Panel
        title="Pool"
        updatedAt={view.dataUpdatedAt}
        staleAfterMs={30_000}
        error={pool?.error ?? null}
        loading={view.isLoading}
        onRefresh={() => view.refetch()}
        refreshing={view.isFetching}
      >
        {poolData && (
          <div className="grid grid-cols-2 gap-x-6 gap-y-2 p-3 sm:grid-cols-4">
            <Stat label="pool" value={poolData.pool_id} mono />
            {/* ALLOCATED against wanted -- machines that exist. The panel below
                counts how many of them hold a task, which is a different fact
                that used to be written the same way and read as a contradiction. */}
            <Stat
              label="nodes"
              value={`${count(poolData.current_dedicated_nodes)} of ${count(
                poolData.target_dedicated_nodes,
              )} wanted`}
            />
            <Stat label="allocation" value={poolData.allocation_state?.split(".").pop() ?? "—"} />
            <Stat label="vm size" value={poolData.vm_size ?? "—"} mono />
            {poolData.resize_errors.map((e) => (
              <p key={e.code} className="col-span-full font-mono text-[12px] text-red-400">
                {e.code}: {e.message}
              </p>
            ))}
          </div>
        )}
      </Panel>

      {/* Directly under the pool because it answers the pool's own follow-up
          question: the panel above says how many nodes there are, and this says
          what the deployed formula thinks there should be. A pool that will not
          grow is diagnosed by reading the two together — and `error` here is a
          FIELD, not a failed request: Batch evaluated the formula and reported
          that it did not compute, which is the answer rather than the absence
          of one. */}
      <Panel
        title="Autoscale"
        updatedAt={view.dataUpdatedAt}
        staleAfterMs={60_000}
        error={autoscale?.error ?? null}
        loading={view.isLoading}
        onRefresh={() => view.refetch()}
        refreshing={view.isFetching}
      >
        {autoscaleData && (
          <div className="p-3">
            {/* Batch reports the CAUSE as a code plus named values, not a
                sentence. This rendered `{autoscaleData.error}` for as long as
                both existed — the contract said `str`, the command has always
                sent an object, and React throws on an object child. It only
                never fired because the formula has not errored since. */}
            {autoscaleData.error && (
              <div className="mb-2 font-mono text-[12px] text-[#E0655C]">
                <p>{autoscaleData.error.code}</p>
                {autoscaleData.error.message && (
                  <p className="opacity-80">{autoscaleData.error.message}</p>
                )}
                {Object.entries(autoscaleData.error.values).map(([name, value]) => (
                  <p key={name} className="opacity-70">
                    {name}: {value}
                  </p>
                ))}
              </div>
            )}
            <div className="flex flex-wrap gap-x-5 gap-y-1 font-mono text-[11px] text-[var(--fg-muted)]">
              {autoscaleData.variables.map((variable) => (
                <span key={variable}>{variable}</span>
              ))}
            </div>
            {autoscaleData.variables.length === 0 && !autoscaleData.error && (
              <p className="text-[var(--fg-faint)]">The formula evaluated to no variables.</p>
            )}
          </div>
        )}
      </Panel>

      {/* Batch and the task log answer DIFFERENT questions; the panels sit
          together because neither is sufficient alone. */}
      <Panel
        title={`In flight — ${busy} running${
          shape.queue.length > 0 ? `, ${shape.queue.length} queued` : ""
        }`}
        updatedAt={view.dataUpdatedAt}
        staleAfterMs={30_000}
        error={jobs?.error ?? null}
        loading={view.isLoading}
        empty={jobsData && shape.slots.length === 0 && shape.queue.length === 0 ? "Idle." : null}
        onRefresh={() => view.refetch()}
        refreshing={view.isFetching}
      >
        {jobsData && (shape.slots.length > 0 || shape.queue.length > 0) && (
          <Pipeline shape={shape} busy={busy} now={now} progress={progress} />
        )}
      </Panel>

      <Panel
        title="Recent tasks"
        updatedAt={view.dataUpdatedAt}
        staleAfterMs={120_000}
        error={tasks?.error ?? null}
        loading={view.isLoading}
        empty={tasksData && tasksData.rows.length === 0 ? "No task records." : null}
        onRefresh={() => view.refetch()}
        refreshing={view.isFetching}
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
              <Link
                to="/tasks/$taskId"
                params={{ taskId: slot.task.task }}
                className="truncate font-mono text-[12px] hover:underline"
                title={slot.task.task}
              >
                {taskLabel(slot.task.task)}
              </Link>
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

function Stat({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <div className="text-[11px] text-[var(--fg-faint)] uppercase tracking-wider">{label}</div>
      <div className={mono ? "font-mono" : "tnum"}>{value}</div>
    </div>
  );
}
