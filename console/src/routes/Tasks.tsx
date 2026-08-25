import { useCancelTask, useTasks } from "@/api/queries";
import type { TaskRow } from "@/api/types";
import { Confirm } from "@/components/Confirm";
import { Panel } from "@/components/Panel";
import { StatusBadge, displayName, toneFor } from "@/components/StatusBadge";
import { Table, Td, Th } from "@/components/Table";
import { errorOf } from "@/lib/error";
import { clock, count, duration, runLabel, since, span, taskLabel } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Link, getRouteApi, useNavigate } from "@tanstack/react-router";
import { useMemo } from "react";

const route = getRouteApi("/tasks");

/**
 * The densest view in the console, and the one opened when something died.
 *
 * Filters live in the URL rather than component state: a filtered view is then
 * a link, which is what someone actually wants to share when asking why a task
 * failed.
 */
export function Tasks() {
  const { cause } = route.useSearch();
  const navigate = useNavigate({ from: "/tasks" });
  const tasks = useTasks(0);
  // One `now` for the whole table, so every open-ended duration in a render is
  // measured against the same instant rather than drifting down the rows.
  const now = Date.now();

  const causes = useMemo(() => {
    const seen = new Set<string>();
    for (const row of tasks.data?.rows ?? []) if (row.cause) seen.add(row.cause);
    return [...seen].sort();
  }, [tasks.data]);

  const cancel = useCancelTask();
  const rows = useMemo(() => {
    const all = [...(tasks.data?.rows ?? [])].reverse();
    return cause ? all.filter((r) => r.cause === cause) : all;
  }, [tasks.data, cause]);

  return (
    <Panel
      title={`Tasks${cause ? ` · ${cause}` : ""}`}
      updatedAt={tasks.dataUpdatedAt}
      staleAfterMs={120_000}
      // The cancel too: it is the console's one destructive write, and a
      // refusal (expired login, task already terminal) left the row running
      // with nothing anywhere saying the click had failed.
      error={errorOf(tasks.error) ?? errorOf(cancel.error)}
      loading={tasks.isLoading}
      empty={tasks.data && rows.length === 0 ? "No tasks match." : null}
      onRefresh={() => tasks.refetch()}
      refreshing={tasks.isFetching}
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
              <Th>progress</Th>
              <Th right>started</Th>
              <Th right>took</Th>
              <Th right>ended</Th>
              {/* The cancel column. Unlabelled on purpose: a header saying
                  "cancel" over a mostly-empty column reads like a bulk action. */}
              <Th right>{""}</Th>
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
                      to="/tasks/$taskId"
                      params={{ taskId: row.task_id }}
                      title={row.task_id}
                      className="hover:underline"
                    >
                      {taskLabel(row.task_id)}
                    </Link>
                  </Td>
                  <Td right className="text-[var(--fg-faint)]">
                    {row.attempt ?? "—"}
                  </Td>
                  <Td className="text-[var(--fg-muted)]">{row.what || row.op || "—"}</Td>
                  {/* Not every task has a run: a `vector-sweep` is a measurement
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
                        `toneFor` keys off what the task log recorded, so passing
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
                  {/* Only a running task has either. A finished one showing a
                      bar would be a sample that stopped arriving, not a task
                      stuck there -- the server drops it for exactly that. */}
                  <Td className="text-[var(--fg-muted)]">
                    <Progress row={row} />
                  </Td>
                  <Td right className="text-[var(--fg-faint)]" title={row.started_at ?? undefined}>
                    {clock(row.started_at)}
                  </Td>
                  {/* Open-ended for a task still running, so "running 2h" and
                      "took 2h" stay distinguishable from "unknown". */}
                  <Td right className="tnum text-[var(--fg-muted)]">
                    {span(row.started_at, row.ended_at, now)}
                  </Td>
                  <Td right className="text-[var(--fg-faint)]" title={row.ended_at ?? undefined}>
                    {row.ended_at ? since(row.ended_at) : "—"}
                  </Td>
                  {/* Only where it means something. A finished task has nothing
                      to cancel, and a button that is present-but-useless on
                      every historical row makes the live ones harder to find. */}
                  <Td right>
                    {isLive(row) && (
                      <Confirm
                        label="cancel"
                        confirmLabel="really?"
                        title={`terminate ${row.task_id}`}
                        pending={cancel.isPending && cancel.variables?.task === row.task_id}
                        onConfirm={() =>
                          cancel.mutate({ job: row.job_id ?? "", task: row.task_id })
                        }
                      />
                    )}
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

/**
 * Worth cancelling only while it is still going. `ended_at` is the honest
 * signal: a row reconciled from Batch may carry a cause without having stopped,
 * and one still running has neither.
 */
function isLive(row: TaskRow): boolean {
  return !row.ended_at;
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

/**
 * A task's own account of how far it has got, and how long is left.
 *
 * Both are optional and independently so: a kind can know roughly how long its
 * work takes without being able to say where it is inside one unit of it, and a
 * kind can report position without enough history to predict a finish.
 */
function Progress({ row }: { row: TaskRow }) {
  const of = row.progress;
  const eta = row.eta_seconds;
  if (!of || of.total <= 0) {
    return eta == null ? <>—</> : <span className="tnum">{duration(eta)} left</span>;
  }
  const fraction = Math.max(0, Math.min(1, of.done / of.total));
  return (
    <span
      className="flex items-center gap-2"
      title={`${count(of.done)} / ${count(of.total)} ${of.unit}`}
    >
      <span className="h-1 w-14 overflow-hidden rounded-full bg-[var(--border)]">
        <span
          className="block h-full rounded-full bg-emerald-500/70"
          style={{ width: `${fraction * 100}%` }}
        />
      </span>
      <span className="tnum text-[11px]">{Math.round(fraction * 100)}%</span>
      {eta != null && (
        <span className="tnum text-[11px] text-[var(--fg-faint)]">{duration(eta)} left</span>
      )}
    </span>
  );
}
