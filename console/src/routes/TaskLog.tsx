import { useLog, useTasks } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { StatusBadge } from "@/components/StatusBadge";
import { errorOf } from "@/lib/error";
import { clock, count, duration, runLabel, span } from "@/lib/format";
import { Link, getRouteApi } from "@tanstack/react-router";
import { ArrowLeft } from "lucide-react";

const route = getRouteApi("/tasks/$taskId");

/**
 * Why a task died, which is the question the Tasks page raises and could not
 * answer.
 *
 * Reads the copy published to the share, NOT the node's live files: the pool
 * scales to zero within minutes of a task ending, so for exactly the failed
 * tasks worth reading the node is already gone. That is what the publish-on-exit
 * trap exists for.
 */
export function TaskLog() {
  const { taskId } = route.useParams();
  const log = useLog(taskId);
  const tasks = useTasks(0);
  const lines = log.data?.lines ?? [];
  // The task's OWN record, which exists long before its log does: the log is
  // published from the node and a running task has usually published none yet,
  // so opening one used to show nothing at all.
  const row = (tasks.data?.rows ?? []).find((r) => r.task_id === taskId);

  return (
    <div className="space-y-3">
      <Link
        to="/tasks"
        className="inline-flex items-center gap-1.5 text-[12px] text-[var(--fg-muted)] hover:text-[var(--fg)]"
      >
        <ArrowLeft className="size-3.5" />
        Tasks
      </Link>

      <Panel
        title={row?.what || taskId}
        updatedAt={tasks.dataUpdatedAt}
        staleAfterMs={120_000}
        error={errorOf(tasks.error)}
        loading={tasks.isLoading}
        empty={tasks.data && !row ? "No record for this task." : null}
        onRefresh={() => tasks.refetch()}
        refreshing={tasks.isFetching}
      >
        {row && (
          <div className="grid grid-cols-2 gap-x-6 gap-y-3 p-3 sm:grid-cols-3 lg:grid-cols-4">
            <Fact label="task" value={row.task_id} mono />
            <div>
              <Label>cause</Label>
              <StatusBadge state={row.cause} />
            </div>
            <Fact label="run" value={row.run_id ? runLabel(row.run_id) : "—"} mono />
            <Fact label="attempt" value={String(row.attempt ?? "—")} />
            <Fact label="workers" value={String(row.workers ?? "—")} />
            <Fact label="started" value={clock(row.started_at)} />
            <Fact label="took" value={span(row.started_at, row.ended_at, Date.now())} />
            {row.progress ? (
              <Fact
                label="progress"
                value={`${count(row.progress.done)} / ${count(row.progress.total)} ${
                  row.progress.unit
                }`}
              />
            ) : (
              <Fact label="did" value={row.units ? count(row.units) : "—"} />
            )}
            {row.eta_seconds != null && <Fact label="left" value={duration(row.eta_seconds)} />}
            <Fact label="node" value={row.node_id ? row.node_id.slice(6, 18) : "—"} mono />
          </div>
        )}
        {/* Its own row, because "which code" is a different question from "what
            happened" and it is the one asked when two arms disagree. Hidden
            entirely for tasks recorded before this existed: three em-dashes read
            as a task with no provenance rather than as a record predating it. */}
        {row && (row.code_snapshot || row.git_commit || row.git_branch) && (
          <div className="grid grid-cols-2 gap-x-6 gap-y-3 border-t border-[var(--border)] p-3 sm:grid-cols-3 lg:grid-cols-4">
            <Fact label="branch" value={row.git_branch || "—"} mono />
            <Fact
              label="commit"
              value={
                row.git_commit
                  ? `${row.git_commit.slice(0, 12)}${row.git_dirty === "1" ? " (dirty)" : ""}`
                  : "—"
              }
              mono
            />
            {/* Last and widest: the only one that stays exact when the tree was
                dirty, which is the normal state of an experiment worktree. */}
            <Fact label="snapshot" value={row.code_snapshot || "—"} mono />
          </div>
        )}
      </Panel>

      <Panel
        title="Published log"
        updatedAt={log.dataUpdatedAt}
        staleAfterMs={Number.POSITIVE_INFINITY}
        error={errorOf(log.error)}
        loading={log.isLoading}
        empty={log.data && lines.length === 0 ? "The published log is empty." : null}
        onRefresh={() => log.refetch()}
        refreshing={log.isFetching}
      >
        {lines.length > 0 && (
          <pre className="max-h-[70vh] overflow-auto px-3 py-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap">
            {lines.map((line, index) => (
              // Index keys are correct here: this is an immutable, ordered
              // snapshot of a finished log, never reordered or spliced.
              // biome-ignore lint/suspicious/noArrayIndexKey: immutable log snapshot
              <div key={index} className={lineTone(line)}>
                {line}
              </div>
            ))}
          </pre>
        )}
      </Panel>

      <p className="max-w-[70ch] px-1 text-[12px] leading-relaxed text-[var(--fg-muted)]">
        The copy published to the share. The node's own files are gone once the pool scales down,
        which is why this is the durable one.
      </p>
    </div>
  );
}

/** Make the lines that explain a death findable without reading all 400. */
function lineTone(line: string): string {
  const lower = line.toLowerCase();
  if (/\b(error|fatal|traceback|failed|refus)/.test(lower)) return "text-red-400";
  if (/\b(warn|timeout|retry|killed)/.test(lower)) return "text-amber-400";
  if (line.startsWith("[run_task")) return "text-[var(--fg-faint)]";
  return "";
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[11px] text-[var(--fg-faint)] uppercase tracking-wider">{children}</div>
  );
}

function Fact({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <Label>{label}</Label>
      <div className={mono ? "truncate font-mono text-[12px]" : "tnum"} title={value}>
        {value}
      </div>
    </div>
  );
}
