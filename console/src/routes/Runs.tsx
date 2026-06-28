import { Link } from "@tanstack/react-router";
import { useMemo } from "react";
import { useRunsView } from "@/api/queries";
import type { Phase } from "@/api/types";
import { Panel } from "@/components/Panel";
import { StatusBadge } from "@/components/StatusBadge";
import { Table, Td, Th } from "@/components/Table";
import { errorOf } from "@/lib/error";
import { count, runLabel } from "@/lib/format";

/**
 * Phases that mean this run still has work in the pool.
 *
 * `queued` counts, and that is deliberate: the question here is "is this run
 * abandoned", and a task waiting for a node is not. It is a DIFFERENT question
 * from the one cost accounting asks — which excludes `queued`, because queue
 * time is not node time — and the two answers are allowed to differ now that
 * both are asked in the same words.
 */
const IN_FLIGHT = new Set<Phase>(["running", "starting", "queued"]);

/**
 * A run's `status` is a CLAIM, not an observation.
 *
 * It lives in the run's own event log, written by the training process itself
 * — so it records what a LIVING process did and cannot record how an attempt
 * died. A task killed by OOM, `maxWallClockTime`, SIGKILL or node loss is gone
 * before it can write `finished`, and the run then claims `running` forever.
 * Four runs on this share have claimed it since 07-31.
 *
 * So the claim is cross-checked against Batch, which is the authority on what
 * is executing: a run is only `running` if one of its tasks is a task Batch has
 * live right now.
 */
type Verdict = {
  label: string;
  tone: "live" | "warn" | "muted";
  title: string;
} | null;

export function verdictFor(
  status: string | null,
  runName: string,
  liveRuns: Set<string>,
  runsWithTasks: Set<string>,
): Verdict {
  if ((status ?? "") !== "running") return null;
  if (liveRuns.has(runName)) {
    return {
      label: "running",
      tone: "live",
      title: "a task for this run is live in Batch",
    };
  }
  if (runsWithTasks.has(runName)) {
    return {
      label: "abandoned",
      tone: "warn",
      title:
        "the run log says running, nothing is executing, and none of its tasks " +
        "reported a terminal cause — it died without cleanup",
    };
  }
  // No task records at all. Runs from before the task log existed (its earliest
  // record is 2026-08-02) have no observer half to reconcile against, so this
  // is an inference rather than a finding — hence the question mark.
  return {
    label: "abandoned?",
    tone: "muted",
    title:
      "the run log says running and nothing is executing, but this run predates " +
      "the task log so there is no record of how it ended",
  };
}

export function Runs() {
  const view = useRunsView();
  const parts = view.data?.parts;
  const runs = parts?.runs.payload ?? null;

  const { liveRuns, runsWithTasks } = useMemo(() => {
    // Batch knows which TASKS are live; `task_runs` says which RUN each task
    // was for. The server does that projection because it needs the whole task
    // log and this page does not — but which STATES count as live stays here,
    // because that is what "running" means and it is a UI decision.
    const taskRuns = view.data?.task_runs ?? {};
    const liveTasks = new Set(
      (parts?.jobs.payload?.jobs ?? []).flatMap((job) =>
        job.tasks.filter((task) => IN_FLIGHT.has(task.phase)).map((task) => task.task),
      ),
    );
    const live = new Set<string>();
    for (const [taskId, runId] of Object.entries(taskRuns)) {
      if (liveTasks.has(taskId)) live.add(runId);
    }
    return { liveRuns: live, runsWithTasks: new Set(Object.values(taskRuns)) };
  }, [view.data?.task_runs, parts?.jobs.payload]);

  // Only claim a run is abandoned once BOTH cross-check sources have answered.
  // Before that every run would look abandoned, which is worse than saying
  // nothing: the page would cry wolf on every load. They now arrive together,
  // so this is one condition rather than two — but it still has to be checked:
  // either part can be individually unavailable.
  const checked = Boolean(parts?.jobs.payload && parts?.tasks.payload);

  return (
    <Panel
      title="Runs"
      updatedAt={view.dataUpdatedAt}
      staleAfterMs={120_000}
      // The run list itself failing is what blanks this panel. The two
      // cross-check parts failing is not: the table still answers "what runs
      // exist", it just cannot second-guess a claimed status, and `checked`
      // above is what holds it back from guessing.
      error={errorOf(view.error) ?? parts?.runs.error ?? null}
      loading={view.isLoading}
      empty={runs && runs.runs.length === 0 ? "No published runs." : null}
      onRefresh={() => view.refresh()}
      refreshing={view.isFetching}
    >
      {runs && runs.runs.length > 0 && (
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
            {runs.runs.map((run) => {
              const verdict = checked
                ? verdictFor(run.status, run.name, liveRuns, runsWithTasks)
                : null;
              return (
                <tr key={run.name}>
                  <Td mono>
                    <Link
                      to="/runs/$runId"
                      params={{ runId: run.name }}
                      title={run.name}
                      className="hover:underline"
                    >
                      {runLabel(run.name)}
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
                    {verdict ? (
                      <StatusBadge
                        state={verdict.label}
                        tone={verdict.tone}
                        title={verdict.title}
                      />
                    ) : (
                      <StatusBadge state={run.status} />
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
