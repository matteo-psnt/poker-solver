import { useJobs, useLegs, useRuns } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { StatusBadge, shortState } from "@/components/StatusBadge";
import { Table, Td, Th } from "@/components/Table";
import { errorOf } from "@/lib/error";
import { count, runLabel } from "@/lib/format";
import { Link } from "@tanstack/react-router";
import { useMemo } from "react";

/** Batch task states that mean a node is actually working on it. */
const LIVE = new Set(["running", "preparing", "active"]);

/**
 * A run's `status` is a CLAIM, not an observation.
 *
 * It lives in the run's own event log, written by the training process itself
 * — so it records what a LIVING process did and cannot record how an attempt
 * died. A leg killed by OOM, `maxWallClockTime`, SIGKILL or node loss is gone
 * before it can write `finished`, and the run then claims `running` forever.
 * Four runs on this share have claimed it since 07-31.
 *
 * So the claim is cross-checked against Batch, which is the authority on what
 * is executing: a run is only `running` if one of its legs is a task Batch has
 * live right now.
 */
type Verdict = { label: string; tone: "live" | "warn" | "muted"; title: string } | null;

export function verdictFor(
  status: string | null,
  runName: string,
  liveRuns: Set<string>,
  runsWithLegs: Set<string>,
): Verdict {
  if ((status ?? "") !== "running") return null;
  if (liveRuns.has(runName)) {
    return { label: "running", tone: "live", title: "a leg for this run is live in Batch" };
  }
  if (runsWithLegs.has(runName)) {
    return {
      label: "abandoned",
      tone: "warn",
      title:
        "the run log says running, nothing is executing, and none of its legs " +
        "reported a terminal cause — it died without cleanup",
    };
  }
  // No leg records at all. Runs from before the leg log existed (its earliest
  // record is 2026-08-02) have no observer half to reconcile against, so this
  // is an inference rather than a finding — hence the question mark.
  return {
    label: "abandoned?",
    tone: "muted",
    title:
      "the run log says running and nothing is executing, but this run predates " +
      "the leg log so there is no record of how it ended",
  };
}

export function Runs() {
  const runs = useRuns();
  const jobs = useJobs(50);
  const legs = useLegs(0);

  const { liveRuns, runsWithLegs } = useMemo(() => {
    // Join on task id: Batch knows which TASKS are live, the leg log knows
    // which run each task belonged to. Neither can answer alone.
    const liveTasks = new Set(
      (jobs.data?.jobs ?? []).flatMap((job) =>
        job.tasks.filter((task) => LIVE.has(shortState(task.state))).map((task) => task.task),
      ),
    );
    const live = new Set<string>();
    const withLegs = new Set<string>();
    for (const leg of legs.data?.rows ?? []) {
      if (!leg.run_id) continue;
      withLegs.add(leg.run_id);
      if (liveTasks.has(leg.task_id)) live.add(leg.run_id);
    }
    return { liveRuns: live, runsWithLegs: withLegs };
  }, [jobs.data, legs.data]);

  // Only claim a run is abandoned once BOTH cross-check sources have answered.
  // Before that every run would look abandoned, which is worse than saying
  // nothing: the page would cry wolf on every load.
  const checked = Boolean(jobs.data && legs.data);

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
            {runs.data.runs.map((run) => {
              const verdict = checked
                ? verdictFor(run.status, run.name, liveRuns, runsWithLegs)
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
