import { Link } from "@tanstack/react-router";
import { useMemo, useState } from "react";
import { useBlueprintRun, useRunsView } from "@/api/queries";
import type { Runs as RunsPayload } from "@/api/types";
import { Panel } from "@/components/Panel";
import { StatusBadge } from "@/components/StatusBadge";
import { Table, Td, Th } from "@/components/Table";
import { errorOf } from "@/lib/error";
import { count, runLabel } from "@/lib/format";
import { IN_FLIGHT } from "@/lib/phase";
import { cn } from "@/lib/utils";

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
  /** Whether the task log holds a row for this run — `RunSummary.has_tasks`,
      answered by the `runs` command rather than by downloading the log. */
  hasTasks: boolean,
): Verdict {
  // Batch first: a continuation (`submit --run`) trains a run whose log still
  // says `completed`, so a live task outranks whatever the claim is.
  if (liveRuns.has(runName)) {
    return {
      label: "running",
      tone: "live",
      title:
        status === "running"
          ? "a task for this run is live in Batch"
          : `a task for this run is live in Batch; the run log still says ${status ?? "nothing"}`,
    };
  }
  if ((status ?? "") !== "running") return null;
  if (hasTasks) {
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

type RunSummary = RunsPayload["runs"][number];

/** `100 bb` from the arm (`d100`) or the run name (`-d100-`); empty when a run names no depth. */
export function depthOf(run: Pick<RunSummary, "name" | "arm">): string {
  const match = (run.arm ?? "").match(/^d(\d+)/) ?? run.name.match(/-d(\d+)(?:[-_]|$)/);
  return match ? `${Number(match[1])} bb` : "";
}

/**
 * Every published run, grouped by the experiment that made it.
 *
 * The list opens on what matters now: the runs the live bot plays (the blueprint
 * box's ladder) and the runs with work in Batch. Everything else is one click
 * away under "all", and a search narrows either view. Experiments keep the run
 * list's own order -- newest first -- so the most recent work leads.
 */
export function Runs() {
  const view = useRunsView();
  const box = useBlueprintRun();
  const parts = view.data?.parts;
  const runs = parts?.runs.payload ?? null;
  const [scope, setScope] = useState<"active" | "all">("active");
  const [query, setQuery] = useState("");

  const liveRuns = useMemo(() => {
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
    return live;
  }, [view.data?.task_runs, parts?.jobs.payload]);

  // The box reports its ladder in chips; the console speaks big blinds.
  const fieldedAt = useMemo(() => {
    const bigBlind = box.data?.big_blind ?? 2;
    const byRun = new Map<string, number>();
    for (const [chips, runId] of Object.entries(box.data?.runs_by_depth ?? {})) {
      byRun.set(runId, Number(chips) / bigBlind);
    }
    return byRun;
  }, [box.data]);

  // Only claim a run is abandoned once BOTH cross-check sources have answered.
  // Before that every run would look abandoned, which is worse than saying
  // nothing: the page would cry wolf on every load.
  const checked = Boolean(parts?.jobs.payload && parts?.tasks.payload);

  const groups = useMemo(() => {
    const needle = query.trim().toLowerCase();
    const shown = (runs?.runs ?? []).filter((run) => {
      const inScope =
        scope === "all" ||
        fieldedAt.has(run.name) ||
        liveRuns.has(run.name) ||
        run.status === "running";
      const haystack = `${run.name} ${run.config_name ?? ""} ${run.experiment_id ?? ""}`;
      return inScope && (!needle || haystack.toLowerCase().includes(needle));
    });
    const byExperiment = new Map<string, RunSummary[]>();
    for (const run of shown) {
      const key = run.experiment_id ?? "no experiment";
      byExperiment.set(key, [...(byExperiment.get(key) ?? []), run]);
    }
    return [...byExperiment.entries()];
  }, [runs, scope, query, fieldedAt, liveRuns]);
  const shownCount = groups.reduce((n, [, rs]) => n + rs.length, 0);

  return (
    <Panel
      title={`Runs — ${shownCount} of ${runs?.runs.length ?? 0}`}
      aside={
        <span className="flex items-center gap-2 text-[11px]">
          {(
            [
              ["active", "fielded and active"],
              ["all", "all"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              onClick={() => setScope(value)}
              className={cn(
                "rounded px-2 py-0.5",
                scope === value
                  ? "bg-white/[0.08] text-[var(--fg)]"
                  : "text-[var(--fg-muted)] hover:text-[var(--fg)]",
              )}
            >
              {label}
            </button>
          ))}
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="find run, config, experiment"
            aria-label="Find a run"
            className="w-[200px] rounded border border-[var(--border)] bg-transparent px-2 py-0.5 text-[11px] text-[var(--fg)] placeholder:text-[var(--fg-faint)]"
          />
        </span>
      }
      updatedAt={view.dataUpdatedAt}
      staleAfterMs={120_000}
      // The run list itself failing is what blanks this panel. The two
      // cross-check parts failing is not: the table still answers "what runs
      // exist", it just cannot second-guess a claimed status, and `checked`
      // above is what holds it back from guessing.
      error={errorOf(view.error) ?? parts?.runs.error ?? null}
      loading={view.isLoading}
      empty={
        runs && shownCount === 0
          ? runs.runs.length === 0
            ? "No published runs."
            : "No run matches."
          : null
      }
      onRefresh={() => view.refresh()}
      refreshing={view.isFetching}
    >
      {shownCount > 0 && (
        <Table>
          <thead>
            <tr>
              <Th>run</Th>
              <Th>depth</Th>
              <Th>config</Th>
              <Th right>iterations</Th>
              <Th>status</Th>
            </tr>
          </thead>
          {groups.map(([experiment, members]) => (
            <tbody key={experiment}>
              <tr>
                <td
                  colSpan={5}
                  className="border-b border-[var(--border)] bg-white/[0.02] px-3 pt-3 pb-1.5 text-[11px] text-[var(--fg-muted)]"
                >
                  <span className="font-mono text-[var(--fg)]">{experiment}</span>
                  <span className="ml-2 text-[var(--fg-faint)]">
                    {members.length} run{members.length === 1 ? "" : "s"}
                  </span>
                </td>
              </tr>
              {members.map((run) => {
                const verdict = checked
                  ? verdictFor(run.status, run.name, liveRuns, run.has_tasks)
                  : null;
                const seat = fieldedAt.get(run.name);
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
                    <Td className="text-[var(--fg-muted)]">{depthOf(run) || "—"}</Td>
                    <Td className="text-[var(--fg-muted)]">{run.config_name ?? "—"}</Td>
                    <Td right>{count(run.iterations)}</Td>
                    <Td>
                      <span className="flex flex-wrap items-center gap-1.5">
                        {seat != null && (
                          <StatusBadge
                            state={`fielded at ${seat} bb`}
                            tone="live"
                            title="the live bot plays this run at this depth"
                          />
                        )}
                        {verdict ? (
                          <StatusBadge
                            state={verdict.label}
                            tone={verdict.tone}
                            title={verdict.title}
                          />
                        ) : (
                          <StatusBadge state={run.status} />
                        )}
                      </span>
                    </Td>
                  </tr>
                );
              })}
            </tbody>
          ))}
        </Table>
      )}
    </Panel>
  );
}
