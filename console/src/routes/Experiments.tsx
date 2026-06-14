import { useExperimentView, usePromote, useRuns } from "@/api/queries";
import type { Report, RunSummary } from "@/api/types";
import { Actions, Field, Outcome, Run, Select, Text } from "@/components/Form";
import { Panel } from "@/components/Panel";
import { Table, Td, Th } from "@/components/Table";
import { given } from "@/lib/body";
import { errorOf } from "@/lib/error";
import { count, mbb, runLabel } from "@/lib/format";
import { cn } from "@/lib/utils";
import { useMemo, useState } from "react";

/**
 * Where an experiment is decided: every arm of it, then the promotion that
 * closes the loop.
 *
 * Both are one page because they are one sequence — `report` says which arm won
 * and `promote` moves the baseline, and the second is the only irreversible
 * thing here, which is easier to weigh with the evidence directly above rather
 * than a click away.
 *
 * The rule the page obeys: **never compare across knob tiers.** `report`
 * refuses to, and the console offers no way to override it.
 */
export function Experiments() {
  const runs = useRuns();
  return (
    <div className="space-y-3">
      <ArmReport runs={runs.data?.runs ?? []} loading={runs.isLoading} error={runs.error} />
      <Promote runs={runs.data?.runs ?? []} />
    </div>
  );
}

/**
 * `report`. The experiment picker is derived from the run listing rather than
 * typed, because the set of experiments exists nowhere else — `report` takes an
 * id and `runinfo` reports one run's, so without this the report was reachable
 * only by someone who already knew the answer.
 */
function ArmReport({
  runs,
  loading,
  error,
}: {
  runs: RunSummary[];
  loading: boolean;
  error: unknown;
}) {
  const experiments = useMemo(() => {
    const seen = new Set<string>();
    for (const run of runs) if (run.experiment_id) seen.add(run.experiment_id);
    return [...seen].sort();
  }, [runs]);

  const [experiment, setExperiment] = useState("");
  const view = useExperimentView(experiment || null);
  const report = view.data?.parts?.report;

  // Each arm's run record, keyed for the table. `report` says how an arm SCORED;
  // this says what it WAS, and the two were previously on different pages.
  const armRuns = useMemo(() => {
    const byName = new Map<string, RunSummary>();
    for (const run of view.data?.arm_runs ?? []) byName.set(run.name, run);
    return byName;
  }, [view.data?.arm_runs]);

  return (
    <Panel
      title="Experiment report"
      updatedAt={view.dataUpdatedAt}
      staleAfterMs={120_000}
      error={errorOf(error) ?? errorOf(view.error) ?? report?.error ?? null}
      loading={loading || view.isLoading}
      empty={
        experiments.length === 0 && !loading
          ? "No run carries an experiment tag. Submit with an experiment id to build one."
          : null
      }
      onRefresh={() => view.refetch()}
      refreshing={view.isFetching}
    >
      {experiments.length > 0 && (
        <div className="border-b border-[var(--border)]">
          <Field label="experiment" hint="Built from the runs that carry a tag.">
            <Select value={experiment} onChange={setExperiment} options={experiments} />
          </Field>
        </div>
      )}
      {report?.payload && <Arms report={report.payload} armRuns={armRuns} />}
    </Panel>
  );
}

function Arms({ report, armRuns }: { report: Report; armRuns: Map<string, RunSummary> }) {
  return (
    <>
      {report.notes.length > 0 && (
        <p className="border-b border-[var(--border)] px-3 py-1.5 text-[11px] text-[var(--fg-faint)]">
          {report.notes.join(" · ")}
        </p>
      )}
      <Table>
        <thead>
          <tr>
            <Th>arm</Th>
            <Th>run</Th>
            <Th>config</Th>
            <Th>branch</Th>
            <Th right>at</Th>
            <Th right>mbb/g</Th>
            <Th right>vs control</Th>
            <Th right>p</Th>
          </tr>
        </thead>
        <tbody>
          {report.arms.map((arm) => {
            const control = arm.run_id === report.control_run_id;
            // A blocked pairing is NOT "no difference". Rendering it as a dash
            // would claim a comparison that never happened, which is the one
            // reading of this table that could move a baseline wrongly.
            const blocked = (arm.vs_control_blocked ?? []).length > 0;
            return (
              <tr key={arm.run_id} className={cn(control && "bg-white/[0.03]")}>
                <Td>
                  {arm.arm}
                  {control && <span className="ml-2 text-[var(--fg-faint)]">control</span>}
                </Td>
                <Td mono title={arm.run_id}>
                  {runLabel(arm.run_id)}
                </Td>
                <Td className="text-[var(--fg-muted)]">
                  {armRuns.get(arm.run_id)?.config_name ?? "—"}
                </Td>
                <Td className="text-[var(--fg-muted)]">{arm.git_branch ?? "—"}</Td>
                <Td right className="text-[var(--fg-muted)]">
                  {count(arm.checkpoint_iteration)}
                </Td>
                <Td right>{mbb(arm.exploitability_mbb, arm.std_error_mbb)}</Td>
                <Td right title={blocked ? (arm.vs_control_blocked ?? []).join("; ") : undefined}>
                  {blocked ? (
                    <span className="text-amber-400">unpaired</span>
                  ) : arm.vs_control_mbb == null ? (
                    "—"
                  ) : (
                    <span
                      className={cn(
                        arm.vs_control_mbb < 0 ? "text-emerald-400" : "text-[var(--fg-muted)]",
                      )}
                    >
                      {arm.vs_control_mbb > 0 ? "+" : ""}
                      {arm.vs_control_mbb.toFixed(1)}
                    </span>
                  )}
                </Td>
                <Td right className="text-[var(--fg-muted)]">
                  {arm.vs_control_p_value == null ? "—" : arm.vs_control_p_value.toFixed(3)}
                </Td>
              </tr>
            );
          })}
        </tbody>
      </Table>
    </>
  );
}

/** `promote`. The only irreversible control on this page. */
function Promote({ runs }: { runs: RunSummary[] }) {
  const promote = usePromote();
  const [run, setRun] = useState("");
  const [rationale, setRationale] = useState("");

  const promotable = runs.filter((entry) => entry.loadable).map((entry) => entry.name);
  // The command requires a rationale and would refuse an empty one. Requiring
  // it here too is not a second rule — it is the same one, enforced before a
  // share read rather than after.
  const ready = Boolean(run && rationale.trim());

  return (
    <Panel title="Promote — move the baseline">
      <div className="divide-y divide-[var(--border)]/50">
        <Field label="run" hint="Must be published and have a checkpoint.">
          <Select value={run} onChange={setRun} options={promotable} />
        </Field>
        <Field
          label="rationale"
          hint="Required. A lineage that moved for an unrecorded reason cannot be audited later — which is the whole point of recording it."
        >
          <Text value={rationale} onChange={setRationale} mono={false} placeholder="why this run" />
        </Field>
      </div>

      <Actions note={ready ? null : "Needs a run and a rationale."}>
        <Run
          label="Promote"
          danger
          pending={promote.isPending}
          disabled={!ready}
          onClick={() => promote.mutate(given({ run, rationale }))}
        />
      </Actions>

      <Outcome error={errorOf(promote.error)}>
        {promote.data && (
          <span>
            baseline is now {promote.data.run_id}
            {promote.data.checkpoint_iteration != null &&
              ` at ${count(promote.data.checkpoint_iteration)}`}
          </span>
        )}
      </Outcome>
    </Panel>
  );
}
