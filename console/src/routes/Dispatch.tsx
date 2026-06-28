import type { UseMutationResult } from "@tanstack/react-query";
import { useState } from "react";
import { useConfigs, usePrecompute, useRuns, useScore, useSubmit } from "@/api/queries";
import type { Dispatched } from "@/api/types";
import { Actions, Field, Guard, Num, Outcome, Queued, Run, Select, Text } from "@/components/Form";
import { Panel } from "@/components/Panel";
import { given, int, overrides, rungs } from "@/lib/body";
import { errorOf } from "@/lib/error";

/**
 * The three commands that put work on the pool.
 *
 * Each button is one `Command.invoke` and nothing else. No "submit and then
 * score", no retry-then-cancel -- a composite becomes a COMMAND first and a button
 * second, because anything else is behaviour that is neither scriptable nor
 * reproducible.
 *
 * All three forms on ONE page: they share the question of what the pool should do
 * next, and the answer usually depends on the other two -- you score the run you
 * just trained, and you precompute the abstraction the next run needs.
 */
export function Dispatch() {
  return (
    <div className="space-y-3">
      <Train />
      <Score />
      <Precompute />
    </div>
  );
}

/** `submit`. Starting and continuing are the same command, and the same form. */
function Train() {
  const configs = useConfigs();
  const runs = useRuns();
  const submit = useSubmit();

  const [config, setConfig] = useState("");
  const [run, setRun] = useState("");
  const [to, setTo] = useState("");
  const [experiment, setExperiment] = useState("");
  const [arm, setArm] = useState("");
  const [parent, setParent] = useState("");
  const [sets, setSets] = useState("");
  const [workers, setWorkers] = useState("");
  const [pool, setPool] = useState("");
  const [kernel, setKernel] = useState("");

  const target = int(to);
  const trainingConfigs = configs.data?.kinds.find((k) => k.kind === "training")?.names ?? [];
  const runIds = (runs.data?.runs ?? []).map((r) => r.name);

  // A fresh run needs a config and a continuation needs a run id; the command
  // refuses either way, but refusing here costs nothing and refusing there
  // costs a round trip to find out.
  const ready = target != null && Boolean(config || run);

  return (
    <Panel
      title="Train — submit"
      // Both: `continue` and `parent` are fed by `runs`, so a failed run list
      // rendered as two empty pickers and read as "there is nothing to continue".
      error={errorOf(configs.error) ?? errorOf(runs.error)}
      onRefresh={() => {
        configs.refetch();
        runs.refetch();
      }}
      refreshing={configs.isFetching || runs.isFetching}
    >
      <div className="divide-y divide-[var(--border)]/50">
        <Field
          label="target"
          hint="ABSOLUTE, not an increment. Re-submitting past it is a no-op, which is what makes a retry converge instead of training twice."
        >
          <Num value={to} onChange={setTo} placeholder="25,000,000" />
        </Field>
        <Field label="config" hint="Fresh runs. Leave empty when continuing one.">
          <Select
            value={config}
            onChange={setConfig}
            options={trainingConfigs}
            disabled={Boolean(run)}
            placeholder={configs.isLoading ? "loading…" : "—"}
          />
        </Field>
        <Field label="continue" hint="Continue this run instead of starting one.">
          <Select value={run} onChange={setRun} options={runIds} />
        </Field>
        <Field
          label="experiment"
          hint="Tagging is what makes `report` see the run at all — an untagged arm is simply absent from the comparison."
        >
          <Text value={experiment} onChange={setExperiment} placeholder="exp-7" />
        </Field>
        <Field
          label="arm"
          hint="Defaults to the submitting branch when an experiment is set. On this surface there is no branch to fall back on, so name it."
        >
          <Text value={arm} onChange={setArm} placeholder="variant:pruning" />
        </Field>
        <Field label="parent" hint="Parent run id, for a fork lineage.">
          <Select value={parent} onChange={setParent} options={runIds} />
        </Field>
        <Field label="overrides" hint="One KEY=VALUE per line (`--set`).">
          <textarea
            value={sets}
            onChange={(event) => setSets(event.target.value)}
            rows={2}
            placeholder="solver__cfr_plus=true"
            className="w-full rounded border border-[var(--border)] bg-transparent px-2 py-1 font-mono text-[12px] outline-none placeholder:text-[var(--fg-faint)] focus:border-[var(--fg-faint)]"
          />
        </Field>
        <Field
          label="pool"
          hint="Blank is train (D16). big = D32, huge = D64 — the fastest box per wall-clock."
        >
          <Select value={pool} onChange={setPool} options={["train", "big", "huge"]} />
        </Field>
        <Field
          label="kernel"
          hint="Blank is scalar MCCFR. pcs = vector kernel on a sampled board per iteration; board-free = bucket-transition vector kernel."
        >
          <Select value={kernel} onChange={setKernel} options={["scalar", "pcs", "board-free"]} />
        </Field>
        <Field
          label="workers"
          hint="Blank is all CPUs. Worth setting below the core count on a big abstraction — every worker loads its own copy."
        >
          <Num value={workers} onChange={setWorkers} placeholder="all CPUs" />
        </Field>
      </div>

      <Actions
        note={ready ? null : "Needs a target, and either a config (fresh) or a run to continue."}
      >
        <Run
          label="Queue training"
          pending={submit.isPending}
          disabled={!ready}
          onClick={() =>
            submit.mutate(
              given({
                to: target,
                config,
                run,
                experiment,
                arm,
                parent,
                sets: overrides(sets),
                workers: int(workers),
                pool,
                kernel,
              }),
            )
          }
        />
      </Actions>
      <Result mutation={submit} />
    </Panel>
  );
}

/** `score`. One task per rung, because rungs are independent. */
function Score() {
  const runs = useRuns();
  const score = useScore();

  const [run, setRun] = useState("");
  const [method, setMethod] = useState("");
  const [at, setAt] = useState("");

  // Only runs that actually have a checkpoint: scoring one that never wrote a
  // ladder costs a pool spin-up to be told so.
  const scorable = (runs.data?.runs ?? []).filter((r) => r.loadable).map((r) => r.name);

  return (
    <Panel
      title="Score — evaluate on the pool"
      error={errorOf(runs.error)}
      onRefresh={() => runs.refetch()}
      refreshing={runs.isFetching}
    >
      <div className="divide-y divide-[var(--border)]/50">
        <Field label="run" hint="Published runs with a checkpoint.">
          <Select
            value={run}
            onChange={setRun}
            options={scorable}
            placeholder={runs.isLoading ? "loading…" : "—"}
          />
        </Field>
        <Field
          label="method"
          hint="Blank is the command's default, exact_br — the zero-variance gate."
        >
          <Select value={method} onChange={setMethod} options={["exact_br", "lbr"]} />
        </Field>
        <Field
          label="rungs"
          hint="Comma-separated ladder iterations; each becomes its own task. Blank scores the latest checkpoint."
        >
          <Text value={at} onChange={setAt} placeholder="10000000, 20000000" />
        </Field>
      </div>

      <Actions note={run ? null : "Pick a run."}>
        <Run
          label="Queue scoring"
          pending={score.isPending}
          disabled={!run}
          onClick={() => score.mutate(given({ run, method, at: rungs(at) }))}
        />
      </Actions>
      <Result mutation={score} />
    </Panel>
  );
}

/** `submit-precompute`. The one dispatch with a guard-rail flag. */
function Precompute() {
  const configs = useConfigs();
  const precompute = usePrecompute();

  const [config, setConfig] = useState("");
  const [workers, setWorkers] = useState("");
  const [force, setForce] = useState(false);

  const names = configs.data?.kinds.find((k) => k.kind === "abstraction")?.names ?? [];

  return (
    <Panel
      title="Precompute — build a card abstraction"
      error={errorOf(configs.error)}
      onRefresh={() => configs.refetch()}
      refreshing={configs.isFetching}
    >
      <div className="divide-y divide-[var(--border)]/50">
        <Field label="config" hint="Abstraction config stem.">
          <Select
            value={config}
            onChange={setConfig}
            options={names}
            placeholder={configs.isLoading ? "loading…" : "—"}
          />
        </Field>
        <Field
          label="workers"
          hint="Blank is all CPUs. An exact-runout abstraction enumerates every canonical board, which dominates the cost."
        >
          <Num value={workers} onChange={setWorkers} placeholder="all CPUs" />
        </Field>
      </div>

      <div className="border-t border-[var(--border)]">
        <Guard
          label="--force"
          because="republish over an existing name. Bucket assignment is not pinned by the abstraction hash, so this silently invalidates the provenance of every run trained against it."
          checked={force}
          onChange={setForce}
        />
      </div>

      <Actions note={config ? null : "Pick a config."}>
        <Run
          label="Queue precompute"
          danger={force}
          pending={precompute.isPending}
          disabled={!config}
          onClick={() => precompute.mutate(given({ config, workers: int(workers), force }))}
        />
      </Actions>
      <Result mutation={precompute} />
    </Panel>
  );
}

/**
 * What the dispatch reported, or why it refused.
 *
 * Kept after a success rather than cleared: the job id and task names are how
 * this submission is found again on the Tasks page, and a message that
 * disappears takes the only record of them with it.
 */
function Result({
  mutation,
}: {
  mutation: UseMutationResult<Dispatched, Error, Record<string, unknown>>;
}) {
  return (
    <Outcome error={errorOf(mutation.error)}>
      {mutation.data && (
        <Queued
          snapshot={mutation.data.code_snapshot}
          job={mutation.data.job_id}
          tasks={mutation.data.tasks}
        />
      )}
    </Outcome>
  );
}
