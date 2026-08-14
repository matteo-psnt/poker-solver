import { useCost } from "@/api/queries";
import type { Billed } from "@/api/types";
import { Panel } from "@/components/Panel";
import { StepChart } from "@/components/StepChart";
import { errorOf } from "@/lib/error";
import { count, since } from "@/lib/format";
import { useMemo } from "react";

/**
 * Two numbers, from two sources, because neither one answers both questions.
 *
 * **Billed** is Azure Cost Management -- the actual invoice, the same source
 * `just credit-check` reads. It is the answer to "what has this cost", and the
 * headline for that reason.
 *
 * **Node time** is derived from the task log. It is the only one of the two that
 * can be attributed to a run, an op, or an exit cause, and it is complete back to
 * the first task because the node wrapper writes a record whether or not anything
 * is watching.
 *
 * Node time multiplied by a rate is NOT a spend estimate and must not be
 * presented as one: it cannot see the ~28% of the bill that is not compute, and
 * no single rate has ever been charged across the SKUs in the history.
 */
export function Cost() {
  const cost = useCost(0);

  // MILLISECONDS, which is what `Date` and Recharts both speak. The `/ 1000`
  // that used to be here existed for uPlot alone: its `time: true` scale reads
  // Unix SECONDS unless told `ms: 1`. Nothing downstream wants seconds now, and
  // leaving the division in would have put every tick in January 1970.
  const [times, values] = useMemo(() => {
    const rows = cost.data?.series ?? [];
    return [rows.map((row) => Date.parse(row.at)), rows.map((row) => row.running)] as [
      number[],
      number[],
    ];
  }, [cost.data]);

  const data = cost.data;
  const any = (data?.tasks ?? 0) > 0;

  return (
    <div className="space-y-3">
      <Panel
        title="Billed · Azure Cost Management"
        updatedAt={cost.dataUpdatedAt}
        staleAfterMs={1_800_000}
        error={errorOf(cost.error)}
        loading={cost.isLoading}
        /* The reason comes from the server, because the two that matter are not
           the same problem: Cost Management throttling means wait and the
           figures are fine, everything else means something is broken. Guessing
           "check az login" locally would send someone to fix an identity that
           was never wrong -- and throttling is the one that actually happens. */
        empty={data && !data.billed ? (data.billed_reason ?? "Billing unavailable.") : null}
        onRefresh={() => cost.refetch()}
        refreshing={cost.isFetching}
      >
        {data?.billed && <BilledPanel billed={data.billed} />}
      </Panel>

      <Panel
        title="Node time · derived from the task log"
        updatedAt={cost.dataUpdatedAt}
        staleAfterMs={180_000}
        error={errorOf(cost.error)}
        loading={cost.isLoading}
        empty={data && !any ? "No tasks have run yet." : null}
        onRefresh={() => cost.refetch()}
        refreshing={cost.isFetching}
      >
        {data && any && (
          <>
            <div className="grid grid-cols-2 gap-x-6 gap-y-3 p-3 sm:grid-cols-4">
              <Stat
                label="node-hours"
                value={data.task_hours.toFixed(1)}
                note="time tasks spent executing"
              />
              <Stat
                label="at the pool rate"
                value={data.dollars == null ? "—" : `$${data.dollars.toFixed(2)}`}
                note={
                  data.rate_per_node_hour == null
                    ? "rate unreadable"
                    : `$${data.rate_per_node_hour.toFixed(3)}/node-hr · execution only`
                }
              />
              <Stat
                label="tasks"
                value={count(data.tasks)}
                note={`peak ${data.peak_concurrency} at once`}
              />
              <Stat label="first task" value={since(data.first_at)} note="start of the record" />
            </div>

            {/* Unknown, not zero. Silently dropping these is what the previous
                version got wrong in the other direction by silently keeping
                them, so the count is stated either way. */}
            {data.unended > 0 && (
              <p className="px-3 pb-2 text-[11px] text-[var(--fg-muted)]">
                {data.unended} attempt{data.unended === 1 ? "" : "s"} started and never recorded an
                end — excluded, because their node time is unknown rather than zero.
              </p>
            )}

            <div className="px-2 pb-1">
              <StepChart times={times} values={values} label="tasks running" />
            </div>
            <p className="px-3 pb-3 text-[11px] text-[var(--fg-faint)]">
              Tasks running concurrently. The area under this curve is the node-hours above.
            </p>
          </>
        )}
      </Panel>

      <p className="max-w-[70ch] px-1 text-[12px] leading-relaxed text-[var(--fg-muted)]">
        Node time counts only the interval a task was{" "}
        <strong className="font-medium text-[var(--fg)]">executing</strong>, so compare it against{" "}
        <strong className="font-medium text-[var(--fg)]">pool</strong> and nothing else: a node is
        allocated before its task starts and released after it ends, and the task log begins after
        the first charges. <strong className="font-medium text-[var(--fg)]">Left running</strong> is
        a separate machine outside the pool — it bills whether or not anything trains, and no task
        log will ever account for it. Billed is the authority on spend; node time is the only half
        that can be attributed to a run.
      </p>
    </div>
  );
}

function BilledPanel({ billed }: { billed: Billed }) {
  const unit = billed.currency === "USD" ? "$" : `${billed.currency} `;
  const money = (value: number) =>
    `${unit}${value.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  return (
    <>
      <div className="grid grid-cols-2 gap-x-6 gap-y-3 p-3 sm:grid-cols-4">
        <Stat
          label="total billed"
          value={money(billed.total)}
          note={billed.first_at ? `since ${billed.first_at}` : undefined}
        />
        <Stat
          label="pool"
          value={money(billed.pool_cost)}
          note={`${billed.pool_node_hours.toLocaleString(undefined, { maximumFractionDigits: 0 })} billed node-hours`}
        />
        {/* Its own stat, never folded into pool compute. A machine left ON bills
            24h a day and no task log will ever mention it — folded in, it made
            node time look like 1.45x of allocation overhead when the pool's real
            figure is 1.19x. */}
        <Stat
          label="left running"
          value={money(billed.standing_cost)}
          note={
            // The biggest one names it; the rest are in the CLI breakdown.
            billed.standing[0]
              ? `${billed.standing_hours.toFixed(0)} h — ${billed.standing[0].resource_group}`
              : "nothing outside the pool"
          }
        />
        <Stat
          label="everything else"
          value={money(billed.other)}
          note={`${Math.round((billed.other / (billed.total || 1)) * 100)}% — storage, disks, network`}
        />
        {/* The freshness caveat is a headline number, not a footnote: cost data
            lags hours and the most recent day always reads low, which invites
            exactly the wrong conclusion about a pool that is running now. */}
        <Stat
          label="complete to"
          value={billed.as_of ?? "—"}
          note="cost data lags and is restated"
        />
      </div>

      <div className="border-t border-[var(--border)] px-3 py-2">
        {billed.by_service.slice(0, 5).map((line) => (
          <div key={line.service} className="flex justify-between py-0.5 text-[12px]">
            <span className="text-[var(--fg-muted)]">{line.service}</span>
            <span className="tnum">{money(line.cost)}</span>
          </div>
        ))}
      </div>
    </>
  );
}

function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div>
      <div className="text-[11px] tracking-wider text-[var(--fg-faint)] uppercase">{label}</div>
      <div className="tnum text-base">{value}</div>
      {note && <div className="text-[11px] text-[var(--fg-faint)]">{note}</div>}
    </div>
  );
}
