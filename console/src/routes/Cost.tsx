import { useCost } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { StepChart } from "@/components/StepChart";
import { errorOf } from "@/lib/error";
import { count, since } from "@/lib/format";
import { useMemo } from "react";

/**
 * Node time, DERIVED from the leg log rather than recorded.
 *
 * The first version sampled the pool from inside this server, which meant it
 * only recorded while the server ran — 3% of a 24h window in practice, so the
 * totals were worthless however honestly they were labelled. Every leg already
 * writes its own start and end to the share, whether or not anything is
 * watching, so the history is complete back to the first leg and there is no
 * coverage caveat to make.
 */
export function Cost() {
  const cost = useCost(0);

  const [times, values] = useMemo(() => {
    const rows = cost.data?.series ?? [];
    return [rows.map((row) => Date.parse(row.at) / 1000), rows.map((row) => row.running)] as [
      number[],
      (number | null)[],
    ];
  }, [cost.data]);

  const data = cost.data;
  const any = (data?.legs ?? 0) > 0;

  return (
    <div className="space-y-3">
      <Panel
        title="Node time · all recorded history"
        updatedAt={cost.dataUpdatedAt}
        staleAfterMs={180_000}
        error={errorOf(cost.error)}
        loading={cost.isLoading}
        empty={data && !any ? "No legs have run yet." : null}
        onRefresh={() => cost.refetch()}
        refreshing={cost.isFetching}
      >
        {data && any && (
          <>
            <div className="grid grid-cols-2 gap-x-6 gap-y-3 p-3 sm:grid-cols-4">
              <Stat
                label="node-hours"
                value={data.task_hours.toFixed(1)}
                note="time legs spent executing"
              />
              <Stat
                label="at Terraform's rate"
                value={data.dollars == null ? "—" : `$${data.dollars.toFixed(2)}`}
                note={
                  data.rate_per_node_hour == null
                    ? "rate unreadable"
                    : `$${data.rate_per_node_hour.toFixed(2)}/node-hr`
                }
              />
              <Stat
                label="legs"
                value={count(data.legs)}
                note={`peak ${data.peak_concurrency} at once`}
              />
              <Stat label="since" value={since(data.first_at)} note="first leg on record" />
            </div>

            <div className="px-2 pb-1">
              <StepChart times={times} values={values} label="legs running" />
            </div>
            <p className="px-3 pb-3 text-[11px] text-[var(--fg-faint)]">
              Legs running concurrently. The area under this curve is the node-hours above.
            </p>
          </>
        )}
      </Panel>

      <p className="max-w-[70ch] px-1 text-[12px] leading-relaxed text-[var(--fg-muted)]">
        A <strong className="font-medium text-[var(--fg)]">lower bound</strong>, not billed cost. It
        counts the time legs were executing, taken from the leg log — so it is complete, but a node
        is allocated a little before its task starts and released after it ends, and pool spin-up is
        not free. The real allocation is somewhat higher, and{" "}
        <code className="rounded bg-white/[0.06] px-1 py-0.5 font-mono text-[var(--fg)]">
          just credit-check
        </code>{" "}
        remains the authority on spend.
      </p>
    </div>
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
