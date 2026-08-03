import { useCost } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { StepChart } from "@/components/StepChart";
import { errorOf } from "@/lib/error";
import { duration, since } from "@/lib/format";
import { useMemo } from "react";

/**
 * The only page reading something this project RECORDS rather than asks.
 *
 * Batch retains no node history, so the series begins when the console's server
 * first ran and can never be backfilled. The page says so rather than showing an
 * empty chart that looks like an idle pool.
 */
export function Cost() {
  const cost = useCost(24);

  const [times, values] = useMemo(() => {
    const rows = cost.data?.series ?? [];
    return [
      rows.map((row) => Date.parse(row.at) / 1000),
      rows.map((row) => (typeof row.nodes === "number" ? row.nodes : null)),
    ] as [number[], (number | null)[]];
  }, [cost.data]);

  const recording = (cost.data?.samples ?? 0) > 0;

  return (
    <div className="space-y-3">
      <Panel
        title="Pool allocation · last 24h"
        updatedAt={cost.dataUpdatedAt}
        staleAfterMs={60_000}
        error={errorOf(cost.error)}
        loading={cost.isLoading}
        empty={
          cost.data && !recording
            ? "Nothing has been recording. The console's server writes this series while it runs, and Batch keeps no node history — so it cannot be backfilled."
            : null
        }
        onRefresh={() => cost.refetch()}
        refreshing={cost.isFetching}
      >
        {cost.data && recording && (
          <>
            <div className="grid grid-cols-2 gap-x-6 gap-y-2 p-3 sm:grid-cols-4">
              <Stat label="node-hours" value={cost.data.node_hours.toFixed(2)} />
              <Stat
                label="at rate"
                value={
                  cost.data.dollars == null ? "rate unknown" : `$${cost.data.dollars.toFixed(2)}`
                }
              />
              <Stat label="samples" value={cost.data.samples.toLocaleString("en-US")} />
              <Stat label="since" value={since(cost.data.first_at)} />
            </div>
            <div className="px-2 pb-2">
              <StepChart times={times} values={values} label="nodes" />
            </div>
          </>
        )}
      </Panel>

      {/* A caption, not a footnote: this number invites being quoted. */}
      <p className="max-w-[70ch] px-1 text-[12px] leading-relaxed text-[var(--fg-muted)]">
        This is <strong className="font-medium text-[var(--fg)]">pool allocation</strong>, not
        billed cost. It cannot see anything Azure bills outside the pool, and the rate comes from
        Terraform rather than an invoice —{" "}
        <code className="rounded bg-white/[0.06] px-1 py-0.5 font-mono text-[var(--fg)]">
          just credit-check
        </code>{" "}
        is the authority.
        {cost.data && cost.data.unobserved_seconds > 0 && (
          <>
            {" "}
            <span className="text-amber-400">
              {duration(cost.data.unobserved_seconds)} of the window is unobserved
            </span>{" "}
            — nothing was recording then, and it is excluded from the total rather than billed at
            whatever was last seen.
          </>
        )}
      </p>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[11px] tracking-wider text-[var(--fg-faint)] uppercase">{label}</div>
      <div className="tnum text-base">{value}</div>
    </div>
  );
}
