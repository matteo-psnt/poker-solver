import { useCost } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { StepChart } from "@/components/StepChart";
import { errorOf } from "@/lib/error";
import { duration } from "@/lib/format";
import { useMemo } from "react";

/**
 * What this page is: how many nodes Azure had allocated to the pool, sampled
 * every 15s, integrated to node-hours and priced at Terraform's rate.
 *
 * The number that decides how to read all the others is COVERAGE. The sampler
 * only runs while the console's server runs, so a 24h window is routinely a few
 * observed minutes — and "$2.15 · last 24h" then reads as a daily spend when it
 * is nothing of the sort. Coverage is therefore a headline stat, not a caption,
 * and a thinly-observed window says so in place of the total.
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

  const data = cost.data;
  const recording = (data?.samples ?? 0) > 0;
  const observedH = (data?.observed_seconds ?? 0) / 3600;
  const share = data && data.hours > 0 ? observedH / data.hours : 0;
  const thin = recording && share < 0.5;

  return (
    <div className="space-y-3">
      <Panel
        title="Pool allocation"
        updatedAt={cost.dataUpdatedAt}
        staleAfterMs={60_000}
        error={errorOf(cost.error)}
        loading={cost.isLoading}
        empty={
          data && !recording
            ? "Nothing has been recording. The console's server writes this series while it runs, and Batch keeps no node history — so it cannot be backfilled."
            : null
        }
        onRefresh={() => cost.refetch()}
        refreshing={cost.isFetching}
      >
        {data && recording && (
          <>
            <div className="grid grid-cols-2 gap-x-6 gap-y-3 p-3 sm:grid-cols-4">
              <Stat
                label="node-hours"
                value={data.node_hours.toFixed(2)}
                note="nodes × time allocated"
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
              {/* Third, and deliberately not last: it qualifies the two above. */}
              <Stat
                label="observed"
                value={`${duration(data.observed_seconds)} of ${data.hours}h`}
                note={`${(share * 100).toFixed(0)}% of the window`}
                tone={thin ? "warn" : undefined}
              />
              <Stat
                label="not counted"
                value={duration(data.unobserved_seconds)}
                note="nothing was recording"
                tone={data.unobserved_seconds > 0 ? "warn" : undefined}
              />
            </div>

            {thin && (
              <p className="mx-3 mb-3 rounded border border-amber-500/30 bg-amber-500/[0.06] px-3 py-2 text-[12px] leading-relaxed text-amber-200/90">
                Only <strong>{duration(data.observed_seconds)}</strong> of the last {data.hours}h
                was recorded, so the totals above cover that time — not the day. The sampler runs
                only while this server does.
              </p>
            )}

            <div className="px-2 pb-1">
              <StepChart times={times} values={values} label="nodes" />
            </div>
            <p className="px-3 pb-3 text-[11px] text-[var(--fg-faint)]">
              Nodes allocated over time. Breaks in the line are stretches with no observation — they
              are excluded from the totals rather than filled in.
            </p>
          </>
        )}
      </Panel>

      <p className="max-w-[70ch] px-1 text-[12px] leading-relaxed text-[var(--fg-muted)]">
        This is <strong className="font-medium text-[var(--fg)]">pool allocation</strong>, not
        billed cost: it counts nodes the Batch pool held, priced at the rate in Terraform. It cannot
        see anything Azure bills outside the pool, and it is not an invoice —{" "}
        <code className="rounded bg-white/[0.06] px-1 py-0.5 font-mono text-[var(--fg)]">
          just credit-check
        </code>{" "}
        is the authority.
      </p>
    </div>
  );
}

function Stat({
  label,
  value,
  note,
  tone,
}: {
  label: string;
  value: string;
  note?: string;
  tone?: "warn";
}) {
  return (
    <div>
      <div className="text-[11px] tracking-wider text-[var(--fg-faint)] uppercase">{label}</div>
      <div className={`tnum text-base ${tone === "warn" ? "text-amber-400" : ""}`}>{value}</div>
      {note && <div className="text-[11px] text-[var(--fg-faint)]">{note}</div>}
    </div>
  );
}
