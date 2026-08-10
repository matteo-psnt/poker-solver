import { useActivity } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { Table, Td, Th } from "@/components/Table";
import { errorOf } from "@/lib/error";
import { count, duration, since } from "@/lib/format";
import { cn } from "@/lib/utils";
import { useState } from "react";

/**
 * What this tool has been costing itself.
 *
 * The only page here that is about the console and the CLI rather than about
 * the solver or the pool. It reads the local activity log — one row per command
 * that ran, from either surface — which exists because `test_read_cost.py`
 * PREVENTS a latency regression but nothing NOTICES one. Every number in that
 * test was originally found by someone timing a screen by hand.
 *
 * Sorted by total time, not by p95. The question this page answers is "what is
 * worth fixing", and a command taking 0.4s three thousand times outranks one
 * taking nine seconds twice — which sorting by the tail gets exactly backwards.
 */
const WINDOWS = [1, 7, 30, 0] as const;

export function Activity() {
  const [days, setDays] = useState<number>(7);
  const activity = useActivity(days);
  const data = activity.data;

  return (
    <div className="space-y-3">
      <Panel
        title="Activity — what the commands cost"
        updatedAt={activity.dataUpdatedAt}
        staleAfterMs={120_000}
        error={errorOf(activity.error)}
        loading={activity.isLoading}
        empty={data && !data.exists ? emptyReason(data.enabled) : null}
        onRefresh={() => activity.refetch()}
        refreshing={activity.isFetching}
      >
        <div className="flex items-center gap-3 border-b border-[var(--border)] px-3 py-1.5">
          <span className="text-[11px] text-[var(--fg-faint)] uppercase tracking-wider">
            window
          </span>
          {WINDOWS.map((window) => (
            <button
              key={window}
              type="button"
              onClick={() => setDays(window)}
              className={cn(
                "rounded px-2 py-0.5 font-mono text-[11px]",
                days === window
                  ? "bg-white/[0.08] text-[var(--fg)]"
                  : "text-[var(--fg-muted)] hover:text-[var(--fg)]",
              )}
            >
              {window === 0 ? "all" : `${window}d`}
            </button>
          ))}
          {data?.exists && (
            <span className="ml-auto font-mono text-[11px] text-[var(--fg-faint)]">
              {count(data.rows)} of {count(data.total_rows)} rows ·{" "}
              {Object.entries(data.by_surface)
                .map(([surface, calls]) => `${surface} ${calls}`)
                .join(" · ")}
            </span>
          )}
        </div>

        {data?.exists &&
          (data.commands.length === 0 ? (
            <p className="px-3 py-6 text-center text-[var(--fg-faint)]">Nothing in this window.</p>
          ) : (
            <Table>
              <thead>
                <tr>
                  <Th>command</Th>
                  <Th right>calls</Th>
                  <Th right>p50</Th>
                  <Th right>p95</Th>
                  <Th right>max</Th>
                  <Th right>total</Th>
                  <Th right>refused</Th>
                  <Th right>errors</Th>
                </tr>
              </thead>
              <tbody>
                {data.commands.map((entry) => (
                  <tr key={entry.command}>
                    <Td mono>{entry.command}</Td>
                    <Td right className="text-[var(--fg-muted)]">
                      {count(entry.calls)}
                    </Td>
                    <Td right>{entry.p50_seconds.toFixed(2)}s</Td>
                    {/* The tail is the half someone is complaining about, so it
                        is the one that gets marked when it separates from p50. */}
                    <Td
                      right
                      className={cn(entry.p95_seconds > 4 * entry.p50_seconds && "text-amber-400")}
                    >
                      {entry.p95_seconds.toFixed(2)}s
                    </Td>
                    <Td right className="text-[var(--fg-faint)]">
                      {entry.max_seconds.toFixed(2)}s
                    </Td>
                    <Td right>{duration(entry.total_seconds)}</Td>
                    {/* A refusal is not a fault — the command understood and the
                        answer was no — so it is never coloured as one. */}
                    <Td right className="text-[var(--fg-muted)]">
                      {entry.refusals || "—"}
                    </Td>
                    <Td right className={cn(entry.errors > 0 && "text-[#E0655C]")}>
                      {entry.errors || "—"}
                    </Td>
                  </tr>
                ))}
              </tbody>
            </Table>
          ))}
      </Panel>

      {data && data.failures.length > 0 && (
        <Panel title={`Failures — ${data.failures.length}`}>
          <Table>
            <thead>
              <tr>
                <Th>when</Th>
                <Th>command</Th>
                <Th>from</Th>
                <Th>kind</Th>
                <Th>reason</Th>
              </tr>
            </thead>
            <tbody>
              {data.failures.map((failure, index) => (
                <tr key={`${failure.at}-${failure.command}-${index}`}>
                  <Td className="text-[var(--fg-faint)]" title={failure.at ?? undefined}>
                    {failure.at ? since(failure.at) : "—"}
                  </Td>
                  <Td mono>{failure.command ?? "—"}</Td>
                  <Td className="text-[var(--fg-muted)]">{failure.surface ?? "—"}</Td>
                  <Td
                    className={
                      failure.outcome === "refusal" ? "text-[var(--fg-muted)]" : "text-[#E0655C]"
                    }
                    title={failure.error_type ?? undefined}
                  >
                    {failure.outcome ?? "—"}
                  </Td>
                  <Td className="text-[var(--fg-muted)]">
                    {failure.error || failure.error_type || "—"}
                    {/* What was ASKED is usually the whole diagnosis: which run,
                        which config. Without it a list of refusals is just a
                        count. */}
                    {Object.keys(failure.asked).length > 0 && (
                      <span className="ml-2 font-mono text-[11px] text-[var(--fg-faint)]">
                        {Object.entries(failure.asked)
                          .map(([key, value]) => `${key}=${String(value)}`)
                          .join(" ")}
                      </span>
                    )}
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </Panel>
      )}
    </div>
  );
}

/** Two empty states with different fixes; collapsing them sends you hunting. */
function emptyReason(enabled: boolean): string {
  return enabled
    ? "No commands recorded yet on this machine."
    : "Recording is switched off (POKER_SOLVER_TELEMETRY).";
}
