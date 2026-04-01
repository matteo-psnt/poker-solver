import { useCurve, useRun } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { StatusBadge } from "@/components/StatusBadge";
import { errorOf } from "@/lib/error";
import { count, duration, mbb, percent, rate } from "@/lib/format";
import { getRouteApi } from "@tanstack/react-router";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

const route = getRouteApi("/runs/$runId");

const AXIS = { stroke: "var(--fg-faint)", fontSize: 10 };
const GRID = "var(--border)";

export function RunDetail() {
  const { runId } = route.useParams();
  const run = useRun(runId);
  const curve = useCurve(runId);

  return (
    <div className="space-y-3">
      <Panel
        title={runId}
        updatedAt={run.dataUpdatedAt}
        staleAfterMs={120_000}
        error={errorOf(run.error)}
        loading={run.isLoading}
        onRefresh={() => run.refetch()}
        refreshing={run.isFetching}
      >
        {run.data && (
          <div className="grid grid-cols-2 gap-x-6 gap-y-3 p-3 sm:grid-cols-3 lg:grid-cols-5">
            <Stat label="config" value={run.data.config_name ?? "—"} />
            <Stat label="iterations" value={count(run.data.iterations)} />
            <Stat label="attempts" value={count(run.data.attempts)} />
            <Stat label="compute" value={duration(run.data.runtime_seconds)} />
            <div>
              <Label>status</Label>
              <StatusBadge state={run.data.status} />
            </div>
            <Stat label="abstraction" value={run.data.card_abstraction_hash ?? "—"} mono />
            <Stat label="commit" value={run.data.git_commit?.slice(0, 10) ?? "—"} mono />
          </div>
        )}
      </Panel>

      <div className="grid gap-3 xl:grid-cols-2">
        <Panel
          title="Progress"
          updatedAt={run.dataUpdatedAt}
          staleAfterMs={120_000}
          error={errorOf(run.error)}
          loading={run.isLoading}
          empty={run.data && run.data.progress.length === 0 ? "No checkpoint history." : null}
        >
          {run.data && run.data.progress.length > 0 && (
            <div className="h-56 p-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={run.data.progress}>
                  <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
                  <XAxis
                    dataKey="iteration"
                    tick={AXIS}
                    tickFormatter={(v) => `${(v / 1e6).toFixed(0)}M`}
                  />
                  <YAxis
                    tick={AXIS}
                    domain={[0, 1]}
                    tickFormatter={(v) => percent(v, 0)}
                    width={44}
                  />
                  <Tooltip
                    contentStyle={{ background: "var(--panel)", border: `1px solid ${GRID}` }}
                    formatter={(v: number) => percent(v)}
                    labelFormatter={(v: number) => `${count(v)} it`}
                  />
                  {/* No animation: this redraws on every poll, and a chart that
                      re-animates each time is a chart nobody can read. */}
                  <Line
                    type="monotone"
                    dataKey="coverage"
                    stroke="#3b82f6"
                    dot={false}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
              <p className="px-1 text-[11px] text-[var(--fg-faint)]">
                coverage · latest{" "}
                {rate(run.data.progress[run.data.progress.length - 1]?.iters_per_sec)}
              </p>
            </div>
          )}
        </Panel>

        <Panel
          title="Exploitability"
          updatedAt={curve.dataUpdatedAt}
          staleAfterMs={120_000}
          error={errorOf(curve.error)}
          loading={curve.isLoading}
          empty={curve.data && curve.data.points.length === 0 ? "No scored rungs." : null}
        >
          {curve.data && curve.data.points.length > 0 && (
            <div className="h-56 p-2">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={curve.data.points}>
                  <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
                  <XAxis
                    dataKey="iteration"
                    tick={AXIS}
                    tickFormatter={(v) => `${(v / 1e6).toFixed(0)}M`}
                  />
                  <YAxis tick={AXIS} width={52} />
                  <Tooltip
                    contentStyle={{ background: "var(--panel)", border: `1px solid ${GRID}` }}
                    formatter={(v: number) => mbb(v)}
                    labelFormatter={(v: number) => `${count(v)} it`}
                  />
                  <Line
                    type="monotone"
                    dataKey="exploitability_mbb"
                    stroke="#f59e0b"
                    dot={{ r: 2 }}
                    isAnimationActive={false}
                  />
                </LineChart>
              </ResponsiveContainer>
              <p className="px-1 text-[11px] text-[var(--fg-faint)]">
                mbb/g · tier {curve.data.tier ?? "—"}
                {curve.data.missing_iterations.length > 0 &&
                  ` · ${curve.data.missing_iterations.length} unscored rung(s)`}
              </p>
            </div>
          )}
        </Panel>
      </div>
    </div>
  );
}

function Label({ children }: { children: React.ReactNode }) {
  return (
    <div className="text-[11px] text-[var(--fg-faint)] uppercase tracking-wider">{children}</div>
  );
}

function Stat({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <Label>{label}</Label>
      <div className={mono ? "font-mono text-[12px]" : "tnum"}>{value}</div>
    </div>
  );
}
