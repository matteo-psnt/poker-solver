import { useCurve, useLegs, useProgress, useRun } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { StatusBadge, displayName, toneFor } from "@/components/StatusBadge";
import { Table, Td, Th } from "@/components/Table";
import { errorOf } from "@/lib/error";
import { count, duration, legLabel, mbb, percent, rate, runLabel, since } from "@/lib/format";
import { Link, getRouteApi } from "@tanstack/react-router";
import {
  CartesianGrid,
  Legend,
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
const TOOLTIP = { background: "var(--panel)", border: `1px solid ${GRID}`, fontSize: 11 };

const COVERAGE = "#3b82f6";
const VISITS = "#a78bfa";
const EXPLOIT = "#f59e0b";

/** `1.2M`, `140M` — an iteration axis is unreadable in full digits. */
const iterTick = (v: number) =>
  v >= 1e6 ? `${(v / 1e6).toFixed(0)}M` : `${(v / 1e3).toFixed(0)}k`;

export function RunDetail() {
  const { runId } = route.useParams();
  const run = useRun(runId);
  const curve = useCurve(runId);

  return (
    <div className="space-y-3">
      <Panel
        title={runLabel(runId)}
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
            <Stat label="compute time" value={duration(run.data.runtime_seconds)} />
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
        <ProgressPanel runId={runId} />
        <Panel
          title="Exploitability"
          updatedAt={curve.dataUpdatedAt}
          staleAfterMs={120_000}
          error={errorOf(curve.error)}
          loading={curve.isLoading}
          empty={curve.data && curve.data.points.length === 0 ? "No scored rungs." : null}
        >
          {curve.data && curve.data.points.length > 0 && (
            <Chart
              caption={
                <>
                  mbb/g · tier {curve.data.tier ?? "—"} ·{" "}
                  <Coverage
                    have={curve.data.points.length}
                    missing={curve.data.missing_iterations.length}
                    noun="rung"
                  />
                </>
              }
            >
              <LineChart data={curve.data.points}>
                <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
                <XAxis dataKey="iteration" tick={AXIS} tickFormatter={iterTick} />
                <YAxis tick={AXIS} width={52} />
                <Tooltip
                  contentStyle={TOOLTIP}
                  formatter={(v: number) => mbb(v)}
                  labelFormatter={(v: number) => `${count(v)} it`}
                />
                <Line
                  type="monotone"
                  dataKey="exploitability_mbb"
                  stroke={EXPLOIT}
                  dot={{ r: 2 }}
                  isAnimationActive={false}
                />
              </LineChart>
            </Chart>
          )}
        </Panel>
      </div>

      <RunLegs runId={runId} />
    </div>
  );
}

/**
 * The frame every chart on this page sits in.
 *
 * The caption is a flex SIBLING of the plot, not a block after it. Recharts'
 * `ResponsiveContainer` measures `height="100%"` against the parent, so putting
 * it and a caption in one fixed-height box makes the content taller than the
 * box and the caption spills past the panel's border. `min-h-0` is what lets
 * the plot shrink to whatever the caption leaves — without it a flex child
 * refuses to go below its content size and the same overflow comes back.
 */
function Chart({ children, caption }: { children: React.ReactElement; caption: React.ReactNode }) {
  return (
    <div className="flex h-64 flex-col gap-1 p-2">
      <div className="min-h-0 flex-1">
        <ResponsiveContainer width="100%" height="100%">
          {children}
        </ResponsiveContainer>
      </div>
      <p className="px-1 text-[11px] text-[var(--fg-faint)]">{caption}</p>
    </div>
  );
}

/**
 * How much of the series is actually plotted.
 *
 * A sparse line and a complete one look identical once drawn, so a chart that
 * silently omits most of its points reads as a finished measurement. Saying
 * "10 of 150" is the difference between "this is the answer" and "this needs
 * backfilling".
 */
function Coverage({ have, missing, noun }: { have: number; missing: number; noun: string }) {
  const total = have + missing;
  if (missing === 0) return <>all {count(have)} scored</>;
  return (
    <span className="text-amber-400/80">
      {count(have)} of {count(total)} {noun}s scored — {count(missing)} unscored
    </span>
  );
}

/**
 * Per-checkpoint history, read from `progress` rather than `runinfo`.
 *
 * `runinfo` carries a progress array too, but it is truncated to its `--last`
 * default of EIGHT — so this panel drew 8 of 112 checkpoints and looked like a
 * complete history. `progress` with `last=0` is the full series.
 *
 * Two lines, because coverage alone says nothing: it saturates at 100% early
 * (this run plateaued at 40M of 150M) and is flat for the rest of the run.
 * `mean_visits_per_touched` is the convergence diagnostic — it is the quantity
 * to compare against the 1e3-1e4 regret updates per infoset CFR needs.
 */
function ProgressPanel({ runId }: { runId: string }) {
  const progress = useProgress(runId);
  const rows = progress.data?.rows ?? [];
  const last = rows[rows.length - 1];
  const plateau = progress.data?.coverage_plateau_iteration ?? null;

  return (
    <Panel
      title="Progress"
      updatedAt={progress.dataUpdatedAt}
      staleAfterMs={120_000}
      error={errorOf(progress.error)}
      loading={progress.isLoading}
      empty={progress.data && rows.length === 0 ? "No checkpoint history." : null}
      onRefresh={() => progress.refetch()}
      refreshing={progress.isFetching}
    >
      {rows.length > 0 && (
        <Chart
          caption={
            <>
              {count(rows.length)} checkpoints · latest {rate(last?.iters_per_sec)} ·{" "}
              {last?.mean_visits_per_touched != null && (
                <>visits/infoset {last.mean_visits_per_touched.toFixed(1)} · </>
              )}
              {plateau != null ? (
                <>coverage flat from {iterTick(plateau)}</>
              ) : (
                <>coverage still climbing</>
              )}
            </>
          }
        >
          <LineChart data={rows}>
            <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
            <XAxis dataKey="iteration" tick={AXIS} tickFormatter={iterTick} />
            <YAxis
              yAxisId="coverage"
              tick={AXIS}
              domain={[0, 1]}
              tickFormatter={(v) => percent(v, 0)}
              width={44}
            />
            <YAxis
              yAxisId="visits"
              orientation="right"
              tick={AXIS}
              width={40}
              tickFormatter={(v) => v.toFixed(0)}
            />
            <Tooltip
              contentStyle={TOOLTIP}
              formatter={(v: number, name: string) =>
                name === "coverage" ? percent(v) : v.toFixed(1)
              }
              labelFormatter={(v: number) => `${count(v)} it`}
            />
            <Legend wrapperStyle={{ fontSize: 10 }} iconSize={8} />
            {/* No animation: this redraws on every poll, and a chart that
                re-animates each time is a chart nobody can read. */}
            <Line
              yAxisId="coverage"
              type="monotone"
              dataKey="coverage"
              stroke={COVERAGE}
              dot={false}
              isAnimationActive={false}
            />
            <Line
              yAxisId="visits"
              type="monotone"
              dataKey="mean_visits_per_touched"
              name="visits/infoset"
              stroke={VISITS}
              dot={false}
              isAnimationActive={false}
            />
          </LineChart>
        </Chart>
      )}
    </Panel>
  );
}

/**
 * The legs that built this run.
 *
 * Filtered client-side from the full leg log rather than read from
 * `runinfo.legs`, which comes back EMPTY for runs whose records predate that
 * field — including the production run. The leg log is the durable account, so
 * it is the one to join against.
 *
 * The join is `leg.run_id`, and it crosses daily jobs: a run outlives the job
 * its legs happen to land in, so grouping by job here would split one lineage
 * across three headings for a reason that is purely about scheduling.
 */
function RunLegs({ runId }: { runId: string }) {
  const legs = useLegs(0);
  const mine = (legs.data?.rows ?? []).filter((row) => row.run_id === runId).reverse();
  const ops = [...new Set(mine.map((row) => row.op).filter(Boolean))];

  return (
    <Panel
      title="Legs"
      updatedAt={legs.dataUpdatedAt}
      staleAfterMs={180_000}
      error={errorOf(legs.error)}
      loading={legs.isLoading}
      empty={legs.data && mine.length === 0 ? "No legs recorded for this run." : null}
      onRefresh={() => legs.refetch()}
      refreshing={legs.isFetching}
    >
      {mine.length > 0 && (
        <>
          <p className="px-3 pt-2 text-[11px] text-[var(--fg-faint)]">
            {mine.length} attempt{mine.length === 1 ? "" : "s"} built this run
            {ops.length > 0 && ` — ${ops.join(", ")}`}
          </p>
          <Table>
            <thead>
              <tr>
                <Th>task</Th>
                <Th right>#</Th>
                <Th>op</Th>
                <Th>cause</Th>
                <Th right>exit</Th>
                <Th right>ended</Th>
              </tr>
            </thead>
            <tbody>
              {mine.map((row) => (
                <tr key={`${row.task_id}-${row.attempt}`}>
                  <Td mono>
                    <Link
                      to="/legs/$taskId"
                      params={{ taskId: row.task_id }}
                      title={row.task_id}
                      className="hover:underline"
                    >
                      {legLabel(row.task_id)}
                    </Link>
                  </Td>
                  <Td right className="text-[var(--fg-faint)]">
                    {row.attempt ?? "—"}
                  </Td>
                  <Td className="text-[var(--fg-muted)]">{row.op ?? "—"}</Td>
                  <Td>
                    <StatusBadge
                      state={displayName(row.cause)}
                      tone={toneFor(row.cause)}
                      title={`recorded as "${row.cause}"`}
                    />
                  </Td>
                  <Td right>{row.exit_code ?? "—"}</Td>
                  <Td right className="text-[var(--fg-faint)]">
                    {since(row.ended_at)}
                  </Td>
                </tr>
              ))}
            </tbody>
          </Table>
        </>
      )}
    </Panel>
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
