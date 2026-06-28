import { getRouteApi, Link } from "@tanstack/react-router";
import { useMemo } from "react";
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
import { useRunView } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { displayName, StatusBadge, toneFor } from "@/components/StatusBadge";
import { Table, Td, Th } from "@/components/Table";
import {
  clock,
  count,
  duration,
  mbb,
  percent,
  rate,
  runLabel,
  since,
  span,
  taskLabel,
} from "@/lib/format";
import { timelineBars } from "@/lib/timeline";
import { cn } from "@/lib/utils";

const route = getRouteApi("/runs/$runId");

/** What `useRunView` returns, so the child panels can be handed it. */
type RunViewQuery = ReturnType<typeof useRunView>;

const AXIS = { stroke: "var(--fg-faint)", fontSize: 10 };
const GRID = "var(--border)";
const TOOLTIP = {
  background: "var(--panel)",
  border: `1px solid ${GRID}`,
  fontSize: 11,
};

const COVERAGE = "#3b82f6";
const VISITS = "#a78bfa";
const EXPLOIT = "#f59e0b";

/** `1.2M`, `140M` — an iteration axis is unreadable in full digits. */
const iterTick = (v: number) =>
  v >= 1e6 ? `${(v / 1e6).toFixed(0)}M` : `${(v / 1e3).toFixed(0)}k`;

/**
 * One run, as ONE request.
 *
 * It used to be four — `runinfo`, `progress`, `curve`, and the ENTIRE task log,
 * which the browser then filtered down to this run's handful of rows. The
 * filtering is now a server-side join (`run_tasks`) and the log itself never
 * crosses the wire, so the page costs what it shows.
 *
 * The parts still fail one at a time: a run with no scored rungs greys the
 * exploitability panel and leaves the rest of the page intact.
 */
export function RunDetail() {
  const { runId } = route.useParams();
  const view = useRunView(runId);
  const parts = view.data?.parts;
  const run = parts?.run;
  const runData = run?.payload;
  const curve = parts?.curve;
  const curveData = curve?.payload;

  return (
    <div className="space-y-3">
      <Panel
        title={runLabel(runId)}
        updatedAt={view.dataUpdatedAt}
        staleAfterMs={120_000}
        error={run?.error ?? null}
        loading={view.isLoading}
        onRefresh={() => view.refresh()}
        refreshing={view.isFetching}
      >
        {runData && (
          <div className="grid grid-cols-2 gap-x-6 gap-y-3 p-3 sm:grid-cols-3 lg:grid-cols-5">
            <Stat label="config" value={runData.config_name ?? "—"} />
            <Stat label="iterations" value={count(runData.iterations)} />
            <Stat label="training tasks" value={count(runData.training_tasks)} />
            <Stat label="compute time" value={duration(runData.runtime_seconds)} />
            <div>
              <Label>status</Label>
              <StatusBadge state={runData.status} />
            </div>
            <Stat label="abstraction" value={runData.card_abstraction_hash ?? "—"} mono />
            <Stat label="commit" value={runData.git_commit?.slice(0, 10) ?? "—"} mono />
          </div>
        )}
      </Panel>

      <div className="grid gap-3 xl:grid-cols-2">
        <ProgressPanel view={view} />
        <Panel
          title="Exploitability"
          updatedAt={view.dataUpdatedAt}
          staleAfterMs={120_000}
          error={curve?.error ?? null}
          loading={view.isLoading}
          empty={curveData && curveData.points.length === 0 ? "No scored rungs." : null}
        >
          {curveData && curveData.points.length > 0 && (
            <Chart
              caption={
                <>
                  mbb/g · tier {curveData.tier ?? "—"} ·{" "}
                  <Coverage
                    have={curveData.points.length}
                    missing={curveData.missing_iterations.length}
                    noun="rung"
                  />
                </>
              }
            >
              <LineChart data={curveData.points}>
                <CartesianGrid stroke={GRID} strokeDasharray="2 4" />
                <XAxis dataKey="iteration" tick={AXIS} tickFormatter={iterTick} />
                <YAxis tick={AXIS} width={52} />
                <Tooltip
                  contentStyle={TOOLTIP}
                  formatter={(v) => mbb(Number(v))}
                  labelFormatter={(v) => `${count(Number(v))} it`}
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

      <RunEvals view={view} />
      <RunTasks view={view} />
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
function ProgressPanel({ view }: { view: RunViewQuery }) {
  const progress = view.data?.parts?.progress;
  const rows = progress?.payload?.rows ?? [];
  const last = rows[rows.length - 1];
  const plateau = progress?.payload?.coverage_plateau_iteration ?? null;

  return (
    <Panel
      title="Progress"
      updatedAt={view.dataUpdatedAt}
      staleAfterMs={120_000}
      error={progress?.error ?? null}
      loading={view.isLoading}
      empty={progress?.payload && rows.length === 0 ? "No checkpoint history." : null}
      onRefresh={() => view.refresh()}
      refreshing={view.isFetching}
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
              formatter={(v, name) =>
                name === "coverage" ? percent(Number(v)) : Number(v).toFixed(1)
              }
              labelFormatter={(v) => `${count(Number(v))} it`}
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

const str = (value: unknown) => (value == null ? "—" : String(value));
const num = (value: unknown) => (typeof value === "number" ? value : null);

/**
 * What this run scored, against the run that scored it.
 *
 * These used to live on a global `/evals` page: every evaluation in the record,
 * one flat list, so answering "is THIS run any good" meant scanning a table
 * mixed with every other run's rows and matching ids by eye. The scores belong
 * to the run — the question is never "what evaluations exist".
 *
 * Filtered by `ledger --run`, which is the command's OWN flag, so this is not a
 * join at all: the server asks the narrower question rather than asking a wide
 * one and discarding most of the answer.
 *
 * The run column is gone because the page IS the run. The knobs stay: an
 * exploitability figure is meaningless without the tier it was measured at, and
 * comparing across tiers is the mistake `compare` refuses by design.
 */
function RunEvals({ view }: { view: RunViewQuery }) {
  const evals = view.data?.parts?.evals;
  const rows = evals?.payload?.rows ?? [];

  return (
    <Panel
      title="Evaluations"
      updatedAt={view.dataUpdatedAt}
      staleAfterMs={120_000}
      error={evals?.error ?? null}
      loading={view.isLoading}
      empty={
        evals?.payload && rows.length === 0
          ? "Never scored. Older evaluations may be legacy files on the share that the rebuild skips."
          : null
      }
      onRefresh={() => view.refresh()}
      refreshing={view.isFetching}
    >
      {rows.length > 0 && (
        <Table>
          <thead>
            <tr>
              <Th>scorer</Th>
              <Th>opponent</Th>
              <Th right>seed</Th>
              <Th right>hands</Th>
              <Th right>mbb/g</Th>
              <Th>commit</Th>
            </tr>
          </thead>
          <tbody>
            {rows.map((row, index) => (
              <tr key={`${row.run_id}-${index}`}>
                <Td>{str(row.knobs?.scorer)}</Td>
                <Td>{str(row.knobs?.opponent)}</Td>
                <Td right className="text-[var(--fg-muted)]">
                  {str(row.knobs?.base_seed)}
                </Td>
                <Td right>{str(row.results?.num_hands)}</Td>
                <Td right>
                  {mbb(num(row.results?.exploitability_mbb), num(row.results?.std_error_mbb))}
                </Td>
                <Td mono className="text-[var(--fg-faint)]">
                  {row.eval_git_commit?.slice(0, 7) ?? "—"}
                </Td>
              </tr>
            ))}
          </tbody>
        </Table>
      )}
    </Panel>
  );
}

const BAR_TONE: Record<string, string> = {
  ok: "bg-emerald-500/60",
  live: "bg-emerald-400/70",
  bad: "bg-red-500/60",
  warn: "bg-amber-500/60",
  pending: "bg-sky-500/50",
  muted: "bg-white/20",
};

/**
 * The run's tasks against one shared time axis.
 *
 * The table below says what each task did; this says how they relate. Gaps
 * between bars are days nothing touched the run, overlapping bars are attempts
 * that ran at once, and a short red bar is a task that died early — none of
 * which is visible reading down a column of timestamps.
 */
function TaskTimeline({ timeline }: { timeline: ReturnType<typeof timelineBars> }) {
  if (!timeline) return null;
  return (
    <div className="space-y-1 px-3 py-2">
      {timeline.bars.map((bar) => (
        <div key={bar.key} className="flex items-center gap-2">
          <div className="w-40 shrink-0 truncate text-[11px] text-[var(--fg-muted)]">
            {bar.label}
          </div>
          <div className="relative h-3 flex-1 rounded-sm bg-[var(--border)]/40">
            <div
              className={cn(
                "absolute inset-y-0 rounded-sm",
                BAR_TONE[bar.tone] ?? BAR_TONE.muted,
                // A running task has no right edge yet, so it is striped rather
                // than drawn as if it had finished at this instant.
                bar.running && "animate-pulse",
              )}
              style={{ left: `${bar.leftPct}%`, width: `${bar.widthPct}%` }}
              title={`${bar.taskId} — ${bar.label}${bar.running ? " (running)" : ""}`}
            />
          </div>
        </div>
      ))}
      <div className="flex justify-between pt-0.5 pl-42 text-[10px] text-[var(--fg-faint)]">
        <span>{new Date(timeline.from).toLocaleString()}</span>
        <span>{new Date(timeline.to).toLocaleString()}</span>
      </div>
    </div>
  );
}

/**
 * The tasks that built this run.
 *
 * Filtered client-side from the full task log rather than read from
 * `runinfo.tasks`, which comes back EMPTY for runs whose records predate that
 * field — including the production run. The task log is the durable account, so
 * it is the one to join against.
 *
 * The join is `task.run_id`, and it crosses daily jobs: a run outlives the job
 * its tasks happen to land in, so grouping by job here would split one lineage
 * across three headings for a reason that is purely about scheduling.
 */
function RunTasks({ view }: { view: RunViewQuery }) {
  const tasks = view.data?.parts?.tasks;
  const now = Date.now();
  // Already filtered to this run, server-side. The page used to download the
  // whole task log and do this itself — over the wire, into the browser, to
  // discard ~95% of it — because `runinfo.tasks` is empty for runs whose records
  // predate that field, the production run among them. The join is the same one;
  // it just happens where the data already is.
  const mine = useMemo(() => [...(view.data?.run_tasks ?? [])].reverse(), [view.data?.run_tasks]);
  const ops = [...new Set(mine.map((row) => row.op).filter(Boolean))];
  const timeline = useMemo(() => timelineBars(mine, now), [mine, now]);

  return (
    <Panel
      title="Tasks"
      updatedAt={view.dataUpdatedAt}
      staleAfterMs={180_000}
      error={tasks?.error ?? null}
      loading={view.isLoading}
      // `tasks.payload` present with an empty join means the log was read and
      // holds nothing for this run. A FAILED part must not render as that — it
      // renders as the error above, because "no tasks" is a claim and "I could
      // not look" is not.
      empty={tasks?.payload && mine.length === 0 ? "No tasks recorded for this run." : null}
      onRefresh={() => view.refresh()}
      refreshing={view.isFetching}
    >
      {mine.length > 0 && (
        <>
          <p className="px-3 pt-2 text-[11px] text-[var(--fg-faint)]">
            {mine.length} attempt{mine.length === 1 ? "" : "s"} built this run
            {ops.length > 0 && ` — ${ops.join(", ")}`}
            {timeline &&
              ` · over ${span(new Date(timeline.from).toISOString(), null, timeline.to)}`}
          </p>
          <TaskTimeline timeline={timeline} />
          <Table>
            <thead>
              <tr>
                <Th>task</Th>
                <Th right>#</Th>
                <Th>what</Th>
                <Th>cause</Th>
                <Th right>started</Th>
                <Th right>took</Th>
                <Th right>ended</Th>
              </tr>
            </thead>
            <tbody>
              {mine.map((row) => (
                <tr key={`${row.task_id}-${row.attempt}`}>
                  <Td mono>
                    <Link
                      to="/tasks/$taskId"
                      params={{ taskId: row.task_id }}
                      title={row.task_id}
                      className="hover:underline"
                    >
                      {taskLabel(row.task_id)}
                    </Link>
                  </Td>
                  <Td right className="text-[var(--fg-faint)]">
                    {row.attempt ?? "—"}
                  </Td>
                  <Td className="text-[var(--fg-muted)]">{row.what || row.op || "—"}</Td>
                  <Td>
                    <StatusBadge
                      state={displayName(row.cause)}
                      tone={toneFor(row.cause)}
                      title={`recorded as "${row.cause}"${
                        row.exit_code == null ? "" : `, exit ${row.exit_code}`
                      }`}
                    />
                  </Td>
                  <Td right className="text-[var(--fg-faint)]" title={row.started_at ?? undefined}>
                    {clock(row.started_at)}
                  </Td>
                  <Td right className="tnum text-[var(--fg-muted)]">
                    {span(row.started_at, row.ended_at, now)}
                  </Td>
                  <Td right className="text-[var(--fg-faint)]" title={row.ended_at ?? undefined}>
                    {row.ended_at ? since(row.ended_at) : "—"}
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
