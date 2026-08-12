import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

/**
 * Concurrency over time, drawn as the step function it is.
 *
 * `stepAfter` is the honest shape and the only part of this that is not
 * negotiable: a point means "this many tasks were running from here until the
 * next change", which is a step, not a ramp. A smoothed line would draw node
 * counts that never existed — fractional nodes, between two integers.
 *
 * Why this is Recharts now
 * -----------------------
 * It was uPlot, on a stated argument that turned out to describe code that does
 * not exist: *"the pool is sampled every 15s and kept for 30 days: 172,800
 * points, and Recharts renders one DOM node per point."* There is no sampling.
 * `cost/node_time.timeline` is an EVENT SWEEP — exactly two events per task
 * span, start and end, collapsed where they coincide — so the series is about
 * twice the task count. Hundreds of points, not six figures, and SVG is
 * completely comfortable there.
 *
 * With the arithmetic gone, what was left was one canvas chart needing
 * machinery no SVG chart needs: `lib/theme.ts` existed solely to read CSS
 * custom properties into JS, because a canvas cannot resolve `var(--fg-faint)`
 * and silently falls back to black — which is how these axes once rendered
 * black on a near-black panel. Plus 24 lines of `index.css` overriding uPlot's
 * defaults. That is ongoing complexity for one chart, and it is why the console
 * carried two charting libraries for one page each.
 *
 * If a genuinely long series ever arrives — a sampled pool metric, say — this is
 * the right place to reach for a canvas again. The premise just has to be true.
 */
export function StepChart({
  times,
  values,
  label,
}: {
  /**
   * Epoch MILLISECONDS. Checked against `Cost`, not assumed: it used to divide
   * by 1000 for uPlot, whose `time: true` scale reads Unix seconds unless told
   * `ms: 1`. Recharts and `Date` both want milliseconds, so the division is
   * gone — and a mismatch here is not subtle, it puts every tick in 1970.
   */
  times: number[];
  /**
   * Concurrency, which is a count and never unknown: `timeline()` emits an
   * integer at every event. Typed as such rather than nullable, because this
   * page is emphatic elsewhere that unknown is not zero (`unended` says so
   * directly), and a `?? 0` here would quietly render a gap as an idle pool.
   */
  values: number[];
  label: string;
}) {
  const data = times.map((at, index) => ({ at, running: values[index] }));

  return (
    <div className="h-[200px] w-full">
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: 0 }}>
          <CartesianGrid stroke="var(--border)" strokeDasharray="2 4" />
          <XAxis
            dataKey="at"
            type="number"
            scale="time"
            domain={["dataMin", "dataMax"]}
            tick={{ stroke: "var(--fg-faint)", fontSize: 10 }}
            tickFormatter={(at: number) => new Date(at).toLocaleDateString()}
          />
          <YAxis
            width={44}
            allowDecimals={false}
            tick={{ stroke: "var(--fg-faint)", fontSize: 10 }}
          />
          <Tooltip
            contentStyle={{
              background: "var(--panel)",
              border: "1px solid var(--border)",
              fontSize: 11,
            }}
            labelFormatter={(at: number) => new Date(at).toLocaleString()}
            formatter={(running: number) => [running, label]}
          />
          {/* Filled, because the AREA is the quantity: node-hours is the
              integral of this curve, and the page says so directly underneath. */}
          <Area
            type="stepAfter"
            dataKey="running"
            name={label}
            stroke="#3b82f6"
            fill="#3b82f6"
            fillOpacity={0.15}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
