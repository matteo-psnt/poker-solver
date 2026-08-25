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
 * negotiable: a point means "this many tasks were running from here until the next
 * change", which is a step, not a ramp. A smoothed line would draw node counts
 * that never existed -- fractional nodes, between two integers.
 *
 * The series is short. `cost/node_time.timeline` is an EVENT SWEEP, two events per
 * task span collapsed where they coincide, so it is about twice the task count --
 * hundreds of points, which SVG is comfortable with. If a genuinely long series
 * ever arrives, a sampled pool metric say, that is when to reach for a canvas
 * again.
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
            // recharts 3 hands the formatters `ReactNode`/`ValueType`, not the
            // datum's own type -- the coercion is where that widening stops.
            labelFormatter={(at) => new Date(Number(at)).toLocaleString()}
            formatter={(running) => [Number(running), label]}
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
