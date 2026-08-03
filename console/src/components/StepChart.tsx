import { chartTheme } from "@/lib/theme";
import { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

/**
 * uPlot, not Recharts, and the reason is arithmetic rather than taste.
 *
 * The pool is sampled every 15s and kept for 30 days: 172,800 points. Recharts
 * renders SVG — one DOM node per point — so the page would hang rather than
 * degrade. uPlot draws to canvas and is comfortable well past that.
 *
 * `stepAfter` is the honest shape: a sample means "the pool held N nodes from
 * here until the next observation", which is a step, not a ramp. Interpolating
 * would draw nodes that never existed.
 */
export function StepChart({
  times,
  values,
  label,
}: {
  times: number[];
  values: (number | null)[];
  label: string;
}) {
  const host = useRef<HTMLDivElement>(null);
  const plot = useRef<uPlot | null>(null);

  useEffect(() => {
    const element = host.current;
    if (!element) return;

    const build = () => {
      // Resolved HERE, not written as `var(--…)`: these reach a canvas, which
      // ignores custom properties and falls back to black.
      const theme = chartTheme();
      plot.current?.destroy();
      plot.current = new uPlot(
        {
          width: element.clientWidth || 600,
          height: 200,
          padding: [8, 8, 0, 0],
          cursor: { y: false },
          legend: { show: false },
          scales: { x: { time: true } },
          // `stroke` is the TICK LABEL colour, and uPlot passes it straight to
          // `ctx.fillStyle`. A `var(--…)` there is not an error: the assignment
          // is ignored and uPlot's default `#000` stands, which is how these
          // rendered black on a near-black panel.
          axes: [
            {
              stroke: theme.axis,
              grid: { stroke: theme.grid, width: 1 },
              ticks: { stroke: theme.grid, show: true, size: 4 },
              font: "10px ui-monospace, monospace",
            },
            {
              stroke: theme.axis,
              grid: { stroke: theme.grid, width: 1 },
              ticks: { show: false },
              font: "10px ui-monospace, monospace",
              size: 44,
            },
          ],
          series: [
            {},
            {
              label,
              stroke: theme.series,
              fill: theme.seriesFill,
              width: 1.5,
              paths: uPlot.paths.stepped?.({ align: 1 }),
              points: { show: false },
            },
          ],
        },
        [times, values] as uPlot.AlignedData,
        element,
      );
    };

    build();
    const observer = new ResizeObserver(() => {
      if (element.clientWidth) plot.current?.setSize({ width: element.clientWidth, height: 200 });
    });
    observer.observe(element);
    return () => {
      observer.disconnect();
      plot.current?.destroy();
      plot.current = null;
    };
  }, [times, values, label]);

  return <div ref={host} className="w-full" />;
}
