import { Link } from "@tanstack/react-router";
import { Fragment, useMemo, useState } from "react";
import { useGates } from "@/api/queries";
import type { Gate } from "@/api/types";
import { Panel } from "@/components/Panel";
import { Table, Td, Th } from "@/components/Table";
import { errorOf } from "@/lib/error";
import { runLabel, since } from "@/lib/format";
import { cn } from "@/lib/utils";

/** `24k`, `2.3B` — a rung the way the run list counts iterations. */
export function rungLabel(at: number | null | undefined): string {
  if (at == null) return "?";
  for (const [scale, suffix] of [
    [1e9, "B"],
    [1e6, "M"],
    [1e3, "k"],
  ] as const) {
    if (at >= scale) return `${Number((at / scale).toFixed(1))}${suffix}`;
  }
  return String(at);
}

/** Decided means the ±2 SE interval excludes zero, in either direction. */
export function verdictOf(gate: Pick<Gate, "edge_mbb" | "se_mbb">): "win" | "loss" | "open" {
  if (gate.edge_mbb - 2 * gate.se_mbb > 0) return "win";
  if (gate.edge_mbb + 2 * gate.se_mbb < 0) return "loss";
  return "open";
}

const TONE = { win: "text-emerald-400", loss: "text-[#E0655C]", open: "text-[var(--fg-muted)]" };
const BAR = { win: "bg-emerald-400", loss: "bg-[#E0655C]", open: "bg-[var(--fg-faint)]" };

/**
 * Every head-to-head gate in the record: one row per comparison, its seeds
 * pooled by the `gates` command. The number is A's chips per hand against B,
 * the bar is ±2 standard errors around it, and a bar that crosses the centre
 * line has not decided anything. Click a row for its seeds.
 */
export function Results() {
  const gates = useGates();
  const all = gates.data?.gates ?? [];
  const [experiment, setExperiment] = useState("");
  const [decidedOnly, setDecidedOnly] = useState(false);
  const [open, setOpen] = useState<string | null>(null);

  const experiments = useMemo(() => {
    const counts = new Map<string, number>();
    for (const g of all)
      counts.set(g.experiment_id ?? "", (counts.get(g.experiment_id ?? "") ?? 0) + 1);
    return [...counts.entries()].sort((a, b) => b[1] - a[1]);
  }, [all]);
  const shown = all.filter(
    (g) =>
      (!experiment || (g.experiment_id ?? "") === experiment) &&
      (!decidedOnly || verdictOf(g) !== "open"),
  );
  // One axis for the page, so bars compare across rows. Sized to the 90th
  // percentile, not the maximum: one 2,000 mbb outlier flattened every other bar
  // into the centre line. Outliers clip at the edge; their numbers stay exact.
  const extents = shown.map((g) => Math.abs(g.edge_mbb) + 2 * g.se_mbb).sort((a, b) => a - b);
  const scale = Math.max(30, extents[Math.floor(extents.length * 0.9)] ?? 30);
  const pos = (v: number) => `${Math.max(0, Math.min(100, 50 + (v / scale) * 50))}%`;

  return (
    <Panel
      title={`Head-to-head — ${shown.length} gate${shown.length === 1 ? "" : "s"}`}
      aside={
        <span className="flex items-center gap-2 text-[11px]">
          <select
            value={experiment}
            onChange={(e) => setExperiment(e.target.value)}
            aria-label="Experiment"
            className="rounded border border-[var(--border)] bg-[var(--bg,transparent)] px-1.5 py-0.5 text-[var(--fg)]"
          >
            <option value="">every experiment</option>
            {experiments.map(([id, n]) => (
              <option key={id} value={id}>
                {id || "no experiment"} ({n})
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => setDecidedOnly(!decidedOnly)}
            className={cn(
              "rounded px-2 py-0.5",
              decidedOnly
                ? "bg-white/[0.08] text-[var(--fg)]"
                : "text-[var(--fg-muted)] hover:text-[var(--fg)]",
            )}
          >
            decided only
          </button>
          {gates.data && (
            <span className="text-[var(--fg-faint)]">
              {gates.data.rows} match rows, {gates.data.duplicates} re-scored seeds dropped
            </span>
          )}
        </span>
      }
      updatedAt={gates.dataUpdatedAt}
      staleAfterMs={300_000}
      error={errorOf(gates.error)}
      loading={gates.isLoading}
      empty={gates.data && shown.length === 0 ? "No gate matches." : null}
    >
      {shown.length > 0 && (
        <Table>
          <thead>
            <tr>
              <Th>A</Th>
              <Th>against B</Th>
              <Th>
                <span className="flex justify-between">
                  <span>−{Math.round(scale)}</span>
                  <span>0</span>
                  <span>+{Math.round(scale)} mbb/hand</span>
                </span>
              </Th>
              <Th right>edge ± SE</Th>
              <Th right>σ</Th>
              <Th right>seeds</Th>
              <Th>experiment</Th>
              <Th right>last</Th>
            </tr>
          </thead>
          <tbody>
            {shown.map((g) => {
              const key = `${g.run_id}|${g.at}|${g.run_b}|${g.at_b}|${g.thresholds.join("/")}|${g.b_side}`;
              const verdict = verdictOf(g);
              const expanded = open === key;
              return (
                <Fragment key={key}>
                  <tr
                    onClick={() => setOpen(expanded ? null : key)}
                    className="cursor-pointer hover:bg-white/[0.02]"
                  >
                    <Td mono>
                      <Link
                        to="/runs/$runId"
                        params={{ runId: g.run_id }}
                        title={g.run_id}
                        className="hover:underline"
                        onClick={(e) => e.stopPropagation()}
                      >
                        {runLabel(g.run_id)}
                      </Link>
                      <span className="text-[var(--fg-muted)]">@{rungLabel(g.at)}</span>
                    </Td>
                    <Td mono className="text-[var(--fg-muted)]" title={g.run_b ?? undefined}>
                      {g.run_b ? runLabel(g.run_b) : "?"}@{rungLabel(g.at_b)}
                      {g.b_side === "deployed" && (
                        <span className="ml-1.5 text-[10px] text-[var(--fg-faint)]">
                          as deployed
                        </span>
                      )}
                    </Td>
                    <Td className="w-[240px]">
                      <span className="relative block h-3">
                        <span className="absolute inset-y-0 left-1/2 w-px bg-[var(--fg-faint)]" />
                        <span
                          className={cn("absolute top-[5px] h-[2px] opacity-60", BAR[verdict])}
                          style={{
                            left: pos(g.edge_mbb - 2 * g.se_mbb),
                            width: `calc(${pos(g.edge_mbb + 2 * g.se_mbb)} - ${pos(g.edge_mbb - 2 * g.se_mbb)})`,
                          }}
                        />
                        <span
                          className={cn(
                            "absolute top-[2px] size-2 -translate-x-1 rounded-full",
                            BAR[verdict],
                          )}
                          style={{ left: pos(g.edge_mbb) }}
                        />
                      </span>
                    </Td>
                    <Td right className={cn("tnum font-medium", TONE[verdict])}>
                      {g.edge_mbb >= 0 ? "+" : "−"}
                      {Math.abs(g.edge_mbb).toFixed(1)} ± {g.se_mbb.toFixed(1)}
                    </Td>
                    <Td right className="tnum text-[var(--fg-muted)]">
                      {g.sigma == null ? "—" : g.sigma.toFixed(1)}
                    </Td>
                    <Td right className="tnum text-[var(--fg-muted)]">
                      {(g.seeds ?? []).length}
                      {g.duplicates > 0 && (
                        <span
                          className="ml-1 text-amber-400"
                          title="seed rows scored twice; the newest is used"
                        >
                          +{g.duplicates}
                        </span>
                      )}
                    </Td>
                    <Td className="text-[var(--fg-muted)]">{g.experiment_id ?? "—"}</Td>
                    <Td right className="text-[var(--fg-faint)]" title={g.last}>
                      {since(g.last)}
                    </Td>
                  </tr>
                  {expanded && (
                    <tr>
                      <td
                        colSpan={8}
                        className="border-b border-[var(--border)] bg-white/[0.02] px-3 py-2"
                      >
                        <div className="mb-1.5 text-[11px] text-[var(--fg-muted)]">
                          thresholds {g.thresholds.map((t) => t ?? "—").join(" / ")},{" "}
                          {g.deals_per_seed?.toLocaleString() ?? "?"} deals per seed
                        </div>
                        <div className="grid grid-cols-[60px_120px_1fr] gap-x-4 gap-y-0.5 font-mono text-[11px]">
                          {(g.seeds ?? []).map((s) => (
                            <Fragment key={`${s.seed}-${s.task_id}`}>
                              <span className="text-[var(--fg-muted)]">seed {s.seed ?? "?"}</span>
                              <span
                                className={s.edge_mbb >= 0 ? "text-emerald-400" : "text-[#E0655C]"}
                              >
                                {s.edge_mbb >= 0 ? "+" : "−"}
                                {Math.abs(s.edge_mbb).toFixed(1)} ± {s.se_mbb?.toFixed(1) ?? "?"}
                              </span>
                              <span className="text-[var(--fg-faint)]">
                                {s.task_id ? (
                                  <Link
                                    to="/tasks/$taskId"
                                    params={{ taskId: s.task_id }}
                                    className="hover:underline"
                                  >
                                    {s.task_id}
                                  </Link>
                                ) : (
                                  "—"
                                )}
                              </span>
                            </Fragment>
                          ))}
                        </div>
                      </td>
                    </tr>
                  )}
                </Fragment>
              );
            })}
          </tbody>
        </Table>
      )}
    </Panel>
  );
}
