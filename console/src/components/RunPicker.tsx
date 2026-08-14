import { useRuns } from "@/api/queries";
import type { RunSummary } from "@/api/types";
import { count, runLabel } from "@/lib/format";
import { cn } from "@/lib/utils";
import { Check, ChevronDown, Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

/**
 * Choosing a run, wherever the question starts with one.
 *
 * Every question worth asking here -- is it converging, score it, chart it, play
 * it -- starts with "which run", so this is a control rather than a page
 * navigation and a linear scan.
 *
 * Sorted by iteration count rather than name, and searchable: the ids are
 * timestamped stems (`production-1095-...`) that sort alphabetically into an order
 * nobody wants and are too long to scan. What identifies a run to a person is its
 * config, its size, and whether it is the one still running.
 *
 * `loadable` is surfaced rather than filtered out. A run with no checkpoint cannot
 * be charted, played or scored -- but it can still be continued, and hiding it
 * would make an existing run look deleted.
 */
export function RunPicker({
  value,
  onChange,
  label = "run",
  /** Only offer runs with a checkpoint — for charting, playing, scoring. */
  loadableOnly = false,
}: {
  value: string | null;
  onChange: (runId: string) => void;
  label?: string;
  loadableOnly?: boolean;
}) {
  const runs = useRuns();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const box = useRef<HTMLDivElement>(null);
  const search = useRef<HTMLInputElement>(null);

  // Dismiss on an outside click or Escape. Both, because a popover that only
  // closes on one of them is a trap for whichever habit you have.
  useEffect(() => {
    if (!open) return;
    search.current?.focus();
    const onDown = (event: MouseEvent) => {
      if (!box.current?.contains(event.target as Node)) setOpen(false);
    };
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  const matches = useMemo(() => {
    const all = [...(runs.data?.runs ?? [])]
      .filter((run) => !loadableOnly || run.loadable)
      // Biggest first: the run worth looking at is almost always the one with
      // the most iterations, and it was buried in the middle alphabetically.
      .sort((a, b) => (b.iterations ?? 0) - (a.iterations ?? 0));
    const needle = query.trim().toLowerCase();
    if (!needle) return all;
    return all.filter((run) =>
      [run.name, run.config_name, run.experiment_id, run.arm]
        .filter(Boolean)
        .some((field) => String(field).toLowerCase().includes(needle)),
    );
  }, [runs.data, query, loadableOnly]);

  // Guarded past `runs`, not merely past `data`, the way line 64 already is: a
  // 200 whose body carries no `runs` threw HERE and took the whole page's
  // render with it — a blank screen rather than a greyed panel. Read from
  // `runs.data` rather than `matches`, which the typed query filters: the run
  // you have loaded must keep its label while you search for a different one.
  const current = runs.data?.runs?.find((run) => run.name === value) ?? null;

  return (
    <div ref={box} className="relative">
      <button
        type="button"
        onClick={() => setOpen((was) => !was)}
        className="flex items-center gap-2 rounded border border-[var(--border)] px-2.5 py-1.5 font-mono text-[12px] text-[var(--fg)] hover:border-[var(--fg-faint)]"
      >
        <span className="text-[11px] tracking-wider text-[var(--fg-faint)] uppercase">{label}</span>
        <span className="max-w-[22ch] truncate">
          {value ? runLabel(value) : <span className="text-[var(--fg-faint)]">choose…</span>}
        </span>
        {current?.iterations != null && (
          <span className="tabular-nums text-[11px] text-[var(--fg-faint)]">
            {compact(current.iterations)}
          </span>
        )}
        <ChevronDown className="size-3.5 text-[var(--fg-faint)]" />
      </button>

      {open && (
        <div className="absolute z-30 mt-1 w-[30rem] overflow-hidden rounded-md border border-[var(--border)] bg-[var(--panel)] shadow-lg">
          <div className="flex items-center gap-2 border-b border-[var(--border)] px-2.5 py-2">
            <Search className="size-3.5 shrink-0 text-[var(--fg-faint)]" />
            <input
              // Focused via a ref rather than `autoFocus`: the attribute steals
              // focus on MOUNT, which for a popover means whenever React decides
              // to remount it. Here it happens exactly when the list opens.
              ref={search}
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder="run, config, experiment, arm"
              className="w-full bg-transparent font-mono text-[12px] text-[var(--fg)] outline-none placeholder:text-[var(--fg-faint)]"
            />
            <span className="tabular-nums text-[11px] text-[var(--fg-faint)]">
              {matches.length}
            </span>
          </div>

          <ul className="max-h-[22rem] overflow-y-auto">
            {matches.map((run) => (
              <li key={run.name}>
                <Option
                  run={run}
                  selected={run.name === value}
                  onPick={() => {
                    onChange(run.name);
                    setOpen(false);
                    setQuery("");
                  }}
                />
              </li>
            ))}
            {matches.length === 0 && (
              <li className="px-3 py-6 text-center text-[12px] text-[var(--fg-faint)]">
                {runs.isLoading ? "loading runs…" : "No run matches."}
              </li>
            )}
          </ul>
        </div>
      )}
    </div>
  );
}

/**
 * One row. Everything on it answers "is this the run I mean" — the config it
 * came from, how far it got, and its experiment arm when it has one.
 */
function Option({
  run,
  selected,
  onPick,
}: {
  run: RunSummary;
  selected: boolean;
  onPick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onPick}
      className={cn(
        "flex w-full items-center gap-3 px-2.5 py-1.5 text-left hover:bg-white/[0.05]",
        selected && "bg-white/[0.07]",
      )}
    >
      <Check
        className={cn("size-3.5 shrink-0", selected ? "text-[var(--fg)]" : "text-transparent")}
      />
      <span className="min-w-0 flex-1">
        <span className="block truncate font-mono text-[12px] text-[var(--fg)]">
          {runLabel(run.name)}
        </span>
        <span className="block truncate text-[11px] text-[var(--fg-faint)]">
          {run.config_name ?? "—"}
          {run.experiment_id && ` · ${run.experiment_id}`}
          {run.arm && `/${run.arm}`}
          {/* Stated, not hidden: it is why the chart and play surfaces will
              refuse this run, and that is worth knowing before clicking. */}
          {!run.loadable && ` · ${run.blocker ?? "no checkpoint"}`}
        </span>
      </span>
      <span className="shrink-0 text-right">
        <span className="block tabular-nums font-mono text-[12px] text-[var(--fg-muted)]">
          {count(run.iterations)}
        </span>
        <span className="block text-[11px] text-[var(--fg-faint)]">{run.status ?? "—"}</span>
      </span>
    </button>
  );
}

/** `30,000,000` is unreadable inline; `30M` is the same fact at a glance. */
function compact(iterations: number): string {
  if (iterations >= 1e6) return `${(iterations / 1e6).toFixed(0)}M`;
  if (iterations >= 1e3) return `${(iterations / 1e3).toFixed(0)}k`;
  return String(iterations);
}
