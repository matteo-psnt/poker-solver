import { useBlueprintRun, useCombos, useSolverNode } from "@/api/queries";
import { Panel } from "@/components/Panel";
import { RangeGrid } from "@/components/RangeGrid";
import { type Cell, actionColour, aggregate } from "@/lib/range";
import { cn } from "@/lib/utils";
import { useMemo, useState } from "react";

/**
 * Walk the tree, and see what the blueprint plays with every hand at each spot.
 *
 * A spot is addressed by the LINE that reaches it, never by a node id — a path
 * survives a retrain and an id does not, so a bookmarked spot stays the spot it
 * was. The board is typed rather than dealt for the same reason: bucketing is a
 * function of the board, so the strategy is not defined until you name one, and
 * a runout the page chose would answer a different question on every load.
 */
export function Solver() {
  const [path, setPath] = useState("");
  const [board, setBoard] = useState("");
  const [average, setAverage] = useState(true);
  const [hovered, setHovered] = useState<Cell | null>(null);

  const run = useBlueprintRun();
  const configured = run.error === null || run.error === undefined;
  const combos = useCombos(!!run.data);
  const node = useSolverNode(path, board, average, !!run.data);

  const grid = node.data?.grid ?? null;
  const cells = useMemo(() => {
    if (!grid || !combos.data) return null;
    return aggregate({
      combos: combos.data.combos,
      comboBuckets: grid.combo_buckets,
      buckets: grid.buckets,
      actionCount: grid.actions.length,
    });
  }, [grid, combos.data]);

  const steps = path ? path.split("/") : [];

  return (
    <div className="space-y-3 p-3">
      <Panel
        title="blueprint"
        error={run.error ? String(run.error.message ?? run.error) : null}
        updatedAt={run.dataUpdatedAt}
        staleAfterMs={Number.POSITIVE_INFINITY}
      >
        {run.data ? (
          <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1 px-3 py-2 font-mono text-[12px]">
            <span className="text-[var(--fg)]">{run.data.run}</span>
            <span className="text-[var(--fg-muted)]">
              stack {run.data.starting_stack} · blinds {run.data.small_blind}/{run.data.big_blind}
            </span>
          </div>
        ) : configured ? undefined : (
          <div className="px-3 py-2 text-[12px] text-[var(--fg-muted)]">
            No run is loaded. Start a blueprint server and point{" "}
            <code className="font-mono">POKER_SOLVER_BLUEPRINT_URL</code> at it.
          </div>
        )}
      </Panel>

      <Panel title="line">
        <div className="space-y-2 px-3 py-2">
          <div className="flex flex-wrap items-center gap-1">
            <Crumb label="start" onClick={() => setPath("")} active={steps.length === 0} />
            {steps.map((step, index) => (
              <Crumb
                // Position matters: the same token can repeat in one line.
                key={`${index}-${step}`}
                label={step}
                onClick={() => setPath(steps.slice(0, index + 1).join("/"))}
                active={index === steps.length - 1}
              />
            ))}
          </div>

          <div className="flex flex-wrap items-center gap-2">
            {node.data?.children.map((child) => (
              <button
                key={child.token}
                type="button"
                onClick={() => setPath(path ? `${path}/${child.token}` : child.token)}
                className="rounded border border-[var(--border)] px-2 py-1 font-mono text-[11px] text-[var(--fg-muted)] hover:bg-white/[0.06] hover:text-[var(--fg)]"
              >
                {child.token}
              </button>
            ))}
            {node.data?.terminal && (
              <span className="font-mono text-[11px] text-[var(--fg-muted)]">
                the hand ends here
              </span>
            )}
          </div>

          <div className="flex flex-wrap items-center gap-3 pt-1">
            <label className="flex items-center gap-2 font-mono text-[11px] text-[var(--fg-muted)]">
              board
              <input
                value={board}
                onChange={(event) => setBoard(event.target.value)}
                placeholder="2c7d9h"
                spellCheck={false}
                className="w-32 rounded border border-[var(--border)] bg-transparent px-2 py-1 font-mono text-[11px] text-[var(--fg)]"
              />
            </label>
            <label className="flex items-center gap-2 font-mono text-[11px] text-[var(--fg-muted)]">
              <input
                type="checkbox"
                checked={average}
                onChange={(event) => setAverage(event.target.checked)}
              />
              average strategy
            </label>
            {/* Not a detail: the average is the blueprint and what converges;
                the current strategy is regret-matching's latest guess, and on
                an under-trained run they disagree sharply. */}
            <span className="text-[11px] text-[var(--fg-faint)]">
              {average ? "the blueprint proper" : "regret-matched current strategy"}
            </span>
          </div>
        </div>
      </Panel>

      <Panel
        title={grid ? `range · ${grid.street}` : "range"}
        error={node.error ? String(node.error.message ?? node.error) : null}
        loading={node.isFetching && !node.data}
        updatedAt={node.dataUpdatedAt}
        staleAfterMs={Number.POSITIVE_INFINITY}
        empty={node.data?.terminal ? "No strategy at a spot where nobody acts." : null}
      >
        {cells && grid ? (
          <div className="grid gap-3 px-3 py-3 lg:grid-cols-[minmax(0,1fr)_220px]">
            <RangeGrid cells={cells} actions={grid.actions} onHover={setHovered} />
            <aside className="space-y-3 font-mono text-[11px]">
              <div className="space-y-1">
                {grid.actions.map((token, index) => (
                  <div key={token} className="flex items-center gap-2">
                    <span
                      className="size-2.5 rounded-[2px]"
                      style={{
                        backgroundColor: actionColour(token, index, grid.actions.length),
                      }}
                    />
                    <span className="text-[var(--fg-muted)]">{token}</span>
                  </div>
                ))}
              </div>

              <div className="space-y-1 border-t border-[var(--border)] pt-2 text-[var(--fg-muted)]">
                <Stat label="acting" value={`seat ${grid.actor}`} />
                <Stat label="buckets" value={String(Object.keys(grid.buckets).length)} />
                {/* The number most worth reading: a bucket the solver never
                    visited has no strategy, only an allocated row. */}
                <Stat
                  label="trained"
                  value={`${grid.trained_buckets}/${Object.keys(grid.buckets).length}`}
                />
                <Stat label="blocked" value={`${grid.blocked} combos`} />
              </div>

              <div className="min-h-[72px] border-t border-[var(--border)] pt-2">
                {hovered ? (
                  <Detail cell={hovered} actions={grid.actions} />
                ) : (
                  <span className="text-[var(--fg-faint)]">hover a hand</span>
                )}
              </div>
            </aside>
          </div>
        ) : undefined}
      </Panel>
    </div>
  );
}

function Crumb({
  label,
  onClick,
  active,
}: {
  label: string;
  onClick: () => void;
  active: boolean;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        "rounded px-1.5 py-0.5 font-mono text-[11px]",
        active
          ? "bg-white/[0.09] text-[var(--fg)]"
          : "text-[var(--fg-muted)] hover:text-[var(--fg)]",
      )}
    >
      {label}
    </button>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-[var(--fg-faint)]">{label}</span>
      <span className="tabular-nums text-[var(--fg)]">{value}</span>
    </div>
  );
}

function Detail({ cell, actions }: { cell: Cell; actions: string[] }) {
  if (cell.combos === 0) {
    return (
      <div className="space-y-1">
        <div className="text-[var(--fg)]">{cell.label}</div>
        <div className="text-[var(--fg-faint)]">blocked by the board</div>
      </div>
    );
  }
  return (
    <div className="space-y-1">
      <div className="text-[var(--fg)]">
        {cell.label} <span className="text-[var(--fg-faint)]">{cell.combos} combos</span>
      </div>
      {cell.strategy ? (
        cell.strategy.map((weight, index) => (
          <div key={actions[index] ?? index} className="flex justify-between gap-2">
            <span className="text-[var(--fg-muted)]">{actions[index]}</span>
            <span className="tabular-nums text-[var(--fg)]">{(weight * 100).toFixed(1)}%</span>
          </div>
        ))
      ) : (
        <div className="text-[var(--fg-faint)]">never trained here</div>
      )}
      {cell.untrained > 0 && cell.strategy && (
        <div className="text-[var(--fg-faint)]">
          averaged over {cell.combos - cell.untrained} of {cell.combos}
        </div>
      )}
    </div>
  );
}
