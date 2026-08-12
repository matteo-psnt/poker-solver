import { useBlueprintRun, useLoadBlueprintRun } from "@/api/queries";
import { BoxControl } from "@/components/BoxControl";
import { Panel } from "@/components/Panel";
import { RunPicker } from "@/components/RunPicker";
import { Tabs } from "@/components/Tabs";
import { runLabel } from "@/lib/format";
import { getRouteApi, useNavigate } from "@tanstack/react-router";
import { useState } from "react";
import { Charts } from "./Charts";
import { Play } from "./Play";

const route = getRouteApi("/blueprint");

/**
 * A trained blueprint, read two ways.
 *
 * The chart and the play table were separate destinations, which put the two
 * halves of one question — *what did it learn* — in different places, and made
 * the box that serves them a control belonging to neither. They are one subject
 * asked two ways: the grid is the strategy inspected, the table is the same
 * strategy played against. Both need the same run loaded on the same box.
 *
 * So the box control is here, above both, rather than duplicated into each: it
 * is what makes either tab answer anything at all, and "the box is asleep" is
 * the same fact whichever half you came for.
 *
 * The tab is a search param, so a spot is still shareable — see `Tabs`, and the
 * chart's `path`/`board`/`average`, which are search params for the same reason
 * and which this page had to keep when it absorbed them.
 */
export function Blueprint() {
  const { tab } = route.useSearch();
  const navigate = useNavigate({ from: "/blueprint" });
  // What the box is actually holding. Polled only while a swap is in flight.
  const run = useBlueprintRun();

  return (
    <div className="space-y-3">
      <Loaded
        loaded={run.data?.run ?? null}
        loading={run.data?.loading ?? null}
        canSwitch={run.data?.can_switch ?? false}
        error={run.error}
      />
      <Tabs
        tabs={[
          { id: "chart", label: "Chart", hint: "the strategy at one spot, as a 13x13 grid" },
          { id: "play", label: "Play", hint: "a hand against the loaded blueprint" },
        ]}
        active={tab}
        onPick={(next) => navigate({ search: (old) => ({ ...old, tab: next }) })}
      />
      {/* Mounted one at a time rather than hidden with CSS, and that is
          deliberate for Play: a hand's session lives on the BLUEPRINT SERVER,
          so keeping it mounted would hold a session open behind a tab nobody is
          looking at. Unmounting ends it where it lives. */}
      {tab === "chart" ? <Charts /> : <Play />}
    </div>
  );
}

/**
 * WHICH run you are looking at, and the control that changes it.
 *
 * Switching used to be an SSH script — `just serve-deploy <run>` — that re-synced
 * the code, re-ran `uv sync`, rewrote the unit's environment and restarted it,
 * about three minutes. None of that has to do with which run is loaded, and the
 * console could not do any of it, so this panel used to hand you the command to
 * paste elsewhere. The box swaps in process now, so it is a button.
 *
 * The load is watched rather than awaited: the server answers 202 immediately
 * and does the work on its own thread, so what is rendered here is the polled
 * state of the box, not the state of a request.
 */
function Loaded({
  loaded,
  loading,
  canSwitch,
  error,
}: {
  loaded: string | null;
  loading: string | null;
  canSwitch: boolean;
  error: unknown;
}) {
  const [wanted, setWanted] = useState<string | null>(null);
  const load = useLoadBlueprintRun();
  const target = wanted ?? loaded;
  const mismatch = wanted !== null && wanted !== loaded;

  return (
    <Panel title="loaded on the blueprint box">
      <div className="flex flex-wrap items-center gap-3 px-3 py-2.5">
        {loading ? (
          <span className="flex items-center gap-2 font-mono text-[13px] text-[var(--fg-muted)]">
            <span className="size-1.5 rounded-full bg-amber-400 motion-safe:animate-pulse" />
            loading {runLabel(loading)}…
          </span>
        ) : loaded ? (
          <span className="font-mono text-[13px] text-[var(--fg)]">{runLabel(loaded)}</span>
        ) : (
          <span className="text-[12px] text-[var(--fg-muted)]">
            {error
              ? "No blueprint server is reachable."
              : "Nothing loaded — start a blueprint server."}
          </span>
        )}

        <RunPicker value={target} onChange={setWanted} label="switch to" loadableOnly />

        {mismatch && !loading && (
          <button
            type="button"
            disabled={load.isPending}
            onClick={() => wanted && load.mutate({ run: wanted })}
            className="rounded border border-[var(--fg-faint)] px-2.5 py-1.5 font-mono text-[12px] text-[var(--fg)] hover:bg-white/[0.06] disabled:opacity-40"
          >
            {load.isPending ? "asking…" : "load it"}
          </button>
        )}

        <div className="ml-auto">
          <BoxControl />
        </div>
      </div>

      {/* The duration is stated because it is a minute of nothing happening on
          screen, and an unlabelled spinner reads as a hang at about thirty
          seconds — which is when someone reloads and asks for it twice. */}
      {loading && (
        <p className="border-t border-[var(--border)] px-3 py-2 text-[12px] text-[var(--fg-muted)]">
          Staging from the share and rebuilding — about a minute. Hands in progress ended: a session
          belongs to the run it was dealt from.
        </p>
      )}

      {load.error && (
        <p className="border-t border-red-500/30 bg-red-500/5 px-3 py-2 font-mono text-[12px] text-red-400">
          {String(load.error.message)}
        </p>
      )}

      {mismatch && !canSwitch && !loading && (
        <p className="border-t border-[var(--border)] px-3 py-2 text-[12px] text-[var(--fg-muted)]">
          This server cannot switch runs — it was started with a blueprint handed to it directly
          rather than a runs directory to resolve one from.
        </p>
      )}
    </Panel>
  );
}
