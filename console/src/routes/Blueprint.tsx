import { getRouteApi, useNavigate } from "@tanstack/react-router";
import { useBlueprintRun } from "@/api/queries";
import { BoxControl } from "@/components/BoxControl";
import { Panel } from "@/components/Panel";
import { Tabs } from "@/components/Tabs";
import { runLabel } from "@/lib/format";
import { Charts } from "./Charts";
import { Play } from "./Play";

const route = getRouteApi("/blueprint");

/**
 * The fielded blueprint, read two ways: the grid is the strategy inspected, the
 * table is the same strategy played against.
 *
 * Both read the one run the serving box holds, which is why the box sits here
 * above both rather than duplicated into each -- "the box is asleep" is the same
 * fact whichever half you came for.
 *
 * The tab is a search param, so a spot stays shareable -- as are the chart's
 * `path`/`board`/`average`, which this page had to keep when it absorbed them.
 */
export function Blueprint() {
  const { tab } = route.useSearch();
  const navigate = useNavigate({ from: "/blueprint" });
  const run = useBlueprintRun();

  return (
    <div className="space-y-3">
      <Serving run={run.data?.run ?? null} error={run.error} />
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
 * WHICH run you are reading, stated and not choosable.
 *
 * The box holds a Chipzen ladder slot, so what it serves is the policy being
 * fielded -- a picker here would let this page and the seat disagree while both
 * claimed to be Blueprint. Changing it is `just serve-deploy <run>`, the same
 * act that seats it.
 */
function Serving({ run, error }: { run: string | null; error: unknown }) {
  return (
    <Panel title="serving">
      <div className="flex flex-wrap items-center gap-3 px-3 py-2.5">
        {run ? (
          <span className="font-mono text-[13px] text-[var(--fg)]">{runLabel(run)}</span>
        ) : (
          <span className="text-[12px] text-[var(--fg-muted)]">
            {error
              ? "No blueprint server answered — the host may be starting, or this console has no address for it."
              : "Asking the host what it holds…"}
          </span>
        )}
        <div className="ml-auto">
          <BoxControl />
        </div>
      </div>
      {/* The command is named because the alternative is guessing: nothing on
          this page can change the run, and a reader who wants a different one
          has to be told where that decision lives. */}
      <p className="border-t border-[var(--border)] px-3 py-2 text-[12px] text-[var(--fg-muted)]">
        One run per host, staged by the deploy that seats it —{" "}
        <code className="font-mono text-[11px]">just serve-deploy &lt;run&gt;</code> to change it.
      </p>
    </Panel>
  );
}
