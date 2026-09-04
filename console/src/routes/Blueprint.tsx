import { getRouteApi, useNavigate } from "@tanstack/react-router";
import { Tabs } from "@/components/Tabs";
import { Charts } from "./Charts";
import { Play } from "./Play";

const route = getRouteApi("/blueprint");

/**
 * The fielded blueprint, read two ways: the grid is the strategy inspected, the
 * table is the same strategy played against.
 *
 * Nothing above the tabs. A host serves one run for the life of its process and
 * nothing here can change it, so a header naming that run was a line of chrome
 * on every spot -- and both halves below already say what they are looking at.
 *
 * The tab is a search param, so a spot stays shareable -- as are the chart's
 * `path`/`board`/`average`, which this page had to keep when it absorbed them.
 */
export function Blueprint() {
  const { tab } = route.useSearch();
  const navigate = useNavigate({ from: "/blueprint" });

  return (
    <div className="space-y-3">
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
