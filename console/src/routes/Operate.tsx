import { Tabs } from "@/components/Tabs";
import { getRouteApi, useNavigate } from "@tanstack/react-router";
import { Activity } from "./Activity";
import { Cost } from "./Cost";
import { Dispatch } from "./Dispatch";
import { Share } from "./Share";

const route = getRouteApi("/operate");

/**
 * Doing things, and accounting for what they cost.
 *
 * Four destinations that were each one command's page. They belong together
 * because they are one person's job rather than one subject: queue the work,
 * publish what it needs, see the bill, see what the tool itself has been doing.
 * None of them is something you open to answer a question about the SOLVER,
 * which is what separates them from Runs.
 *
 * Cost sits here rather than beside the record for the reason the command line
 * groups it this way too: it accounts for dispatched work. The burn rate that
 * matters while something is running is on `Now`, where it is read; this is
 * the fuller breakdown, including the standing charges no task log can explain.
 */
export function Operate() {
  const { tab } = route.useSearch();
  const navigate = useNavigate({ from: "/operate" });

  return (
    <div className="space-y-3">
      <Tabs
        tabs={[
          { id: "dispatch", label: "Dispatch", hint: "queue training, scoring or precompute" },
          { id: "share", label: "Share", hint: "publish code and abstractions; compact legs" },
          { id: "cost", label: "Cost", hint: "node hours against what Azure actually billed" },
          {
            id: "activity",
            label: "Activity",
            hint: "what this tool has been doing, and how slowly",
          },
        ]}
        active={tab}
        onPick={(next) => navigate({ search: (old) => ({ ...old, tab: next }) })}
      />
      {tab === "dispatch" && <Dispatch />}
      {tab === "share" && <Share />}
      {tab === "cost" && <Cost />}
      {tab === "activity" && <Activity />}
    </div>
  );
}
