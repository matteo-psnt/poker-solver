import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Experiments } from "./Experiments";

/**
 * The page where a wrong reading moves a baseline.
 *
 * Two things are pinned. The experiment picker is DERIVED from the run listing
 * — no command lists experiments, and typing an id from memory was the only
 * other way to reach `report`. And an arm whose pairing was blocked must not
 * render as a dash: "—" reads as "no difference", which is a claim about a
 * comparison that never happened.
 */

const RUNS = {
  op: "runs",
  runs: [
    {
      name: "run-control",
      commits_ago: 0,
      git_dirty: false,
      loadable: true,
      blocker: null,
      iterations: 1000,
      num_infosets: 10,
      config_name: "production",
      status: "completed",
      experiment_id: "exp-7",
      arm: "control",
    },
    {
      name: "run-variant",
      commits_ago: 0,
      git_dirty: false,
      loadable: true,
      blocker: null,
      iterations: 1000,
      num_infosets: 10,
      config_name: "production",
      status: "completed",
      experiment_id: "exp-7",
      arm: "variant",
    },
    // Untagged, and it must not invent an experiment.
    {
      name: "run-loose",
      commits_ago: 0,
      git_dirty: false,
      loadable: true,
      blocker: null,
      iterations: 1,
      num_infosets: 1,
      config_name: "quick_test",
      status: "completed",
      experiment_id: null,
      arm: null,
    },
  ],
};

const REPORT = {
  op: "report",
  experiment_id: "exp-7",
  control_run_id: "run-control",
  baseline_run_id: null,
  notes: ["Tier: exact_br"],
  arms: [
    {
      arm: "control",
      run_id: "run-control",
      checkpoint_iteration: 1000,
      exploitability_mbb: 900,
      std_error_mbb: 1,
      git_branch: "main",
      vs_control_mbb: null,
      vs_control_p_value: null,
      vs_control_blocked: [],
    },
    {
      arm: "variant",
      run_id: "run-variant",
      checkpoint_iteration: 1000,
      exploitability_mbb: 880,
      std_error_mbb: 1,
      git_branch: "wt-x",
      vs_control_mbb: null,
      vs_control_p_value: null,
      vs_control_blocked: ["payload missing, cannot pair"],
    },
  ],
};

beforeEach(() => {
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string) => {
      const body = String(path).startsWith("/api/experiments") ? REPORT : RUNS;
      return new Response(JSON.stringify(body), { status: 200 });
    }),
  );
});

afterEach(() => vi.unstubAllGlobals());

function mount(node: ReactNode) {
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
}

function panel(title: RegExp) {
  const section = screen.getByText(title).closest("section");
  if (!section) throw new Error(`no panel titled ${title}`);
  return within(section);
}

/**
 * A picker, once its query has filled it.
 *
 * The experiment field does not exist until the run listing lands — there is
 * nothing to choose from before then, and rendering an empty dropdown would
 * claim there are no experiments rather than that none have loaded.
 */
async function picker(title: RegExp, label: RegExp) {
  return waitFor(() => panel(title).getByLabelText(label));
}

describe("the experiment picker", () => {
  it("is built from the runs that carry a tag, deduplicated", async () => {
    mount(<Experiments />);
    const select = await picker(/Experiment report/, /^experiment$/i);
    // Two runs share `exp-7` and one carries no tag: one option, plus the
    // "unset" one that every picker offers.
    await waitFor(() =>
      expect([...select.querySelectorAll("option")].map((o) => o.value)).toEqual(["", "exp-7"]),
    );
  });
});

describe("the arm table", () => {
  it("says unpaired rather than showing a dash for a blocked comparison", async () => {
    mount(<Experiments />);
    const select = await picker(/Experiment report/, /^experiment$/i);
    const report = panel(/Experiment report/);
    await waitFor(() =>
      expect([...select.querySelectorAll("option")].map((o) => o.value)).toContain("exp-7"),
    );
    fireEvent.change(select, { target: { value: "exp-7" } });

    // A dash here would read as "no difference" — a claim about a comparison
    // the report explicitly refused to make.
    await waitFor(() => expect(report.getByText("unpaired")).toBeTruthy());
    expect(report.getByText(/Tier: exact_br/)).toBeTruthy();
  });
});

describe("promoting", () => {
  it("will not promote without a rationale", async () => {
    mount(<Experiments />);
    const select = await picker(/Promote —/, /^run$/i);
    const promote = panel(/Promote —/);
    await waitFor(() =>
      expect([...select.querySelectorAll("option")].map((o) => o.value)).toContain("run-control"),
    );
    fireEvent.change(select, { target: { value: "run-control" } });

    // The command requires one: a lineage that moved for an unrecorded reason
    // cannot be audited later.
    const button = promote.getByRole("button", { name: /^Promote$/ });
    expect(button.hasAttribute("disabled")).toBe(true);

    fireEvent.change(promote.getByLabelText(/^rationale$/i), { target: { value: "best so far" } });
    expect(button.hasAttribute("disabled")).toBe(false);
  });

  it("treats whitespace as no rationale at all", async () => {
    mount(<Experiments />);
    const select = await picker(/Promote —/, /^run$/i);
    const promote = panel(/Promote —/);
    await waitFor(() =>
      expect([...select.querySelectorAll("option")].map((o) => o.value)).toContain("run-control"),
    );
    fireEvent.change(select, { target: { value: "run-control" } });
    fireEvent.change(promote.getByLabelText(/^rationale$/i), { target: { value: "   " } });

    expect(promote.getByRole("button", { name: /^Promote$/ }).hasAttribute("disabled")).toBe(true);
  });
});
