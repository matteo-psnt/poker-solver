import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Dispatch } from "./Dispatch";

/**
 * What the dispatch page actually SENDS.
 *
 * The console gained seven writes at once, and the property they all rest on is
 * invisible in every other check: `tsc` is happy with a body carrying
 * `config: ""`, the Zod schemas only describe what comes BACK, and the Python
 * tests can only pin what the endpoint does with a body it was given. The one
 * place "an untouched field must not become an argument" can be observed is
 * here, at the click.
 *
 * These drive the real components against a stubbed `fetch`, so the form state,
 * the `given()` filter and the mutation are all exercised as they ship.
 */

const PAYLOADS: Record<string, unknown> = {
  "/api/configs": {
    op: "configs",
    root: "/repo/config",
    kinds: [
      { kind: "training", flag: "submit --config", names: ["production", "quick_test"] },
      { kind: "abstraction", flag: "submit-precompute --config", names: ["production"] },
    ],
  },
  "/api/runs": {
    op: "runs",
    runs: [
      {
        name: "run-a",
        commits_ago: 0,
        git_dirty: false,
        loadable: true,
        blocker: null,
        iterations: 1000,
        num_infosets: 10,
        config_name: "production",
        status: "completed",
      },
    ],
  },
};

/** Every POST, in order, as the body the page decided to send. */
let posted: Array<{ path: string; body: Record<string, unknown> }>;

beforeEach(() => {
  posted = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string, init?: RequestInit) => {
      if (init?.method === "POST") {
        posted.push({ path, body: JSON.parse(String(init.body)) });
        return new Response(
          JSON.stringify({
            op: "submit",
            code_snapshot: "code-1",
            job_id: "poker-1",
            tasks: ["t-1"],
          }),
          { status: 200 },
        );
      }
      const payload = PAYLOADS[String(path).split("?")[0] ?? ""];
      return new Response(JSON.stringify(payload ?? {}), { status: payload ? 200 : 404 });
    }),
  );
});

afterEach(() => vi.unstubAllGlobals());

function mount(node: ReactNode) {
  // `retry: false` so a stubbed failure surfaces immediately instead of being
  // retried past the end of the test.
  const client = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  return render(<QueryClientProvider client={client}>{node}</QueryClientProvider>);
}

/**
 * Queries scoped to one panel.
 *
 * All three forms are on one page and share field names — "config" is both a
 * training stem and an abstraction stem — which is fine on screen, where each
 * sits under its own heading, and ambiguous to a flat query. Scoping keeps the
 * test asking the same question a reader does: what does the TRAIN panel say.
 */
function panel(title: RegExp) {
  const section = screen.getByText(title).closest("section");
  if (!section) throw new Error(`no panel titled ${title}`);
  return within(section);
}

/** Wait until a picker has been filled from its query, then choose. */
async function choose(scope: ReturnType<typeof within>, label: RegExp, value: string) {
  const select = scope.getByLabelText(label);
  await waitFor(() =>
    expect([...select.querySelectorAll("option")].map((o) => o.value)).toContain(value),
  );
  fireEvent.change(select, { target: { value } });
}

describe("the training form", () => {
  it("sends only the fields that were filled in", async () => {
    mount(<Dispatch />);
    const train = panel(/Train/);

    fireEvent.change(train.getByLabelText(/^target$/i), { target: { value: "25,000,000" } });
    await choose(train, /^config$/i, "production");
    fireEvent.click(train.getByRole("button", { name: /Queue training/ }));

    await waitFor(() => expect(posted).toHaveLength(1));
    // No `run`, no `experiment`, no `workers`. Each of those absent keys is a
    // default the COMMAND supplies; sending an empty string or a 0 would be
    // this surface quietly answering a question it was not asked.
    expect(posted[0]).toEqual({
      path: "/api/submit",
      body: { to: 25_000_000, config: "production" },
    });
  });

  it("will not queue without a target, whatever else is set", async () => {
    mount(<Dispatch />);
    const train = panel(/Train/);
    await choose(train, /^config$/i, "production");

    const button = train.getByRole("button", { name: /Queue training/ });
    expect(button.hasAttribute("disabled")).toBe(true);
    fireEvent.click(button);
    expect(posted).toHaveLength(0);
  });

  it("labels the one field that is not a Form control", async () => {
    mount(<Dispatch />);
    // `Field` associates by cloning its generated id onto its child. Every
    // other call site passes a `Form` control that forwards `id`; this one
    // passes a raw <textarea>. If the clone ever stops landing — a fragment, a
    // wrapper div, a component that ignores the prop — `htmlFor` points at
    // nothing and the label silently stops working, with no error anywhere.
    const overrides = panel(/Train/).getByLabelText(/^overrides$/i);
    expect(overrides.tagName).toBe("TEXTAREA");

    fireEvent.change(overrides, { target: { value: "solver__cfr_plus=true\nnonsense" } });
    fireEvent.change(panel(/Train/).getByLabelText(/^target$/i), { target: { value: "1000" } });
    await choose(panel(/Train/), /^config$/i, "production");
    fireEvent.click(panel(/Train/).getByRole("button", { name: /Queue training/ }));

    await waitFor(() => expect(posted).toHaveLength(1));
    expect(posted[0]?.body.sets).toEqual(["solver__cfr_plus=true"]);
  });

  it("offers the training configs the server listed, not the abstraction ones", async () => {
    mount(<Dispatch />);
    const select = panel(/Train/).getByLabelText(/^config$/i);
    // The two directories are different vocabularies, and `submit` rejects the
    // wrong one only AFTER sealing a snapshot and spinning up the pool.
    await waitFor(() =>
      expect([...select.querySelectorAll("option")].map((o) => o.value)).toEqual([
        "",
        "production",
        "quick_test",
      ]),
    );
  });
});

describe("the precompute form", () => {
  it("omits --force until it is explicitly armed", async () => {
    mount(<Dispatch />);
    const pre = panel(/Precompute/);
    await choose(pre, /^config$/i, "production");

    fireEvent.click(pre.getByRole("button", { name: /Queue precompute/ }));
    await waitFor(() => expect(posted).toHaveLength(1));
    // Republishing over a name silently invalidates the provenance of every run
    // trained against it. An unchecked box must reach the command as *nothing*,
    // so its own `store_true` default is what answers.
    expect(posted[0]?.body).toEqual({ config: "production" });

    fireEvent.click(pre.getByRole("checkbox"));
    fireEvent.click(pre.getByRole("button", { name: /Queue precompute/ }));
    await waitFor(() => expect(posted).toHaveLength(2));
    expect(posted[1]?.body).toEqual({ config: "production", force: true });
  });
});

describe("the scoring form", () => {
  it("normalises the rungs into what --at parses", async () => {
    mount(<Dispatch />);
    const score = panel(/Score/);
    await choose(score, /^run$/i, "run-a");
    fireEvent.change(score.getByLabelText(/^rungs$/i), {
      target: { value: "10_000_000, 20000000" },
    });

    fireEvent.click(score.getByRole("button", { name: /Queue scoring/ }));
    await waitFor(() => expect(posted).toHaveLength(1));
    expect(posted[0]?.body).toEqual({ run: "run-a", at: "10000000,20000000" });
  });

  it("shows what was queued rather than clearing the form", async () => {
    mount(<Dispatch />);
    const score = panel(/Score/);
    await choose(score, /^run$/i, "run-a");
    fireEvent.click(score.getByRole("button", { name: /Queue scoring/ }));

    // The job id and task names are how this submission is found again on the
    // Tasks page. A message that disappears takes the only record of them.
    await waitFor(() => expect(score.getByText("poker-1")).toBeTruthy());
    expect(score.getByText(/code-1/)).toBeTruthy();
    expect(score.getByText("t-1")).toBeTruthy();
  });
});
