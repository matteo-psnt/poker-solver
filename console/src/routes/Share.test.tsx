import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import type { ReactNode } from "react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { Share } from "./Share";

/**
 * The guard rails on the one irreversible thing the console can do.
 *
 * `compact-legs --delete` removes the loose task records a bundle replaced, and
 * `--backup` is what makes that survivable — nothing else holds a copy. The
 * command enforces the pairing; this page has to enforce it too, or the
 * enforcement arrives only after a share read, which is a slow way to find out
 * and an easy state to click past.
 *
 * The other property here is the ordering: applying and deleting are two
 * decisions, and the form must never present the second before the first.
 */

let posted: Array<{ path: string; body: Record<string, unknown> }>;

beforeEach(() => {
  posted = [];
  vi.stubGlobal(
    "fetch",
    vi.fn(async (path: string, init?: RequestInit) => {
      posted.push({ path, body: JSON.parse(String(init?.body ?? "{}")) });
      return new Response(
        JSON.stringify({
          op: "compact-legs",
          bundle: "sealed.bundle.json",
          files_before: 375,
          files_after: 55,
          movable: 321,
          applied: false,
          verified: false,
          deleted: 0,
          backup: null,
        }),
        { status: 200 },
      );
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

describe("compacting the task record", () => {
  it("defaults to the dry run and sends no flags at all", async () => {
    mount(<Share />);
    const compact = panel(/Compact legs/);

    // The button says Preview because that is what an un-armed form does. It is
    // the same endpoint — the difference is entirely in what is NOT sent.
    fireEvent.click(compact.getByRole("button", { name: /Preview/ }));
    await waitFor(() => expect(posted).toHaveLength(1));
    expect(posted[0]?.body).toEqual({});
  });

  it("cannot arm --delete before --apply", () => {
    mount(<Share />);
    const compact = panel(/Compact legs/);
    const [apply, remove] = compact.getAllByRole("checkbox");

    // Deleting without applying is not a state the command has: the bundle must
    // exist and verify before the files it replaced can go.
    expect(remove?.hasAttribute("disabled")).toBe(true);
    fireEvent.click(apply as HTMLElement);
    expect(remove?.hasAttribute("disabled")).toBe(false);
  });

  it("disarms --delete when --apply is turned back off", () => {
    mount(<Share />);
    const compact = panel(/Compact legs/);
    const [apply, remove] = compact.getAllByRole("checkbox");

    fireEvent.click(apply as HTMLElement);
    fireEvent.click(remove as HTMLElement);
    expect((remove as HTMLInputElement).checked).toBe(true);

    // Otherwise the form holds an armed delete behind a disabled checkbox, and
    // re-arming apply would fire it without the second decision being retaken.
    fireEvent.click(apply as HTMLElement);
    expect((remove as HTMLInputElement).checked).toBe(false);
  });

  it("refuses to delete without a backup path", () => {
    mount(<Share />);
    const compact = panel(/Compact legs/);
    const [apply, remove] = compact.getAllByRole("checkbox");
    fireEvent.click(apply as HTMLElement);
    fireEvent.click(remove as HTMLElement);

    const button = compact.getByRole("button", { name: /Bundle and delete/ });
    expect(button.hasAttribute("disabled")).toBe(true);
    fireEvent.click(button);
    expect(posted).toHaveLength(0);
  });

  it("sends both flags once a backup is named", async () => {
    mount(<Share />);
    const compact = panel(/Compact legs/);
    const [apply, remove] = compact.getAllByRole("checkbox");
    fireEvent.click(apply as HTMLElement);
    fireEvent.click(remove as HTMLElement);
    fireEvent.change(compact.getByLabelText(/^backup$/i), {
      target: { value: "/home/me/legs-backup" },
    });

    fireEvent.click(compact.getByRole("button", { name: /Bundle and delete/ }));
    await waitFor(() => expect(posted).toHaveLength(1));
    expect(posted[0]?.body).toEqual({
      apply: true,
      delete: true,
      backup: "/home/me/legs-backup",
    });
  });
});

describe("publishing", () => {
  it("says the share was already current rather than showing nothing", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(
        async () =>
          new Response(JSON.stringify({ op: "push-data", uploaded: {} }), { status: 200 }),
      ),
    );
    mount(<Share />);
    const push = panel(/Push data —/);

    fireEvent.click(push.getByRole("button", { name: /Push data/ }));
    // An empty result is the ordinary outcome of pushing twice. Rendering it as
    // blank makes a successful no-op look like a button that did not fire.
    await waitFor(() => expect(push.getByText(/already current/)).toBeTruthy());
  });
});
