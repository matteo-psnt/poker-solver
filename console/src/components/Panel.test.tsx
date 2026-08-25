import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { Panel } from "./Panel";

/**
 * `children != null` was true for `false` and for every array, so a panel whose
 * body is a `&&` guard or two expressions reported "has content" and rendered an
 * empty div. Thirteen panels' empty and loading states were unreachable: a fresh
 * share showed `/runs` as a titled panel with a blank body.
 */
describe("Panel says why a body is empty", () => {
  it("shows the empty message when a guard did not fire", () => {
    render(
      <Panel title="Runs" empty="No published runs.">
        {false}
      </Panel>,
    );
    expect(screen.getByText("No published runs.")).toBeTruthy();
  });

  it("shows it when the body is several falsy expressions", () => {
    render(
      <Panel title="Task log" empty="No record for this task.">
        {false}
        {null}
      </Panel>,
    );
    expect(screen.getByText("No record for this task.")).toBeTruthy();
  });

  it("shows the skeleton while loading a body that has not arrived", () => {
    const { container } = render(
      <Panel title="Tasks" loading empty="No tasks match.">
        {[]}
      </Panel>,
    );
    expect(screen.queryByText("No tasks match.")).toBeNull();
    expect(container.querySelectorAll('[class*="animate-pulse"]').length).toBeGreaterThan(0);
  });

  it("still renders a real body", () => {
    render(
      <Panel title="Runs" empty="No published runs.">
        <p>run-a</p>
      </Panel>,
    );
    expect(screen.getByText("run-a")).toBeTruthy();
    expect(screen.queryByText("No published runs.")).toBeNull();
  });
});
