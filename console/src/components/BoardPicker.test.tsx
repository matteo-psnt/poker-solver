import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";
import { BoardPicker } from "./BoardPicker";

/**
 * The control that made postflop reachable, so the three things it has to get
 * right are asserted directly rather than left to look right.
 *
 * It writes into a search param, so every change goes through `onChange` as the
 * spelling the server reads — that is the contract, and a picker that produced
 * `"A♠K♦"` would look correct and bookmark nothing.
 */
function pick(board: string, props: Partial<{ live: number | null; forceOpen: boolean }> = {}) {
  const onChange = vi.fn();
  render(<BoardPicker board={board} onChange={onChange} forceOpen {...props} />);
  return onChange;
}

describe("dealing a board", () => {
  it("appends the card you click, in the spelling parse_board reads", () => {
    const onChange = pick("");
    fireEvent.click(screen.getByTitle("As"));
    expect(onChange).toHaveBeenCalledWith("As");
  });

  it("deals onto the end rather than replacing", () => {
    const onChange = pick("AsKd");
    fireEvent.click(screen.getByTitle("7c"));
    expect(onChange).toHaveBeenCalledWith("AsKd7c");
  });

  it("appends to the CARDS, not to the raw search param", () => {
    // The board comes out of the URL, which anyone can hand-edit and which
    // outlives the page that wrote it. Appending to the raw text carried the
    // junk along: `"AsK"` plus a click wrote `"AsK7c"`, which the server
    // refuses as not a whole number of cards.
    const onChange = pick("AsK");
    fireEvent.click(screen.getByTitle("7c"));
    expect(onChange).toHaveBeenCalledWith("As7c");
  });

  it("canonicalises a spaced board on the next click", () => {
    const onChange = pick("As Kd");
    fireEvent.click(screen.getByTitle("7c"));
    expect(onChange).toHaveBeenCalledWith("AsKd7c");
  });

  it("will not deal a card already on the board", () => {
    pick("As");
    // `parse_board` refuses a repeat, so it is unclickable here rather than
    // sent and bounced.
    expect(screen.getByTitle(/As is already on the board/).hasAttribute("disabled")).toBe(true);
  });

  it("stops at five", () => {
    pick("AsKd7c2h9s");
    expect(screen.getByTitle("3d").hasAttribute("disabled")).toBe(true);
  });
});

describe("the slots are the way in", () => {
  it("opens the deck when you click an empty one", () => {
    // No "pick cards" button: you point at the empty turn to say "deal a turn".
    pick("", { forceOpen: false });
    expect(screen.queryByTitle("As")).toBeNull();
    // Three slots carry this title — the flop is three cards.
    const [flop] = screen.getAllByTitle("pick the flop");
    fireEvent.click(flop as HTMLElement);
    expect(screen.getByTitle("As")).toBeTruthy();
  });

  it("changes nothing by opening — an empty slot is past the end of the board", () => {
    const onChange = pick("AsKd7c", { forceOpen: false });
    fireEvent.click(screen.getByTitle("pick the turn"));
    expect(onChange).not.toHaveBeenCalled();
  });
});

describe("taking a board back", () => {
  it("clears a slot AND everything after it, because a board is an ordered deal", () => {
    const onChange = pick("AsKd7c2h");
    // The turn — index 3. Clearing it alone would promote the river to a turn.
    fireEvent.click(screen.getByTitle(/^2h —/));
    expect(onChange).toHaveBeenCalledWith("AsKd7c");
  });

  it("clears the flop's first card back to nothing", () => {
    const onChange = pick("AsKd7c");
    fireEvent.click(screen.getByTitle(/^As —/));
    expect(onChange).toHaveBeenCalledWith("");
  });

  it("empties on clear", () => {
    const onChange = pick("AsKd7c");
    fireEvent.click(screen.getByText("clear"));
    expect(onChange).toHaveBeenCalledWith("");
  });
});

describe("a board longer than the line uses", () => {
  it("says so, because replay ignores the surplus rather than refusing it", () => {
    pick("AsKd7c2h9s", { live: 3 });
    expect(screen.getByText(/This line reaches 3 of them/)).toBeTruthy();
  });

  it("says nothing when the line reaches all of them", () => {
    pick("AsKd7c", { live: 3 });
    expect(screen.queryByText(/This line reaches/)).toBeNull();
  });

  it("says nothing while the line is unanswerable", () => {
    pick("AsKd7c2h9s", { live: null });
    expect(screen.queryByText(/This line reaches/)).toBeNull();
  });
});
