import { act, renderHook } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useClock } from "./clock";

describe("useClock", () => {
  beforeEach(() => vi.useFakeTimers({ now: 1_000_000 }));
  afterEach(() => vi.useRealTimers());

  it("moves without anything else re-rendering the page", () => {
    const { result } = renderHook(() => useClock(1_000));
    expect(result.current).toBe(1_000_000);
    act(() => vi.advanceTimersByTime(3_000));
    expect(result.current).toBe(1_003_000);
  });

  it("stops ticking when the component is gone", () => {
    const { unmount } = renderHook(() => useClock(1_000));
    unmount();
    expect(vi.getTimerCount()).toBe(0);
  });
});
