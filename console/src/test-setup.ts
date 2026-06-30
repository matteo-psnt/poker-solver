import { configure } from "@testing-library/react";

/**
 * jsdom gaps that a real browser fills.
 *
 * `matchMedia` is the one that matters: uPlot calls it at MODULE LOAD to pick up
 * `prefers-reduced-motion`, so merely importing a page that charts anything —
 * directly or through the route tree — throws before a single test runs. jsdom
 * has never implemented it.
 *
 * Answering "no match" rather than throwing is the honest default: no media
 * query is true of a headless DOM, and a test that depended on one would be
 * asserting about the test environment's viewport.
 */
if (!window.matchMedia) {
  window.matchMedia = (query: string): MediaQueryList =>
    ({
      matches: false,
      media: query,
      onchange: null,
      addListener: () => {},
      removeListener: () => {},
      addEventListener: () => {},
      removeEventListener: () => {},
      dispatchEvent: () => false,
    }) as MediaQueryList;
}

/**
 * `scrollTo` is jsdom's other unimplemented browser API, and the router calls it
 * on every navigation. Unstubbed it prints a stack per navigation and buries
 * whatever a failing test was trying to say.
 */
// Assigned unconditionally: jsdom DEFINES `scrollTo` and makes it throw
// "Not implemented", so a `if (!window.scrollTo)` guard never fires.
window.scrollTo = () => {};

/**
 * `ResizeObserver`, which jsdom does not implement at all.
 *
 * Recharts' `ResponsiveContainer` constructs one on mount, so without this any
 * page carrying a chart THROWS during render and the test sees an empty body
 * rather than a failure it can read.
 *
 * **It stops the crash; it does not make the chart draw.** Firing the callback
 * with a fixed box and stubbing `clientWidth` still produces an empty chart,
 * because jsdom has no layout engine for `ResponsiveContainer` to measure against.
 * So do not write a test that asserts on a rendered axis tick -- it will be
 * testing jsdom. Assert on what crosses the boundary INTO the chart instead;
 * `Cost.test.tsx` does that, and catches the unit bug a rendered tick was supposed
 * to catch.
 */
class NoopResizeObserver implements ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
globalThis.ResizeObserver ??= NoopResizeObserver;

/**
 * `waitFor`'s 1s default is a WALL-CLOCK budget, and vitest runs 21 files at
 * once: under that load a render that takes ~10ms alone overshoots it, so the
 * suite failed 1-3 arbitrary tests per parallel run and passed every one of
 * them serially. A changing failure list is environmental, never a regression.
 *
 * Raised rather than made serial because the assertions are unaffected -- a
 * genuinely broken query still fails, it just is not raced against the CI box.
 */
configure({ asyncUtilTimeout: 5000 });
