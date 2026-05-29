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
