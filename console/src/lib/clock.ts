import { useEffect, useState } from "react";

/**
 * `Date.now()`, re-read every `everyMs`, as state -- so what is drawn from it
 * moves.
 *
 * Anything computed against the wall clock at render time -- an age badge, a
 * "running for" -- is only as current as the last render, and a page served
 * from the server's memo renders only when its payload CHANGES. Eight polls in
 * a row returning the same composition left "9s ago" saying 9s for a minute.
 * A poll is not a tick; this is the tick.
 */
export function useClock(everyMs = 1_000): number {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = setInterval(() => setNow(Date.now()), everyMs);
    return () => clearInterval(timer);
  }, [everyMs]);
  return now;
}
