import { afterEach, describe, expect, it, vi } from "vitest";
import { type ApiError, get, isTransient, send } from "./client";

/**
 * Which failure a message BLAMES.
 *
 * Every panel renders `error` verbatim, so the sentence here is the whole
 * diagnosis -- and blaming the wrong layer sends the reader to the wrong file. The
 * distinction the client is still the only place to draw: a transport failure
 * versus the server declining to answer. Under `console-dev` the Vite proxy
 * answers a refused connection with the SPA's own `index.html` at status 200,
 * which is why a 200 with a non-JSON body must not read as a contract violation.
 */

const answers = (body: string, status = 200, statusText = "OK") =>
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(body, { status, statusText })),
  );

afterEach(() => vi.unstubAllGlobals());

type Pool = { op: "pool-status"; pool_id: string };

describe("a body that is not JSON", () => {
  it("blames the server being down, not the payload", async () => {
    answers("<!DOCTYPE html><div id=root></div>");
    await expect(get<Pool>("/api/pool")).rejects.toThrow(/not JSON/);
    // The fix, in one assertion: the message must NOT send the reader to the
    // contract for a failure the contract had no part in.
    await expect(get<Pool>("/api/pool")).rejects.not.toThrow(/expected shape/);
  });

  it("names the fix", async () => {
    answers("");
    await expect(get<Pool>("/api/pool")).rejects.toThrow(/poker-solver serve/);
  });

  it("does the same for a write", async () => {
    // A dispatch that queued a task and then could not read its receipt must not
    // read as "the submission was malformed" — the work is on the pool.
    answers("<!DOCTYPE html>");
    await expect(send("/api/submit", { to: 1 })).rejects.toThrow(/not JSON/);
  });
});

describe("a successful response", () => {
  it("returns the decoded payload", async () => {
    answers(JSON.stringify({ op: "pool-status", pool_id: "poker-pool" }));
    await expect(get<Pool>("/api/pool")).resolves.toEqual({
      op: "pool-status",
      pool_id: "poker-pool",
    });
  });

  it("passes a write's body through and returns its receipt", async () => {
    const fetcher = vi.fn(
      async (_path: string, _init?: RequestInit) =>
        new Response(JSON.stringify({ job_id: "poker-20260812" })),
    );
    vi.stubGlobal("fetch", fetcher);
    await expect(send("/api/submit", { to: 1 })).resolves.toEqual({ job_id: "poker-20260812" });
    expect(fetcher.mock.calls[0]?.[1]).toMatchObject({
      method: "POST",
      body: JSON.stringify({ to: 1 }),
    });
  });
});

describe("which failures are worth retrying", () => {
  it("retries the server not being there yet", async () => {
    // The state `console-dev` is in for its first few seconds, every time. It
    // healed itself on any panel with a poll interval and stayed broken forever
    // on `configs`, which has `staleTime: Infinity` and no interval.
    answers("<!DOCTYPE html>");
    const error = await get<Pool>("/api/configs").catch((e) => e);
    expect(isTransient(error)).toBe(true);
  });

  it("retries a connection that never opened", async () => {
    // Production has no proxy to dress this up: `fetch` rejects with TypeError.
    expect(isTransient(new TypeError("Failed to fetch"))).toBe(true);
  });

  it("does NOT retry a refusal", async () => {
    // 422 is the server saying no, and it will say no again.
    answers(JSON.stringify({ error: "'run-x' is not published" }), 422, "Unprocessable");
    const error = await get<Pool>("/api/runs/run-x").catch((e) => e);
    expect(isTransient(error)).toBe(false);
  });

  it("does NOT retry an expired credential", async () => {
    // 503 needs `az login`, not patience. Retrying delays the message that
    // names the fix.
    answers(JSON.stringify({ error: "try `az login`" }), 503, "Service Unavailable");
    const error = await get<Pool>("/api/pool").catch((e) => e);
    expect(isTransient(error)).toBe(false);
  });
});

describe("a failed response", () => {
  it("prefers the server's own sentence", async () => {
    // 422 is a refusal the server already made readable; 503 names `az login`.
    answers(JSON.stringify({ error: "'run-x' is not published" }), 422, "Unprocessable");
    await expect(get<Pool>("/api/runs/run-x")).rejects.toThrow(/not published/);
  });

  it("falls back to the status line when the body is not JSON", async () => {
    // A gateway's own error page has no `error` field, and parsing it inside
    // the error path would replace a status with a parse error.
    answers("<html>502 Bad Gateway</html>", 502, "Bad Gateway");
    await expect(get<Pool>("/api/pool")).rejects.toThrow(/502 Bad Gateway/);
  });

  it("carries the status so the caller can tell a refusal from an outage", async () => {
    answers(JSON.stringify({ error: "no" }), 503, "Service Unavailable");
    const error = await get<Pool>("/api/pool").catch((e) => e as ApiError);
    expect((error as ApiError).status).toBe(503);
  });
});
