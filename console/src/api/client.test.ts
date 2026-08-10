import { afterEach, describe, expect, it, vi } from "vitest";
import { z } from "zod";
import { ApiError, get, send } from "./client";

/**
 * Which failure a message BLAMES.
 *
 * Every panel renders `error` verbatim, so the sentence here is the whole
 * diagnosis. The one that went wrong: a 200 whose body was not JSON reached
 * `safeParse(null)` and came back as "did not match the expected shape —
 * Expected object, received null", which names the schema and sends the reader
 * to `schemas.ts`. The actual cause was that `serve` was not running: under
 * `console-dev` the Vite proxy answers a refused connection with the SPA's own
 * `index.html` at status 200. Every panel blamed its payload.
 */

const schema = z.object({ op: z.literal("pool-status"), pool_id: z.string() });

const answers = (body: string, status = 200, statusText = "OK") =>
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => new Response(body, { status, statusText })),
  );

afterEach(() => vi.unstubAllGlobals());

describe("a body that is not JSON", () => {
  it("blames the server being down, not the schema", async () => {
    answers("<!DOCTYPE html><div id=root></div>");
    await expect(get("/api/pool", schema)).rejects.toThrow(/not JSON/);
    // The fix, in one assertion: the message must NOT send the reader to the
    // contract for a failure the contract had no part in.
    await expect(get("/api/pool", schema)).rejects.not.toThrow(/expected shape/);
  });

  it("names the fix", async () => {
    answers("");
    await expect(get("/api/pool", schema)).rejects.toThrow(/poker-solver serve/);
  });

  it("does the same for a write", async () => {
    // A dispatch that queued a task and then could not parse its receipt must
    // not read as "the submission was malformed" — the work is on the pool.
    answers("<!DOCTYPE html>");
    await expect(send("/api/submit", schema, { to: 1 })).rejects.toThrow(/not JSON/);
  });
});

describe("a real schema mismatch", () => {
  it("still names the field", async () => {
    answers(JSON.stringify({ op: "pool-status", pool_id: 7 }));
    await expect(get("/api/pool", schema)).rejects.toThrow(/pool_id — /);
  });

  it("does not leave a stray dash when the mismatch is at the root", async () => {
    // `[].join(".")` is "", which used to print ": — Expected object".
    answers(JSON.stringify([]));
    const error = await get("/api/pool", schema).catch((e) => e as ApiError);
    expect(error).toBeInstanceOf(ApiError);
    expect((error as ApiError).message).toContain("Expected object");
    expect((error as ApiError).message).not.toContain(": — ");
  });
});

describe("a failed response", () => {
  it("prefers the server's own sentence", async () => {
    // 422 is a refusal the server already made readable; 503 names `az login`.
    answers(JSON.stringify({ error: "'run-x' is not published" }), 422, "Unprocessable");
    await expect(get("/api/runs/run-x", schema)).rejects.toThrow(/not published/);
  });

  it("falls back to the status line when the body is not JSON", async () => {
    // A gateway's own error page has no `error` field, and parsing it inside
    // the error path would replace a status with a parse error.
    answers("<html>502 Bad Gateway</html>", 502, "Bad Gateway");
    await expect(get("/api/pool", schema)).rejects.toThrow(/502 Bad Gateway/);
  });

  it("carries the status so the caller can tell a refusal from an outage", async () => {
    answers(JSON.stringify({ error: "no" }), 503, "Service Unavailable");
    const error = await get("/api/pool", schema).catch((e) => e as ApiError);
    expect((error as ApiError).status).toBe(503);
  });
});
