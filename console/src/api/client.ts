/**
 * THE ONLY MODULE THAT CALLS `fetch`.
 *
 * That is the rule the console is subordinate to: every number it shows comes
 * from a command the CLI also runs. The previous browser UI was deleted for
 * carrying its own data layer, and a component reaching for the network is how
 * that starts again.
 */
import type { z } from "zod";

/** A refusal the server already made readable — render it, do not retry it. */
export class ApiError extends Error {
  readonly status: number;
  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * Generic over the SCHEMA, not over a payload type.
 *
 * `z.infer` is the schema's OUTPUT type, and that distinction is load-bearing:
 * a field with `.default()` is optional on input and guaranteed on output, so
 * collapsing the two makes every defaulted field `| undefined` at the call site
 * and every consumer defensive about a value the parser guarantees.
 */
export async function get<S extends z.ZodTypeAny>(path: string, schema: S): Promise<z.infer<S>> {
  const response = await fetch(path, {
    headers: { accept: "application/json" },
  });
  const raw = await response.text();

  if (!response.ok) {
    // The server puts the reason in `error` for exactly this: 422 is a refusal
    // (an unpublished run), 503 is the cloud (an expired `az login`). Both are
    // sentences meant for a person.
    throw new ApiError(reasonFrom(raw, response), response.status);
  }

  return check(path, schema, raw, response);
}

/**
 * The reason a failed response carries, or its status line.
 *
 * A failure whose body is not JSON at all — a proxy's own error page, a gateway
 * timeout — has no `error` field to read, and running it through `JSON.parse`
 * would throw *inside the error path* and replace a status nobody has to guess
 * at with a parse error nobody can act on.
 */
function reasonFrom(raw: string, response: Response): string {
  try {
    const body = JSON.parse(raw);
    if (body && typeof body === "object" && "error" in body) {
      return String((body as { error: unknown }).error);
    }
  } catch {
    // Not JSON. The status line is the honest answer.
  }
  return `${response.status} ${response.statusText}`;
}

/**
 * Parse a successful response, telling the two failures APART.
 *
 * They were one message, and the wrong one won. A 200 whose body is not JSON
 * used to reach `safeParse(null)` and be reported as *"did not match the
 * expected shape — Expected object, received null"*, which names the schema and
 * sends the reader to `schemas.ts`. The actual cause is upstream and has
 * nothing to do with the contract: under `console-dev` the Vite proxy answers a
 * refused connection with the SPA's own `index.html` at status 200, so every
 * panel blamed its payload while the real fact was that `serve` was not
 * running. That is a five-minute detour per occurrence and it recurs, because
 * the dev server outlives the backend by design.
 *
 * So a non-JSON body is a transport failure and says so, and a schema mismatch
 * keeps the loud field-naming message it is good at.
 */
function check<S extends z.ZodTypeAny>(
  path: string,
  schema: S,
  raw: string,
  response: Response,
): z.infer<S> {
  let body: unknown;
  try {
    body = JSON.parse(raw);
  } catch {
    throw new ApiError(
      `${path} answered ${response.status} but the body is not JSON. The API server is probably not running — start it with "just console" (or "poker-solver serve").`,
      502,
    );
  }

  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    // Loud, at the boundary, naming the field. The alternative is `undefined`
    // three components deep, which reads as a UI bug rather than a contract one.
    const issue = parsed.error.issues[0];
    // A root-level mismatch has an empty path, and joining it produced a stray
    // "` — `" with nothing before it.
    const where = issue?.path.length ? `${issue.path.join(".")} — ` : "";
    throw new ApiError(
      `Payload from ${path} did not match the expected shape: ${where}${issue?.message}`,
      500,
    );
  }
  return parsed.data;
}

/**
 * A mutation. The ONLY other verb this module speaks, and it exists for one
 * reason: play is stateful, so sitting down and acting are not reads.
 *
 * Deliberately the same parse-at-the-boundary contract as `get` — a mutation's
 * response is still a payload the UI renders, so letting it through unchecked
 * would put the hole in exactly the place a wrong card would be least visible.
 */
export async function send<S extends z.ZodTypeAny>(
  path: string,
  schema: S,
  body?: unknown,
  method: "POST" | "DELETE" = "POST",
): Promise<z.infer<S>> {
  const response = await fetch(path, {
    method,
    headers: { accept: "application/json", "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const raw = await response.text();

  if (!response.ok) {
    throw new ApiError(reasonFrom(raw, response), response.status);
  }

  // The same two-failures-apart rule as `get`, and it matters MORE here: a
  // dispatch that queued a task and then failed to parse its own receipt must
  // not read as "the submission was malformed". It was not — the work is on the
  // pool, and the message has to point at the response, not the request.
  return check(path, schema, raw, response);
}
