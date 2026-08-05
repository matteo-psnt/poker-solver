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
  const body = await response.json().catch(() => null);

  if (!response.ok) {
    // The server puts the reason in `error` for exactly this: 422 is a refusal
    // (an unpublished run), 503 is the cloud (an expired `az login`). Both are
    // sentences meant for a person.
    const reason =
      body && typeof body === "object" && "error" in body
        ? String((body as { error: unknown }).error)
        : `${response.status} ${response.statusText}`;
    throw new ApiError(reason, response.status);
  }

  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    // Loud, at the boundary, naming the field. The alternative is `undefined`
    // three components deep, which reads as a UI bug rather than a contract one.
    const issue = parsed.error.issues[0];
    throw new ApiError(
      `Payload from ${path} did not match the expected shape: ${issue?.path.join(".")} — ${issue?.message}`,
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
  const payload = await response.json().catch(() => null);

  if (!response.ok) {
    const reason =
      payload && typeof payload === "object" && "error" in payload
        ? String((payload as { error: unknown }).error)
        : `${response.status} ${response.statusText}`;
    throw new ApiError(reason, response.status);
  }

  const parsed = schema.safeParse(payload);
  if (!parsed.success) {
    const issue = parsed.error.issues[0];
    throw new ApiError(
      `Payload from ${path} did not match the expected shape: ${issue?.path.join(".")} — ${issue?.message}`,
      500,
    );
  }
  return parsed.data;
}
