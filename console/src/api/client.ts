/**
 * THE ONLY MODULE THAT CALLS `fetch`.
 *
 * That is the rule the console is subordinate to: every number it shows comes
 * from a command the CLI also runs. The previous browser UI was deleted for
 * carrying its own data layer, and a component reaching for the network is how
 * that starts again.
 *
 * Why there is no runtime schema check any more
 * ---------------------------------------------
 * There used to be one, and it was worth having while the TypeScript shapes were
 * hand-written: two independent declarations of the same payload drift, so
 * catching it at the boundary beat meeting `undefined` three components deep.
 *
 * They are no longer independent. `contract.py` declares the shapes, FastAPI
 * exports them to `openapi.json`, and `types.gen.ts` is generated from that on
 * every build -- so a TypeScript type that disagrees with the server is not a
 * thing that can be committed. Keeping a parser here would mean hand-writing the
 * schemas again, which is the exact cost this removed. (What the parser actually
 * caught, in the end, was itself: `cancelSchema` required `job`/`task` where the
 * command returns `job_id`/`task_id`, so the console cancelled a task and then
 * reported that it had failed.)
 *
 * What is NOT given up is telling a transport failure apart from a payload one.
 * That distinction was learned the hard way and is kept below.
 */

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
 * Whether a failure is worth trying again — the ONE case where it is.
 *
 * Nothing was retried, on the reasoning that a cloud read which failed will not
 * succeed a second later. That is right for everything the server itself
 * answers: a 422 is a refusal it will repeat, and a 503 means Azure did not
 * answer, which needs `az login` and not patience.
 *
 * It is wrong for the server not being there YET, which is the ordinary state
 * for the first seconds of `just console-dev`: it starts Vite and `serve`
 * together, Vite is ready in ~3s and the Python server is not, so every query
 * fired in that window failed permanently. Panels on an interval healed
 * themselves and looked fine; `configs` — `staleTime: Infinity`, no interval —
 * stayed broken for the life of the tab, which is how this surfaced.
 *
 * Two shapes, and neither is the server declining to answer. 502 is our own
 * "the body was not JSON", which in dev is Vite handing back the SPA for a
 * refused connection. A bare `TypeError` is `fetch` failing to connect at all,
 * which is what production looks like when `serve` is down — there is no proxy
 * there to dress it up.
 */
export function isTransient(error: unknown): boolean {
  if (error instanceof ApiError) return error.status === 502;
  return error instanceof TypeError;
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
 * Decode a successful response, naming the failure it can still have.
 *
 * A 200 whose body is not JSON is a TRANSPORT failure, and saying so is the
 * whole job here. It used to be reported as a schema mismatch — *"did not match
 * the expected shape — Expected object, received null"* — which names the
 * contract and sends the reader to the wrong file. The actual cause is upstream:
 * under `console-dev` the Vite proxy answers a refused connection with the SPA's
 * own `index.html` at status 200, so every panel blamed its payload while the
 * real fact was that `serve` was not running. That is a five-minute detour per
 * occurrence and it recurs, because the dev server outlives the backend by
 * design.
 */
function decode<T>(path: string, raw: string, response: Response): T {
  try {
    return JSON.parse(raw) as T;
  } catch {
    throw new ApiError(
      `${path} answered ${response.status} but the body is not JSON. The API server is probably not running — start it with "just console" (or "poker-solver serve").`,
      502,
    );
  }
}

/** A read. Generic over the payload type, which comes from `types.ts`. */
export async function get<T>(path: string): Promise<T> {
  const response = await fetch(path, { headers: { accept: "application/json" } });
  const raw = await response.text();

  if (!response.ok) {
    // The server puts the reason in `error` for exactly this: 422 is a refusal
    // (an unpublished run), 503 is the cloud (an expired `az login`). Both are
    // sentences meant for a person.
    throw new ApiError(reasonFrom(raw, response), response.status);
  }
  return decode<T>(path, raw, response);
}

/**
 * A mutation. The ONLY other verb this module speaks, and it exists for one
 * reason: play is stateful, so sitting down and acting are not reads.
 */
export async function send<T>(
  path: string,
  body?: unknown,
  method: "POST" | "DELETE" = "POST",
): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: { accept: "application/json", "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  const raw = await response.text();

  if (!response.ok) {
    throw new ApiError(reasonFrom(raw, response), response.status);
  }

  // The same transport-versus-payload rule as `get`, and it matters MORE here: a
  // dispatch that queued a task and then failed to read its own receipt must not
  // read as "the submission was malformed". It was not — the work is on the
  // pool, and the message has to point at the response, not the request.
  return decode<T>(path, raw, response);
}
