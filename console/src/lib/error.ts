import { ApiError } from "@/api/client";

/**
 * The sentence to show a person, or null.
 *
 * The server already made its refusals readable — 422 carries the reason, 503
 * names `az login`. Rewriting them here would be a second vocabulary for the
 * same failures.
 */
export function errorOf(error: unknown): string | null {
  if (!error) return null;
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) return error.message;
  return String(error);
}
