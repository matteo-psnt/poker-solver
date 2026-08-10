/**
 * Turning a filled-in form into a request body.
 *
 * The whole write contract rests on one rule: a field the operator left alone
 * must be ABSENT from the body, not present-and-empty. The server drops `null`
 * before `invoke`, so the command's own parser supplies every default — and
 * that only works if the client agrees about what "left alone" looks like.
 *
 * Getting it wrong is quiet in both directions. Sending `workers: 0` where the
 * field was blank pins a 32-core node to one worker; sending `config: ""` on a
 * `--run` continuation is harmless only because that flag happens to default to
 * the empty string. Neither reads as a bug at the call site.
 */

/** The fields actually filled in, ready to be sent. */
export function given(fields: Record<string, unknown>): Record<string, unknown> {
  const body: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(fields)) {
    if (value === undefined || value === null) continue;
    if (typeof value === "string" && value.trim() === "") continue;
    if (Array.isArray(value) && value.length === 0) continue;
    // An unchecked guard flag is not "false", it is unset — the two are the
    // same outcome here, and omitting keeps the command's `store_true` default
    // as the single place that decides what off means.
    if (value === false) continue;
    body[key] = value;
  }
  return body;
}

/**
 * A numeric field's value, or undefined if it has none.
 *
 * Undefined rather than 0 or NaN: both of those are values a command would act
 * on, and the field being blank means the operator did not ask for one. Commas
 * and spaces are tolerated because an iteration target is eight digits long and
 * nobody reads `25000000` without grouping it.
 */
export function int(raw: string): number | undefined {
  const cleaned = raw.replace(/[,\s_]/g, "");
  if (!cleaned) return undefined;
  const value = Number(cleaned);
  return Number.isFinite(value) && Number.isInteger(value) ? value : undefined;
}

/**
 * `--set key=value` overrides, one per line.
 *
 * A textarea rather than a repeatable control: these are typed from a note or a
 * previous command line, and pasting three lines has to work. Lines without an
 * `=` are dropped here rather than sent — the command would refuse them, but it
 * would refuse the whole submission, and losing a nine-field form to a stray
 * blank line is a worse trade than ignoring one.
 */
export function overrides(raw: string): string[] {
  return raw
    .split("\n")
    .map((line) => line.trim())
    .filter((line) => line.includes("="));
}

/**
 * `--at` rungs, as the command wants them: comma-separated, in one string.
 *
 * Normalised rather than passed through, so `10_000_000, 20000000` and
 * `10000000,20000000` are the same request. The command splits on commas and
 * strips, so this is defensive about the shape rather than about the values.
 */
export function rungs(raw: string): string {
  return raw
    .split(",")
    .map((part) => part.replace(/[\s_]/g, ""))
    .filter(Boolean)
    .join(",");
}
