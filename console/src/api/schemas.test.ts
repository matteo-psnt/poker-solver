import { describe, expect, it } from "vitest";
import fixture from "./payloads.fixture.json";
import {
  autoscaleSchema,
  compactSchema,
  compareSchema,
  configsSchema,
  costSchema,
  curveSchema,
  dispatchedSchema,
  jobsSchema,
  ledgerSchema,
  poolSchema,
  promoteSchema,
  pushCodeSchema,
  pushDataSchema,
  reportSchema,
  runinfoSchema,
  runsSchema,
  tasksSchema,
} from "./schemas";

/**
 * The payload contract, checked without a server.
 *
 * The fixture is generated from `PAYLOADS` in
 * `tests/interfaces/cli/test_command_renderers.py` — shapes already pinned by a
 * passing Python test — so this is not a mock that can drift into fiction. A
 * Python-side change fails the generator test, which regenerates the fixture,
 * which fails this one BY NAME.
 */
const CASES = [
  ["pool-status", poolSchema],
  ["jobs", jobsSchema],
  ["tasks", tasksSchema],
  ["runs", runsSchema],
  ["runinfo", runinfoSchema],
  ["curve", curveSchema],
  ["ledger", ledgerSchema],
  ["cost", costSchema],
  ["configs", configsSchema],
  ["autoscale-check", autoscaleSchema],
  ["report", reportSchema],
  ["compare", compareSchema],
  // All three dispatches share one schema, so all three are checked against it.
  // A key that only `score` carries would otherwise be nobody's contract.
  ["submit", dispatchedSchema],
  ["score", dispatchedSchema],
  ["submit-precompute", dispatchedSchema],
  ["push-code", pushCodeSchema],
  ["push-data", pushDataSchema],
  ["compact-legs", compactSchema],
  ["promote", promoteSchema],
] as const;

describe("schemas parse the payloads Python actually emits", () => {
  for (const [op, schema] of CASES) {
    it(op, () => {
      const payload = (fixture as Record<string, unknown>)[op];
      expect(payload, `no '${op}' in the fixture`).toBeDefined();
      const parsed = schema.safeParse(payload);
      if (!parsed.success) {
        throw new Error(`${op}: ${JSON.stringify(parsed.error.issues, null, 2)}`);
      }
    });
  }
});

it("the fixture is the generated one, not a hand-edited copy", () => {
  expect((fixture as { _: string })._).toContain("GENERATED");
});
