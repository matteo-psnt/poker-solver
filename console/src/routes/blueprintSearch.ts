import { z } from "zod";

/** The chart's spot, in the URL. Its own module so the public chart (`chart.tsx`) can share it
 * without bundling the console's route tree. */
export const blueprintSearch = z.object({
  tab: z.enum(["chart", "play"]).default("chart"),
  path: z.string().default(""),
  board: z.string().default(""),
  // How many of `path`'s actions the chart reads; absent reads all of them.
  at: z.number().int().nonnegative().optional(),
  // The DEPTH, when the server holds a ladder. Optional rather than defaulted:
  // absent means "the deepest", which is what a single-run server has, and
  // baking a number in would make a shared link wrong on a different box.
  stack: z.number().optional(),
});
