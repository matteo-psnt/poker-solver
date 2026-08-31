# Schema

`0001_initial.sql` and successors are applied in order by number. **Forward
only**, and **additive only** while any dispatch is in flight — which is nearly
always, since a sealed code snapshot outlives the session that dispatched it by
24–36 hours. A node running last week's code must keep writing successfully:
new columns are nullable, nothing is renamed, nothing narrows a type.

`schema.sql` is not maintained by hand. The intended guard is a test that
applies every migration to an empty database and diffs the result against a
canonical dump — the same shape as the golden-numbers lineage guard, where a
failure means two descriptions of reality have diverged.

**That guard is not written yet, because there is no database yet.** Deciding
how CI gets a Postgres is a real decision and not one to smuggle in with a DDL
file.

## What is deliberately NOT here

- **`tier` as a column.** It is an ordinal into a query result, not a stored
  value. Pairing goes through `evals.tier_digest`, computed in Python by the
  same `tier_key` code path that has always owned the rule.
- **`checkpoints.complete`.** One rung is one atomically-committed blob, so blob
  existence *is* completeness. Every row in `checkpoints` is a derived cache,
  rebuildable from a single prefix listing, and never a claim.
- **`legs.cause` / `cause_source`.** Derived at read time from
  `(exit.cause, observed.state)` by one precedence rule. Materialising it would
  freeze that rule into whichever writer happened to run first.
