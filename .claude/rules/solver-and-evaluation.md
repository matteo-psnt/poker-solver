---
paths:
  - "src/pipeline/**"
  - "src/engine/**"
  - "src/core/**"
  - "tests/pipeline/**"
  - "tests/engine/**"
---

# Solver, training and evaluation

**One solver backend: the statically-enumerated tree.** An infoset is
`(node_id, bucket)` — an index into a table allocated once at full size, so
memory is flat in iteration count. Runs are loadable iff they carry
`STATIC_CHECKPOINT.json`; checkpoints from the deleted dynamic backend are
unreadable by design.

- **EXPLOITABILITY ALONE CANNOT PROMOTE A CHANGE.** It is a worst-case bound, so
  it is maximised by refusing to play: folding every hand scores
  `1000*(sb+bb)/(2*bb)` = **750 mbb/g**, and the arms this project called its
  best read 722-748. `evaluate` prints that floor beside every exact-BR score —
  an arm near it is not a poker strategy whatever its trend. Exploitability is
  separately known not to track one-on-one play (Waugh et al. 2009; Johanson et
  al., AAMAS 2013), and we measured the reversal: `river-r4` beats `turn-river`
  by 41 AIVAT bb/100 against GTO Wizard while exploitability ranks them the
  other way.
- **A change ships only if it passes all three.** They fail in different
  directions, so no one of them is enough:
  1. **Exploitation** — `score --method blueprint_match --run A --at N --vs B
     --vs-at N`, on the POOL, one task per rung (`match` is the local door for
     two runs already on the box). Duplicate deals, seats swapped off one deck.
     Folding cannot win a chip match. **Pass `--deals 20000`**: the 2,000
     default carries SE ~150 mbb and read "inconclusive" on an effect 20,000
     resolved at ~49, and a match is only ~40 s of node time. **Always run the
     same-recipe NULL** — two runs of ONE recipe came out +53.3 ± 22.9 mbb/hand
     apart (2.33σ), so a single pairing cannot separate a recipe from the arm
     lottery; replicate at the ARM level (≥2 training seeds a side), not by
     adding deals.
  2. **Progress** — the same match against a FROZEN reference arm. One-on-one
     results can be intransitive, which is harmless against a fixed yardstick
     and fatal when ranking a field.
  3. **Safety** — exploitability and LBR, read WITH the 750 floor.
  Quote VPIP beside any score (`--policy-profile`): an arm folding the SB >50%
  is not a candidate. For postflop skill use `--decompose` postflop / postflop
  reach — the total is reach-weighted, so a folder hides a broken postflop in it.
- **Never compare arms across knob tiers.** Nothing enforces this; check
  `base_seed` and every tier knob before putting two numbers together, and
  never hand-transcribe a score. `poker-solver arms --experiment X --control Y`
  reads the tags back and subtracts exact_br arms directly — the estimator is
  deterministic, so no p-value is needed.
- **Scores live in the record (Postgres), not in per-run files.** `evaluate`
  writes a row through the record sink; `ledger`, `curve` and `arms` read it.
  **NOTHING writes `<run_dir>/evals/*.json` any more** — the last writer went
  with the head-to-head gate's move to the record, and `rebuild_ledger` only
  READS the pre-sink corpus. So `prune-checkpoints`'s `_scored_iterations`
  answers for legacy runs alone; it reads the record for the rest.
  **Score before pruning.**
- **`--config production` is `cfr_br: off` as of 09-24, and EXPLOITABILITY IS
  RETIRED as the criterion that chose it.** The river-legal responder it used to
  carry won the exact-BR gate (748.6 mbb thresholded) and that gate was measuring
  folding: CFR-BR folds the SB first-in 65-80%, the same recipe with it off folds
  10.1%, and real heads-up folds 10-20%. At matched rung 4,000 the 2.51x
  exploitability advantage is a REACH ARTEFACT — preflop loss is FLAT (694 vs 757
  mbb) and the whole difference is postflop (106 vs 1,252), because a strategy
  that folds never reaches its own postflop. **The postflop was never solved, it
  was hidden.** On chips the playing arms won 11 of 12 arm-seed matches over
  20,000 duplicate deals (p = 0.0063), and within ONE run rung 4,000 beats rung
  1,200 by +81.9 ± 27.5 mbb/hand (2.98σ) while scoring 20% WORSE on the bound.
  Read `settle` for whether training is still buying anything, `blueprint_match`
  against the fielded champion for whether a change is an improvement, and the SB
  fold rate as a shape check. `runouts_per_flop=4` alone reaches 820.7 and R=8 is
  WORSE than R=4. A treatment arm is a YAML that `extends: production.yaml` and
  moves ONE thing (`production_linplus.yaml` is the pattern); it is deleted when
  the question closes.
- **Experiment bookkeeping** goes through `--experiment`/`--arm`/`--parent`,
  with `--set k=v` for config overrides on a FRESH run; the tags are recorded
  on every eval. **A continuation trains what it was**: `submit --run X --to N`
  takes no `--config` and no `--set`, because the node reloads the config off
  the run's record. Runs continued before 09-13 could silently revert to the
  stem's defaults at the boundary — check `runinfo` for a continuation before
  reading a mid-ladder turn in an older ladder as a result.
- **`reference/` may not import the estimators it validates.** The oracles
  check the production estimators to 1e-9; an oracle importing what it
  validates makes that agreement circular while the test still passes. This is
  an import-linter contract, and the one that is about correctness.
- Training and evaluation may not import each other (contract). Keep the
  interfaces between solver, training and evaluation typed and explicit.
- Tests are deterministic: fixed seeds, no wall-clock assertions. Expensive
  tests are `@pytest.mark.slow`; the default timeout is 5s and a longer one is
  declared with `@pytest.mark.timeout(<seconds>)`.
- Run the **full** suite when a change touches training, abstraction/bucketing,
  evaluator logic, config loading, or shared infrastructure.
