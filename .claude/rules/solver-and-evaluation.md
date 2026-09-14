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
  1. **Exploitation** — `poker-solver match --run A --vs B`, duplicate deals
     (seats swapped off one deck, so ~2k deals do what ~46k unpaired would).
     Folding cannot win a chip match.
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
  `<run_dir>/evals/*.json` is written only by resolver-match scoring now, so
  code that reads that directory to decide something sees almost nothing —
  `prune-checkpoints` protects scored rungs from it and cannot see a DB-scored
  rung. **Score before pruning**, and fix that reader before trusting it.
- **`--config production` IS the best measured PCS/CFR-BR setup** — a
  river-legal best response over four runouts sharing a flop and a turn
  (`river-r4`, 748.6 mbb thresholded on the gate — which IS the fold-everything
  floor, so read it as the best of a bad field, not as good). Most of that is
  plain K-fold averaging: `runouts_per_flop=4` alone reaches 820.7, and R=8
  is WORSE than R=4. A treatment arm is a YAML that `extends: production.yaml`
  and moves ONE thing (`production_linplus.yaml` is the pattern); it is deleted
  when the question closes.
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
