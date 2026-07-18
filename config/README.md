# Configuration Guide

This project has two configuration families:

- **Training/runtime config** (`Config`) in `src/shared/config.py`
- **Combo abstraction precompute config** (`PrecomputeConfig`) in
  `src/pipeline/abstraction/config.py`

Both are strict, frozen Pydantic models (`extra="forbid"`), so unknown keys
fail validation — typos in YAML are caught at load time.

## How loading works

### Training/runtime (`Config`)

Defaults live in Python (`src/shared/config.py`); YAML files provide
**overrides only**. Loaders in `src/shared/config_loader.py`:

```python
load_config("config/training/production.yaml", training__num_iterations=500)
load_training_config("production", system__seed=7)   # by name
```

Resolution order (last wins):

1. Python field defaults
2. YAML file — supports `extends: <filename>` chains (same directory;
   current file's values win over the base)
3. Programmatic keyword overrides, using `__` as the nesting separator
   (`training__num_iterations=50_000`)

Persisted run snapshots are reloaded with `Config.from_persisted_dict`,
which tolerates schema drift by pruning fields that no longer exist
(logged), so old runs stay loadable.

### Abstraction precompute (`PrecomputeConfig`)

Loaded by name: `PrecomputeConfig.from_yaml("<name>")` reads
`config/abstraction/<name>.yaml`. `config_name` is set from the filename —
do not put it in the YAML.

## Training config reference (`Config`)

Sections and fields, with defaults:

| Section | Field | Default | Notes |
|---|---|---|---|
| `training` | `num_iterations` | 100000 | |
| | `checkpoint_frequency` | 50000 | iterations between checkpoints |
| | `iterations_per_worker` | 1000 | batch size = this × num_workers |
| | `verbose` | true | |
| | `runs_dir` | `data/runs` | |
| `storage` | `initial_capacity` | 2000000 | max infosets in shared arrays |
| | `max_actions` | 10 | |
| | `checkpoint_enabled` | true | |
| | `max_checkpoint_overhead` | 0.1 | back-pressure: cap checkpointing at ~10% of wall-clock |
| | `zarr_compression_level` | 1 | 1-9; 1 benchmarked fastest *and* smallest |
| | `zarr_chunk_size` | 50000 | |
| `system` | `seed` | null | |
| | `config_name` | `default` | shown in run metadata |
| | `log_level` | `INFO` | |
| `game` | `starting_stack` | 200 | BB units |
| | `small_blind` / `big_blind` | 1 / 2 | validated: BB > SB |
| `action_model` | `preflop_templates` | see schema | required keys: `sb_first_in`, `bb_vs_limp`, `bb_vs_open`, `sb_vs_3bet`, `bb_vs_4bet`, `sb_vs_5bet` |
| | `postflop_templates` | see schema | required keys: `first_aggressive`, `facing_bet`, `after_one_raise`, `after_two_raises` |
| | `jam_spr_threshold` | 2.0 | |
| | `raise_count_rules` | see schema | maps `facing_1/2/3_plus` → postflop template |
| | `off_tree_mapping` | `probabilistic` | or `nearest` |
| | `version` | 1 | |
| `resolver` | `enabled` | true | |
| | `time_budget_ms` | 300 | |
| | `max_depth` | 2 | |
| | `max_raises_per_street` | 2 | production overrides to 5 |
| | `leaf_rollouts` | 8 | board runouts for leaf valuation |
| | `policy_blend_alpha` | 0.35 | resolver↔blueprint blend |
| | `min_strategy_prob` | 1e-6 | |
| | `max_iterations` | null | fixed CFR count; determinism knob (null = wall-clock budget) |
| `solver` | `sampling_method` | `external` | or `outcome` |
| | `cfr_plus` | true | |
| | `iteration_weighting` | `linear` | `none` \| `linear` \| `dcfr`; production uses `dcfr` |
| | `dcfr_alpha/beta/gamma` | 1.5 / 0.0 / 2.0 | used only with `dcfr` |
| | `enable_pruning` | false | requires external sampling |
| | `pruning_threshold` | 300.0 | |
| | `prune_start_iteration` | 100 | |
| | `prune_reactivate_frequency` | 100 | |
| `card_abstraction` | `config` | `default` | name of a `config/abstraction/` preset |

Template tokens are validated: preflop accepts `fold`/`call`/`check`/jam
tokens, numeric open sizes, and multiplier tokens like `"3.5x_open"` /
`"2.3x_last"`; postflop accepts pot fractions and
`min_raise`/`pot_raise`/`jam`.

## Add or update a training config (`config/training/*.yaml`)

1. Add a YAML file under `config/training/` containing **only the keys you
   override** (see `production.yaml`; `default.yaml` is a fully commented
   template of every field).
2. Set `system.config_name` to the profile name you want in run metadata.
3. Optionally `extends:` another YAML in the directory.
4. If a field should be CLI-editable, update prompts in
   `src/interfaces/cli/flows/config.py`.

## Add a new training config field (schema change)

1. Add the field, with default and validation, to the right model in
   `src/shared/config.py` — constraints live there, nowhere else.
2. Wire usage where relevant, typically
   `src/pipeline/training/components.py`,
   `src/pipeline/training/trainer/`, or `src/engine/solver|search/`.
3. Optionally expose it in `src/interfaces/cli/flows/config.py`.
4. Add/update tests in the mirrored test packages and document the field in
   `config/training/default.yaml`.

## Abstraction precompute reference (`PrecomputeConfig`)

```yaml
# config/abstraction/default.yaml
buckets:              # equity buckets per street (required nested keys)
  flop: 50
  turn: 100
  river: 200
flop_runouts: null    # null = exact (all 1,176 runouts); int = sampled
equity_histogram_bins: 8
kmeans_max_iter: 300
kmeans_n_init: 10
num_workers: null     # null = all cores
seed: 42
```

Workflow:

1. Add a YAML under `config/abstraction/`.
2. Precompute: `uv run poker-solver` → "Combo Abstraction Tools" →
   "Precompute Abstraction".
3. Reference it from a training config: `card_abstraction.config: "<name>"`.

The abstraction identity hash (`get_config_hash`) covers `buckets`,
`flop_runouts`, and `equity_histogram_bins` only — changing any of these
produces a new artifact directory
(`data/combo_abstraction/buckets-F..T..R..-r..-<hash>/`) and requires
re-running precompute. `kmeans_*`, `num_workers`, and `seed` do not change
the identity.

## Where config names and hashes are used

- Training runs record `config_name`, `action_config_hash`, and
  `card_abstraction_hash` in `.run.json` via `RunTracker`
  (`src/pipeline/training/run_tracker.py`).
- Abstractions are resolved by config name/hash through
  `AbstractionResolver` and `build_card_abstraction(...)`
  (`src/pipeline/training/abstraction_resolver.py`,
  `src/pipeline/training/components.py`).
- Evaluation auto-pins to a run's recorded `card_abstraction_hash` and
  refuses runs without one, so a strategy is never scored under a different
  abstraction than it was trained with.
