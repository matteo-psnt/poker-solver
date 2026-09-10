# Configuration

Two strict, frozen Pydantic models (`extra="forbid"`, so a typo in YAML is
refused at load time). **The schema is the reference**: every field, its
default, its constraint and the comment saying why it exists lives next to the
field, nowhere else.

- **Training** — `Config` in `src/shared/config/schema.py`; presets in
  `config/training/<name>.yaml`.
- **Card abstraction precompute** — `PrecomputeConfig` in
  `src/pipeline/abstraction/config.py`; presets in
  `config/abstraction/<name>.yaml`.

`poker-solver configs` lists the presets `submit` and `submit-precompute` accept.

## A training preset is overrides, resolved last-wins

1. Python field defaults.
2. The YAML, which may `extends: <file>` another in the same directory (the
   current file wins over its base).
3. `--set section__field=value` on the command line.

```python
load_training_config("production", system__seed=7)
```

`production.yaml` is the recipe the blueprint box fields. A treatment arm is a
YAML that `extends: production.yaml` and moves ONE thing, so its action and
abstraction hashes match the control by construction and an exact-BR score is a
matched tier; it is deleted when the question closes. `quick_test.yaml` is the
smoke-test preset. A persisted run snapshot reloads through
`Config.from_persisted_dict`, which drops fields the schema no longer has, so
old runs stay loadable across schema changes.

## An abstraction preset names an artifact

`PrecomputeConfig.from_yaml("<name>")` reads `config/abstraction/<name>.yaml`;
`config_name` is set from the filename, never written in the YAML. The identity
hash covers `buckets`, `flop_runouts` and `equity_histogram_bins` only, so
changing any of those is a new artifact (`buckets-F..T..R..-r..-<hash>` in the
`abstractions` container) that `submit-precompute --config <name>` must build;
`kmeans_*`, `num_workers` and `seed` do not change the identity.

A training run records the hash it was trained against, and evaluation pins to
that hash rather than to the name, so deleting a preset never makes an existing
run unevaluable — it only stops new runs from being trained on it.

## Adding a field

Add it to the right model in `schema.py` with its default, constraint and a
one-line comment on why it exists; wire it where it is read; test it in the
mirrored test package. Nothing else needs updating.
