"""What a run is configured with: the model, how one is loaded, how two merge.

    schema.py  the frozen pydantic tree -- Config and everything under it
    loader.py  YAML in, `Config` out, with `--set k=v` overrides applied
    merge.py   recursive dict merge, which is how an override wins

Three files that were three loose modules in `shared/`, and only ever spoke to
each other: `deep_merge_dicts` had exactly two callers in the whole tree, both
of them here, so a generic-sounding `shared/dicts.py` was really a config
implementation detail sitting where anything could pick it up.

The model is re-exported here because that is the name 22 call sites already
use. `from src.shared.config import Config` means what it has always meant.
"""

from src.shared.config.schema import (
    DEFAULT_RUNS_DIR,
    ActionModelConfig,
    CardAbstractionConfig,
    Config,
    GameConfig,
    ResolverConfig,
    SolverConfig,
    StorageConfig,
    StrictFrozenModel,
    SystemConfig,
    TrainingConfig,
)

__all__ = [
    "DEFAULT_RUNS_DIR",
    "ActionModelConfig",
    "CardAbstractionConfig",
    "Config",
    "GameConfig",
    "ResolverConfig",
    "SolverConfig",
    "StorageConfig",
    "StrictFrozenModel",
    "SystemConfig",
    "TrainingConfig",
]
