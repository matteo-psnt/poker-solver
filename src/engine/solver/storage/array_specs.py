"""Declarative description of the shared arrays backing SharedArrayStorage.

Every place that must treat "the shared arrays" as a set — shm segment
creation/attachment, numpy view construction, cleanup, resize copy-over, and
checkpoint save/load/validation — loops over :data:`ARRAY_SPECS` instead of
hand-enumerating the five arrays. Adding an array (e.g. a per-infoset iteration
stamp) means adding one spec line plus its semantic consumers.

Deliberately NOT spec-driven: per-infoset view attachment in
``shared_array/infoset.py`` (``action_counts`` is metadata there, not a CFR
accumulator, and regrets/strategy get row views while the scalars get element
views) and ``in_memory.py``'s InfoSet construction — both consume the arrays by
meaning, not as a uniform set.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ArraySpec:
    """One shared array: its storage attribute, shm plumbing, dtype, and layout."""

    # Attribute name on SharedArrayStorage holding the numpy view.
    attr: str
    # SharedMemory-handle attribute on SharedArrayMutableState.
    shm_attr: str
    # Session-namespaced shared-memory name base.
    shm_base: str
    # On-disk Zarr dataset name. NOT always the attr suffix: the strategy_sum
    # array has always been checkpointed as "strategies", and renaming would
    # break every existing checkpoint.
    checkpoint_key: str
    dtype: type[np.generic]
    # True → one row of ``max_actions`` per infoset; False → one scalar per infoset.
    per_action: bool

    def shape(self, capacity: int, max_actions: int) -> tuple[int, ...]:
        return (capacity, max_actions) if self.per_action else (capacity,)

    def nbytes(self, capacity: int, max_actions: int) -> int:
        cells = capacity * (max_actions if self.per_action else 1)
        return cells * np.dtype(self.dtype).itemsize


ARRAY_SPECS: tuple[ArraySpec, ...] = (
    # float32 for the two hot per-action arrays: training is DRAM-bandwidth-bound
    # past ~10M infosets, and halving the bytes per touch is the cheapest lever.
    # CFR tolerates the precision (sampling noise dwarfs float32 rounding; Pluribus
    # stored regrets as integers). Representation v2; migration m0002 downcasts
    # older checkpoints.
    ArraySpec("shared_regrets", "shm_regrets", "sas_reg", "regrets", np.float32, True),
    ArraySpec("shared_strategy_sum", "shm_strategy", "sas_str", "strategies", np.float32, True),
    ArraySpec("shared_action_counts", "shm_actions", "sas_act", "action_counts", np.int32, False),
    ArraySpec("shared_reach_counts", "shm_reach", "sas_reach", "reach_counts", np.int64, False),
    ArraySpec(
        "shared_cumulative_utility",
        "shm_utility",
        "sas_util",
        "cumulative_utility",
        np.float64,
        False,
    ),
)
