"""Which strategy of a checkpoint gets scored: the DCFR average, the current
iterate, or the average over a late WINDOW of training.

``strategy_sum`` accumulates ``t^gamma * reach * sigma_t`` with the weight
applied at add time and never revisited, so the window sum over ``(t0, T]`` is
exactly ``SS(T) - SS(t0)``. That identity is why a windowed average is
measurable from two retained rungs rather than a retrained run.

Every variant is written INTO ``strategy_sum`` after the load, so nothing
downstream changes: the same normaliser turns a stored row into a distribution,
and ``max(regret, 0)`` under it IS regret matching.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import zarr

from src.engine.solver.storage.static_checkpoint import (
    FingerprintMismatchError,
    StaticCheckpointManifest,
)

if TYPE_CHECKING:
    from src.engine.solver.storage.static_array import StaticArrayStorage

PolicyIterate = Literal["average", "current"]

# Weight on the pre-window average, added back so a row the window never
# visited keeps the full average instead of collapsing to the uniform fallback.
# A row with typical window mass is contaminated ~4e-4 by it; a row with almost
# no window evidence shrinks back toward the average, which is the honest
# answer where the window has nothing to say.
WINDOW_SHRINKAGE = 1e-3

# Slots per streamed block. The window variant never holds the earlier array
# whole: on the XL tree that is ~1 GB per worker on top of a load peak the
# fork-join's RAM cap already sizes from a steady footprint.
_BLOCK_SLOTS = 8_000_000


def assemble_policy(
    storage: StaticArrayStorage,
    checkpoint_dir: Path,
    *,
    iterate: PolicyIterate = "average",
    window_from: int | None = None,
) -> dict[str, Any]:
    """Rewrite ``storage.strategy_sum`` as the requested strategy. In place.

    Returns what the record needs to tell two rows apart, empty for the
    identity. ``window_from`` must be a retained rung of the same run.
    """
    if iterate == "current" and window_from is not None:
        raise ValueError(
            "policy_iterate='current' has no averaging window: the current iterate is "
            "one iteration's strategy, not an average over any range of them."
        )
    if iterate == "current":
        np.maximum(storage.regrets, 0, out=storage.strategy_sum)
        return {"policy_iterate": "current"}
    if window_from is None:
        return {}
    return _apply_window(storage, checkpoint_dir, window_from)


def _apply_window(
    storage: StaticArrayStorage, checkpoint_dir: Path, window_from: int
) -> dict[str, Any]:
    """``SS(T) - SS(t0)``, shrunk back toward ``SS(t0)`` where the window is empty."""
    manifest = StaticCheckpointManifest.read(checkpoint_dir)
    if manifest is None:
        raise FileNotFoundError(f"No static checkpoint manifest in {checkpoint_dir}")
    entry = manifest.entry_for(window_from)
    root = zarr.open(zarr.DirectoryStore(Path(checkpoint_dir) / entry["zarr"]), mode="r")
    expected = storage.tree.fingerprint()
    if root.attrs.get("fingerprint") != expected:
        raise FingerprintMismatchError(
            f"Window base {entry['zarr']} carries fingerprint "
            f"{root.attrs.get('fingerprint')}, expected {expected}."
        )
    earlier_sum = root["strategy_sum"]
    if earlier_sum.shape != storage.strategy_sum.shape:
        raise ValueError(
            f"Window base holds {earlier_sum.shape} slots, storage expects "
            f"{storage.strategy_sum.shape} — fingerprints matched, so this is a format bug."
        )
    empty = 0
    trained = 0
    for start in range(0, storage.strategy_sum.shape[0], _BLOCK_SLOTS):
        stop = min(start + _BLOCK_SLOTS, storage.strategy_sum.shape[0])
        earlier = earlier_sum[start:stop]
        # A basic slice is a view, so the in-place ops below rewrite the table.
        block = storage.strategy_sum[start:stop]
        np.subtract(block, earlier, out=block)
        np.maximum(block, 0, out=block)
        had = earlier > 0
        trained += int(np.count_nonzero(had))
        empty += int(np.count_nonzero(had & (block <= 0)))
        block += WINDOW_SHRINKAGE * earlier
    return {
        "avg_window_from": int(entry["iteration"]),
        "avg_window_shrinkage": WINDOW_SHRINKAGE,
        # How much of the trained table the window has nothing to say about,
        # and so falls back to the full average. A large share means the arm
        # measures windowing PLUS a coverage change, not windowing alone.
        "avg_window_empty_slot_fraction": (empty / trained) if trained else 0.0,
    }


__all__ = ("WINDOW_SHRINKAGE", "PolicyIterate", "assemble_policy")
