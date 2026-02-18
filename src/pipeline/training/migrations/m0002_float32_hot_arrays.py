"""Migration 0002: float32 regrets/strategies (v1 → v2).

The two hot per-action arrays moved from float64 to float32 (training is
DRAM-bandwidth-bound at production tree sizes; halving bytes per touch is the
cheapest speedup). This step downcasts an existing checkpoint's arrays to the
dtypes declared in ``ARRAY_SPECS``. APPROXIMATE by nature: the downcast rounds
to ~7 significant digits, far below sampling noise, but the fingerprint changes.
"""

from __future__ import annotations

from pathlib import Path

import numcodecs
import numpy as np
import zarr

from src.engine.solver.storage.array_specs import ARRAY_SPECS
from src.engine.solver.storage.helpers import CheckpointPaths, load_checkpoint_arrays
from src.engine.solver.storage.in_memory import InMemoryStorage
from src.pipeline.training.migrations.base import Migration, MigrationKind

# Mirrors the trainer's checkpoint parameters (benchmarked defaults).
_ZARR_CHUNK_SIZE = 50_000
_ZARR_COMPRESSION_LEVEL = 1


def _migrate(run_dir: Path) -> None:
    arrays = load_checkpoint_arrays(run_dir)
    zarr_path = CheckpointPaths.from_dir(run_dir).checkpoint_zarr

    store = zarr.DirectoryStore(zarr_path)
    old_root = zarr.open(store, mode="r")
    attrs = dict(old_root.attrs)

    max_actions = arrays["regrets"].shape[1]
    compressor = numcodecs.Blosc(
        cname="zstd", clevel=_ZARR_COMPRESSION_LEVEL, shuffle=numcodecs.Blosc.BITSHUFFLE
    )

    root = zarr.open(store, mode="w")
    for spec in ARRAY_SPECS:
        data = arrays[spec.checkpoint_key].astype(spec.dtype, copy=False)
        root.create_dataset(
            spec.checkpoint_key,
            data=data,
            chunks=(_ZARR_CHUNK_SIZE, max_actions) if spec.per_action else (_ZARR_CHUNK_SIZE,),
            compressor=compressor,
            dtype=spec.dtype,
        )
    root.attrs.update(attrs)


def _verify(run_dir: Path) -> None:
    arrays = load_checkpoint_arrays(run_dir)
    for spec in ARRAY_SPECS:
        actual = arrays[spec.checkpoint_key].dtype
        if actual != np.dtype(spec.dtype):
            raise ValueError(
                f"post-migration {spec.checkpoint_key} dtype {actual} != {np.dtype(spec.dtype)}"
            )
    storage = InMemoryStorage(checkpoint_dir=run_dir)
    if storage.num_infosets() <= 0:
        raise ValueError("post-migration checkpoint has no infosets")


MIGRATION = Migration(
    version=2,
    description="float32 regrets/strategies (bandwidth-bound training; downcast checkpoint).",
    kind=MigrationKind.APPROXIMATE,
    migrate=_migrate,
    verify=_verify,
)
