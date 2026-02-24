"""Migration 0002: float32 regrets/strategies (v1 → v2).

The two hot per-action arrays moved from float64 to float32 (training is
DRAM-bandwidth-bound at production tree sizes; halving bytes per touch is the
cheapest speedup). This step downcasts an existing checkpoint's arrays to the
dtypes declared in ``ARRAY_SPECS``. APPROXIMATE by nature: the downcast rounds
to ~7 significant digits, far below sampling noise, but the fingerprint changes.

A v2 run still carries the pickled-key layout, which current loaders reject by
design, so both the migration and its verify resolve the zarr snapshot through
the pre-v3 manifest reader; load coherence is checked by m0003's verify and the
golden-chain fingerprint.
"""

from __future__ import annotations

from pathlib import Path

import numcodecs
import numpy as np
import zarr

from src.engine.solver.storage.array_specs import ARRAY_SPECS
from src.engine.solver.storage.helpers import CHECKPOINT_ZARR_DIR
from src.pipeline.training.migrations.base import Migration, MigrationKind, read_pre_v3_manifest

# Mirrors the trainer's checkpoint parameters (benchmarked defaults).
_ZARR_CHUNK_SIZE = 50_000
_ZARR_COMPRESSION_LEVEL = 1


def _zarr_path(run_dir: Path) -> Path:
    """Current snapshot's zarr directory, resolved manifest-aware for pre-v3 runs."""
    manifest = read_pre_v3_manifest(run_dir)
    name = CHECKPOINT_ZARR_DIR if manifest is None else manifest["zarr"]
    return run_dir / name


def _migrate(run_dir: Path) -> None:
    store = zarr.DirectoryStore(_zarr_path(run_dir))
    old_root = zarr.open(store, mode="r")
    attrs = dict(old_root.attrs)
    arrays = {spec.checkpoint_key: old_root[spec.checkpoint_key][:] for spec in ARRAY_SPECS}

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
    root = zarr.open(zarr.DirectoryStore(_zarr_path(run_dir)), mode="r")
    for spec in ARRAY_SPECS:
        actual = root[spec.checkpoint_key].dtype
        if actual != np.dtype(spec.dtype):
            raise ValueError(
                f"post-migration {spec.checkpoint_key} dtype {actual} != {np.dtype(spec.dtype)}"
            )


MIGRATION = Migration(
    version=2,
    description="float32 regrets/strategies (bandwidth-bound training; downcast checkpoint).",
    kind=MigrationKind.APPROXIMATE,
    migrate=_migrate,
    verify=_verify,
)
