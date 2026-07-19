"""Checkpoint load/save helpers for SharedArrayStorage."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import numcodecs
import numpy as np
import zarr

from src.engine.solver.storage import key_table
from src.engine.solver.storage.array_specs import ARRAY_SPECS
from src.engine.solver.storage.helpers import (
    CheckpointPaths,
    commit_checkpoint_manifest,
    get_missing_checkpoint_files,
    load_checkpoint_arrays,
    validate_action_counts,
)
from src.engine.solver.storage.shared_array.ownership import stable_hash

if TYPE_CHECKING:
    from src.engine.solver.storage.shared_array.storage import SharedArrayStorage


def checkpoint_storage(storage: SharedArrayStorage, iteration: int) -> None:
    """
    Save checkpoint to disk (coordinator only).

    Uses Zarr format with ZStd compression for fast I/O and chunked storage.
    """
    if not storage.checkpoint_dir or not storage.is_coordinator:
        return

    storage.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    num_keys = len(storage.state.owned_keys)
    if num_keys == 0:
        return

    items = sorted(storage.state.owned_keys.items(), key=lambda item: item[1])

    # Write a fresh versioned snapshot; the previous one stays intact until the
    # manifest commit below flips to this one atomically.
    paths = CheckpointPaths.for_iteration(storage.checkpoint_dir, iteration)

    old_ids = np.array([old_id for (_, old_id) in items], dtype=np.int32)

    dense: dict[str, np.ndarray] = {}
    for spec in ARRAY_SPECS:
        array = getattr(storage, spec.attr)
        dense[spec.checkpoint_key] = (
            array[old_ids, :].copy() if spec.per_action else array[old_ids].copy()
        )

    store = zarr.DirectoryStore(paths.checkpoint_zarr)
    root = zarr.open(store, mode="w")

    compressor = numcodecs.Blosc(
        cname="zstd",
        clevel=storage.zarr_compression_level,
        shuffle=numcodecs.Blosc.BITSHUFFLE,
    )

    chunk_size = storage.zarr_chunk_size

    for spec in ARRAY_SPECS:
        root.create_dataset(
            spec.checkpoint_key,
            data=dense[spec.checkpoint_key],
            chunks=(chunk_size, storage.max_actions) if spec.per_action else (chunk_size,),
            compressor=compressor,
            dtype=spec.dtype,
        )

    root.attrs["iteration"] = iteration
    root.attrs["num_infosets"] = len(items)
    root.attrs["max_actions"] = storage.max_actions
    root.attrs["timestamp"] = time.time()
    root.attrs["format_version"] = "1.0"

    # Row order IS the dense id, so keys, hashes and action lists are written as
    # parallel columns and no id column is needed. The hash is written from the
    # same function ownership uses, so a reader can shard without rebuilding keys.
    keys = [key for (key, _) in items]
    action_lists = [storage.state.legal_actions_cache.get(old_id) for (_, old_id) in items]
    key_table.write_key_table(
        paths.key_table,
        keys=keys,
        key_hashes=[stable_hash(key) for key in keys],
        action_lists=action_lists,
    )

    validate_action_counts(
        dense["action_counts"],
        [actions or () for actions in action_lists],
        f"SharedArrayStorage.checkpoint(iter={iteration})",
    )

    # All artifacts written and validated — make this snapshot current.
    commit_checkpoint_manifest(storage.checkpoint_dir, iteration, paths)


def load_storage_checkpoint(storage: SharedArrayStorage) -> bool:
    """
    Load checkpoint from disk and restore this worker's owned partition.

    Returns:
        True if checkpoint was loaded, False otherwise.
    """
    if not storage.checkpoint_dir:
        return False

    missing_files = get_missing_checkpoint_files(storage.checkpoint_dir)
    if missing_files:
        return False

    paths = CheckpointPaths.from_dir(storage.checkpoint_dir)
    total_rows = key_table.num_rows(paths.key_table)
    if total_rows == 0:
        return True

    # Read ONLY this worker's shard. Reading the whole table here (which is what the
    # pickled format forced) cost ~28 GB per worker at 18.9M keys and is what made a
    # 16-worker resume OOM; ownership is decided on a memory-mapped hash column.
    owned = key_table.read_owned_rows(paths.key_table, storage.num_workers, storage.worker_id)
    my_keys = owned.keys
    if not my_keys:
        print(f"Worker {storage.worker_id} owns 0/{total_rows} keys from checkpoint")
        return True

    my_old_ids_array = owned.row_ids.astype(np.int32, copy=False)

    arrays = load_checkpoint_arrays(storage.checkpoint_dir)
    max_actions = arrays["regrets"].shape[1]

    if max_actions != storage.max_actions:
        raise ValueError(f"Checkpoint max_actions mismatch: {max_actions} vs {storage.max_actions}")

    if storage.capacity < total_rows:
        raise ValueError(f"Storage capacity too small: {storage.capacity} vs {total_rows}")

    new_ids = []
    for key in my_keys:
        new_id = storage.allocate_id()
        storage.state.owned_keys[key] = new_id
        storage.state.unshipped_keys.append((key, new_id))
        new_ids.append(new_id)

    new_ids_array = np.array(new_ids, dtype=np.int32)

    for spec in ARRAY_SPECS:
        array = getattr(storage, spec.attr)
        saved = arrays[spec.checkpoint_key]
        if spec.per_action:
            array[new_ids_array, :] = saved[my_old_ids_array, :]
        else:
            array[new_ids_array] = saved[my_old_ids_array]

    for new_id, legal_actions in zip(new_ids, owned.action_lists):
        storage.state.legal_actions_cache[new_id] = legal_actions

    print(f"Worker {storage.worker_id} loaded {len(my_keys)}/{total_rows} infosets from checkpoint")
    return True
