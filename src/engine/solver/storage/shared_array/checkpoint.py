"""Checkpoint load/save helpers for SharedArrayStorage."""

from __future__ import annotations

import pickle
import time
from typing import TYPE_CHECKING

import numcodecs
import numpy as np
import zarr

from src.engine.solver.storage.array_specs import ARRAY_SPECS
from src.engine.solver.storage.helpers import (
    CheckpointPaths,
    _validate_action_signatures,
    build_legal_actions,
    get_missing_checkpoint_files,
    load_action_signatures,
    load_checkpoint_data,
    load_key_mapping,
)

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
    dense_ids = {key: idx for idx, (key, _) in enumerate(items)}

    paths = CheckpointPaths.from_dir(storage.checkpoint_dir)
    with open(paths.key_mapping, "wb") as f:
        pickle.dump({"owned_keys": dense_ids}, f)

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

    action_sigs = {}
    for new_id, (_, old_id) in enumerate(items):
        actions = storage.state.legal_actions_cache.get(old_id)
        if actions is None:
            continue
        action_sigs[new_id] = [(action.type.name, action.amount) for action in actions]

    with open(paths.action_signatures, "wb") as f:
        pickle.dump(action_sigs, f)

    _validate_action_signatures(
        dense["action_counts"],
        action_sigs,
        f"SharedArrayStorage.checkpoint(iter={iteration})",
    )


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
    mapping_data = load_key_mapping(paths)
    saved_owned_keys = mapping_data["owned_keys"]
    saved_action_sigs = load_action_signatures(paths)

    if not saved_owned_keys:
        return True

    my_keys = []
    my_old_ids = []
    for key, old_id in saved_owned_keys.items():
        if storage.get_owner(key) == storage.worker_id:
            my_keys.append(key)
            my_old_ids.append(old_id)

    if not my_keys:
        print(f"Worker {storage.worker_id} owns 0/{len(saved_owned_keys)} keys from checkpoint")
        return True

    my_old_ids_array = np.array(my_old_ids, dtype=np.int32)

    data = load_checkpoint_data(
        storage.checkpoint_dir, context="SharedArrayStorage.load_checkpoint"
    )

    if data.max_actions != storage.max_actions:
        raise ValueError(
            f"Checkpoint max_actions mismatch: {data.max_actions} vs {storage.max_actions}"
        )

    if storage.capacity < data.max_id + 1:
        raise ValueError(f"Storage capacity too small: {storage.capacity} vs {data.max_id + 1}")

    new_ids = []
    for key in my_keys:
        new_id = storage.allocate_id()
        storage.state.owned_keys[key] = new_id
        new_ids.append(new_id)

    new_ids_array = np.array(new_ids, dtype=np.int32)

    for spec in ARRAY_SPECS:
        array = getattr(storage, spec.attr)
        saved = data.arrays[spec.checkpoint_key]
        if spec.per_action:
            array[new_ids_array, :] = saved[my_old_ids_array, :]
        else:
            array[new_ids_array] = saved[my_old_ids_array]

    for new_id, old_id in zip(new_ids, my_old_ids):
        legal_actions = build_legal_actions(
            saved_action_sigs, old_id, "SharedArrayStorage.load_checkpoint"
        )
        storage.state.legal_actions_cache[new_id] = legal_actions

    print(
        f"Worker {storage.worker_id} loaded {len(my_keys)}/{len(saved_owned_keys)} "
        "infosets from checkpoint"
    )
    return True
