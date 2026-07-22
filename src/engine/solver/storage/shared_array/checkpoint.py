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
    load_checkpoint_rows,
    validate_action_counts,
)
from src.engine.solver.storage.shared_array.ownership import stable_hash
from src.shared import checkpoint_profile

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

    with checkpoint_profile.phase("sort_owned_keys"):
        items = sorted(storage.state.owned_keys.items(), key=lambda item: item[1])

    # Write a fresh versioned snapshot; the previous one stays intact until the
    # manifest commit below flips to this one atomically.
    paths = CheckpointPaths.for_iteration(storage.checkpoint_dir, iteration)

    with checkpoint_profile.phase("gather_array_rows"):
        old_ids = np.array([old_id for (_, old_id) in items], dtype=np.int32)

        dense: dict[str, np.ndarray] = {}
        for spec in ARRAY_SPECS:
            array = getattr(storage, spec.attr)
            dense[spec.checkpoint_key] = (
                array[old_ids, :].copy() if spec.per_action else array[old_ids].copy()
            )

    with checkpoint_profile.phase("zarr_write"):
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
    with checkpoint_profile.phase("extract_keys_and_actions"):
        keys = [key for (key, _) in items]
        action_lists = [storage.state.legal_actions_cache.get(old_id) for (_, old_id) in items]

    with checkpoint_profile.phase("stable_hash"):
        key_hashes = [stable_hash(key) for key in keys]

    with checkpoint_profile.phase("write_key_table"):
        key_table.write_key_table(
            paths.key_table,
            keys=keys,
            key_hashes=key_hashes,
            action_lists=action_lists,
        )

    with checkpoint_profile.phase("validate_action_counts"):
        validate_action_counts(
            dense["action_counts"],
            [actions or () for actions in action_lists],
            f"SharedArrayStorage.checkpoint(iter={iteration})",
        )

    # All artifacts written and validated — make this snapshot current.
    with checkpoint_profile.phase("commit_manifest"):
        commit_checkpoint_manifest(storage.checkpoint_dir, iteration, paths)

    # File count, not byte size, is what a Modal Volume commit scales with.
    with checkpoint_profile.phase("measure_artifacts"):
        zarr_tree = checkpoint_profile.measure_tree(paths.checkpoint_zarr)
        table_tree = checkpoint_profile.measure_tree(paths.key_table)
    checkpoint_profile.add_stats(
        num_infosets=len(items),
        zarr_files=zarr_tree["files"],
        zarr_bytes=zarr_tree["bytes"],
        key_table_files=table_tree["files"],
        key_table_bytes=table_tree["bytes"],
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
    total_rows = key_table.num_rows(paths.key_table)
    if total_rows == 0:
        return True

    # Progress logging matters here specifically: for the num_workers=1 resume
    # bootstrap, this worker owns EVERY row, so each phase below is single-threaded
    # over the whole table -- minutes at production sizes. Without the per-phase
    # timing, resume emits only the trailing summary and looks hung the whole time.
    wid = storage.worker_id
    load_start = time.perf_counter()
    print(
        f"[checkpoint-load] worker {wid}: loading up to {total_rows:,} rows "
        f"(num_workers={storage.num_workers}) from {storage.checkpoint_dir}...",
        flush=True,
    )

    # Read ONLY this worker's shard. Reading the whole table here (which is what the
    # pickled format forced) cost ~28 GB per worker at 18.9M keys and is what made a
    # 16-worker resume OOM; ownership is decided on a memory-mapped hash column.
    phase_start = time.perf_counter()
    owned = key_table.read_owned_rows(paths.key_table, storage.num_workers, storage.worker_id)
    my_keys = owned.keys
    if not my_keys:
        print(f"Worker {wid} owns 0/{total_rows} keys from checkpoint", flush=True)
        return True
    print(
        f"[checkpoint-load] worker {wid}: read {len(my_keys):,}/{total_rows:,} owned keys "
        f"in {time.perf_counter() - phase_start:.1f}s",
        flush=True,
    )

    my_old_ids_array = owned.row_ids.astype(np.int32, copy=False)

    # Read only this worker's rows; the full arrays would be ~1.9 GB per worker at
    # 18.9M keys (~30 GB across 16) of data it discards immediately.
    phase_start = time.perf_counter()
    arrays, max_actions = load_checkpoint_rows(storage.checkpoint_dir, my_old_ids_array)
    print(
        f"[checkpoint-load] worker {wid}: read {len(my_keys):,} array rows "
        f"in {time.perf_counter() - phase_start:.1f}s",
        flush=True,
    )

    if max_actions != storage.max_actions:
        raise ValueError(f"Checkpoint max_actions mismatch: {max_actions} vs {storage.max_actions}")

    if storage.capacity < total_rows:
        raise ValueError(f"Storage capacity too small: {storage.capacity} vs {total_rows}")

    phase_start = time.perf_counter()
    new_ids = []
    for key in my_keys:
        new_id = storage.allocate_id()
        storage.state.owned_keys[key] = new_id
        storage.state.unshipped_keys.append((key, new_id))
        new_ids.append(new_id)

    new_ids_array = np.array(new_ids, dtype=np.int32)

    for spec in ARRAY_SPECS:
        array = getattr(storage, spec.attr)
        # Rows come back positionally in row_ids order, which is new_ids order.
        saved = arrays[spec.checkpoint_key]
        if spec.per_action:
            array[new_ids_array, :] = saved
        else:
            array[new_ids_array] = saved

    for new_id, legal_actions in zip(new_ids, owned.action_lists):
        storage.state.legal_actions_cache[new_id] = legal_actions
    print(
        f"[checkpoint-load] worker {wid}: populated {len(my_keys):,} infosets "
        f"in {time.perf_counter() - phase_start:.1f}s",
        flush=True,
    )

    print(
        f"Worker {wid} loaded {len(my_keys)}/{total_rows} infosets from checkpoint "
        f"in {time.perf_counter() - load_start:.1f}s total",
        flush=True,
    )
    return True
