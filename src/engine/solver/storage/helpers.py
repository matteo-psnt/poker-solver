from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from src.core.game.actions import Action
from src.engine.solver.infoset import InfoSetKey
from src.engine.solver.storage import key_table
from src.engine.solver.storage.array_specs import ARRAY_SPECS

# Legacy fixed checkpoint file names (pre-manifest runs; still readable).

logger = logging.getLogger(__name__)
CHECKPOINT_ZARR_DIR = "checkpoint.zarr"
KEY_MAPPING_FILE = "key_mapping.pkl"
ACTION_SIGNATURES_FILE = "action_signatures.pkl"

# Current layout: keys and their action signatures live in one columnar directory
# (see key_table). The two pickles above are the pre-v3 layout, read only by the
# migration that converts them.
KEY_TABLE_DIR = "keys"

# Manifest pointing at the current snapshot. Snapshots are written under
# versioned names and become current only via an atomic manifest replace, so a
# crash mid-write can never corrupt the last good checkpoint.
CHECKPOINT_MANIFEST_FILE = "CHECKPOINT.json"


@dataclass(frozen=True)
class CheckpointPaths:
    base: Path
    checkpoint_zarr: Path
    key_table: Path

    @classmethod
    def from_dir(cls, checkpoint_dir: Path) -> CheckpointPaths:
        """Resolve the current snapshot from the manifest.

        Current-layout only: a run old enough to lack a manifest, or to carry the
        pre-v3 key/signature pickles, is stopped by the representation-version
        guard before any loader gets here, and is converted by its migration.
        """
        manifest = read_checkpoint_manifest(checkpoint_dir)
        if manifest is None:
            return cls(
                base=checkpoint_dir,
                checkpoint_zarr=checkpoint_dir / CHECKPOINT_ZARR_DIR,
                key_table=checkpoint_dir / KEY_TABLE_DIR,
            )
        return cls(
            base=checkpoint_dir,
            checkpoint_zarr=checkpoint_dir / manifest["zarr"],
            key_table=checkpoint_dir / manifest["key_table"],
        )

    @classmethod
    def for_iteration(cls, checkpoint_dir: Path, iteration: int) -> CheckpointPaths:
        """Versioned artifact paths for writing a new snapshot."""
        return cls(
            base=checkpoint_dir,
            checkpoint_zarr=checkpoint_dir / f"checkpoint-{iteration}.zarr",
            key_table=checkpoint_dir / f"keys-{iteration}",
        )


def read_checkpoint_manifest(checkpoint_dir: Path) -> dict | None:
    """Read the checkpoint manifest, or None if this run predates manifests."""
    path = checkpoint_dir / CHECKPOINT_MANIFEST_FILE
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    for field in ("iteration", "zarr", "key_table"):
        if field not in manifest:
            raise ValueError(f"Invalid checkpoint manifest {path}: missing {field!r}")
    return manifest


def resolve_resume_iteration(checkpoint_dir: Path, metadata_iterations: int) -> int | None:
    """Resolve the iteration a resume should continue from.

    The manifest is authoritative: ``commit_checkpoint_manifest`` writes its
    ``iteration`` in the same atomic ``os.replace`` that publishes the arrays it
    describes, whereas run metadata (``.run.json``) is written afterwards in a
    separate, non-atomic step. A hard kill between the two -- SIGKILL, OOM, the
    Modal guillotine -- leaves metadata behind the data on disk. Trusting
    metadata there would re-run ``[metadata, manifest)`` and, because the deal
    stream is a pure function of the absolute iteration index, replay deals
    already baked into the loaded arrays and double-count them.

    Falls back to metadata only for pre-manifest runs, which have no better
    source. Returns None when neither knows of a checkpoint.
    """
    manifest = read_checkpoint_manifest(checkpoint_dir)
    if manifest is None:
        return metadata_iterations if metadata_iterations > 0 else None

    manifest_iteration = int(manifest["iteration"])
    if manifest_iteration != metadata_iterations:
        logger.info(
            f"[resume] Run metadata reports iteration {metadata_iterations} but the "
            f"checkpoint manifest reports {manifest_iteration}; trusting the manifest "
            "(metadata was likely not flushed before the previous leg died).",
        )
    return manifest_iteration if manifest_iteration > 0 else None


def commit_checkpoint_manifest(
    checkpoint_dir: Path, iteration: int, paths: CheckpointPaths
) -> None:
    """Atomically make a fully-written snapshot current, then prune superseded artifacts."""
    manifest = {
        "iteration": iteration,
        "zarr": paths.checkpoint_zarr.name,
        "key_table": paths.key_table.name,
    }
    tmp = checkpoint_dir / (CHECKPOINT_MANIFEST_FILE + ".tmp")
    tmp.write_text(json.dumps(manifest))
    os.replace(tmp, checkpoint_dir / CHECKPOINT_MANIFEST_FILE)
    _prune_superseded_snapshots(checkpoint_dir, keep=manifest)


def _prune_superseded_snapshots(checkpoint_dir: Path, keep: dict) -> None:
    """Delete snapshots the manifest no longer references. Never fails a checkpoint."""
    keep_names = {keep["zarr"], keep["key_table"]}
    doomed: list[Path] = [
        checkpoint_dir / CHECKPOINT_ZARR_DIR,
        checkpoint_dir / KEY_MAPPING_FILE,
        checkpoint_dir / ACTION_SIGNATURES_FILE,
    ]
    for pattern in ("checkpoint-*.zarr", "keys-*"):
        doomed.extend(p for p in checkpoint_dir.glob(pattern) if p.name not in keep_names)
    for path in doomed:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        except OSError as exc:
            logger.warning(f"Warning: could not prune superseded checkpoint {path.name}: {exc}")


def get_missing_checkpoint_files(checkpoint_dir: Path) -> list[str]:
    """Check for missing checkpoint files."""
    paths = CheckpointPaths.from_dir(checkpoint_dir)
    missing = [paths.checkpoint_zarr.name] if not paths.checkpoint_zarr.exists() else []
    if not paths.key_table.exists():
        return [*missing, paths.key_table.name]
    return missing + [f.name for f in key_table.table_files(paths.key_table) if not f.exists()]


def load_checkpoint_arrays(checkpoint_dir: Path) -> dict[str, np.ndarray]:
    """Load checkpoint arrays from Zarr format (directory store for performance)."""
    zarr_path = CheckpointPaths.from_dir(checkpoint_dir).checkpoint_zarr
    if not zarr_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {zarr_path}")

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.open(store, mode="r")

    # Load arrays ([:] triggers decompression)
    return {spec.checkpoint_key: root[spec.checkpoint_key][:] for spec in ARRAY_SPECS}


def load_checkpoint_rows(checkpoint_dir: Path, row_ids: np.ndarray) -> tuple[dict, int]:
    """Load ONLY ``row_ids`` from each checkpoint array, plus max_actions.

    Workers own ~1/N of the rows, so reading the full arrays into every worker
    costs N x the whole checkpoint in private memory (~1.9 GB each at 18.9M keys,
    ~30 GB across 16) for data they immediately discard. Zarr's orthogonal
    selection materializes only the selected rows. ``row_ids`` must be sorted
    ascending, which is what ``flatnonzero`` on the ownership mask yields.

    Returned rows are positional: result row ``k`` is ``row_ids[k]``.
    """
    zarr_path = CheckpointPaths.from_dir(checkpoint_dir).checkpoint_zarr
    if not zarr_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {zarr_path}")

    root = zarr.open(zarr.DirectoryStore(zarr_path), mode="r")
    selected: dict[str, np.ndarray] = {}
    max_actions = 0
    for spec in ARRAY_SPECS:
        array = root[spec.checkpoint_key]
        if spec.per_action:
            max_actions = array.shape[1]
            selected[spec.checkpoint_key] = array.oindex[row_ids, :]
        else:
            selected[spec.checkpoint_key] = array.oindex[row_ids]
    return selected, max_actions


def _validate_key_table(table_dir: Path, context: str) -> tuple[int, int]:
    """Row count and exclusive max id. Ids are row indices, so they cannot be invalid."""
    rows = key_table.num_rows(table_dir)
    if rows < 0:
        raise ValueError(f"{context}: key table reports a negative row count")
    return rows, rows


def _validate_checkpoint_arrays(arrays: dict[str, np.ndarray], max_id: int, context: str) -> int:
    for spec in ARRAY_SPECS:
        array = arrays[spec.checkpoint_key]
        if array.shape[0] != max_id:
            raise ValueError(
                f"{context}: max_id does not match {spec.checkpoint_key} shape "
                f"({max_id} vs {array.shape[0]})"
            )
    max_actions = arrays["regrets"].shape[1]
    if arrays["strategies"].shape[1] != max_actions:
        raise ValueError(
            f"{context}: max_actions does not match strategies shape "
            f"({max_actions} vs {arrays['strategies'].shape[1]})"
        )
    return max_actions


def validate_action_counts(
    action_counts: np.ndarray, action_lists: Sequence[Sequence[Action]], context: str
) -> None:
    """Each row's action list must match the action count stored for that row.

    Row index is the infoset id, so ids can no longer be duplicated or out of range
    -- the only failure left is a length disagreement.
    """
    if len(action_lists) > len(action_counts):
        raise ValueError(
            f"{context}: {len(action_lists)} action lists for {len(action_counts)} counted rows"
        )
    mismatches: list[tuple[int, int, int]] = []
    for infoset_id, actions in enumerate(action_lists):
        expected = int(action_counts[infoset_id])
        if len(actions) != expected:
            mismatches.append((infoset_id, len(actions), expected))

    if mismatches:
        preview = "; ".join(
            f"id {mid}: sig_len {slen} vs count {cnt}" for mid, slen, cnt in mismatches[:5]
        )
        raise ValueError(
            f"{context}: {len(mismatches)} action signature/count mismatches. Examples: {preview}"
        )


@dataclass(frozen=True)
class CheckpointData:
    """Every row of a checkpoint, for single-process readers (evaluation, charts).

    Workers must NOT use this: it materializes the whole table, which is what made
    a 16-worker resume need ~444 GB. They read their shard via
    ``key_table.read_owned_rows``.
    """

    max_id: int
    max_actions: int
    keys: list[InfoSetKey]
    action_lists: list[list[Action]]
    arrays: dict[str, np.ndarray]

    @property
    def owned_keys(self) -> dict[InfoSetKey, int]:
        return {key: i for i, key in enumerate(self.keys)}


def load_checkpoint_data(checkpoint_dir: Path, *, context: str) -> CheckpointData:
    paths = CheckpointPaths.from_dir(checkpoint_dir)
    _, max_id = _validate_key_table(paths.key_table, context)
    arrays = load_checkpoint_arrays(checkpoint_dir)
    max_actions = _validate_checkpoint_arrays(arrays, max_id, context)
    rows = key_table.read_all_rows(paths.key_table)
    validate_action_counts(arrays["action_counts"], rows.action_lists, context)
    return CheckpointData(
        max_id=max_id,
        max_actions=max_actions,
        keys=rows.keys,
        action_lists=rows.action_lists,
        arrays=arrays,
    )
