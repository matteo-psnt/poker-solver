from __future__ import annotations

import json
import os
import pickle
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from src.core.game.actions import Action, ActionType
from src.engine.solver.storage.array_specs import ARRAY_SPECS

# Legacy fixed checkpoint file names (pre-manifest runs; still readable).
CHECKPOINT_ZARR_DIR = "checkpoint.zarr"
KEY_MAPPING_FILE = "key_mapping.pkl"
ACTION_SIGNATURES_FILE = "action_signatures.pkl"

# Manifest pointing at the current snapshot. Snapshots are written under
# versioned names and become current only via an atomic manifest replace, so a
# crash mid-write can never corrupt the last good checkpoint.
CHECKPOINT_MANIFEST_FILE = "CHECKPOINT.json"


@dataclass(frozen=True)
class CheckpointPaths:
    base: Path
    checkpoint_zarr: Path
    key_mapping: Path
    action_signatures: Path

    @classmethod
    def from_dir(cls, checkpoint_dir: Path) -> CheckpointPaths:
        """Resolve the current snapshot: manifest if present, legacy names otherwise."""
        manifest = read_checkpoint_manifest(checkpoint_dir)
        if manifest is not None:
            return cls(
                base=checkpoint_dir,
                checkpoint_zarr=checkpoint_dir / manifest["zarr"],
                key_mapping=checkpoint_dir / manifest["key_mapping"],
                action_signatures=checkpoint_dir / manifest["action_signatures"],
            )
        return cls(
            base=checkpoint_dir,
            checkpoint_zarr=checkpoint_dir / CHECKPOINT_ZARR_DIR,
            key_mapping=checkpoint_dir / KEY_MAPPING_FILE,
            action_signatures=checkpoint_dir / ACTION_SIGNATURES_FILE,
        )

    @classmethod
    def for_iteration(cls, checkpoint_dir: Path, iteration: int) -> CheckpointPaths:
        """Versioned artifact paths for writing a new snapshot."""
        return cls(
            base=checkpoint_dir,
            checkpoint_zarr=checkpoint_dir / f"checkpoint-{iteration}.zarr",
            key_mapping=checkpoint_dir / f"key_mapping-{iteration}.pkl",
            action_signatures=checkpoint_dir / f"action_signatures-{iteration}.pkl",
        )


def read_checkpoint_manifest(checkpoint_dir: Path) -> dict | None:
    """Read the checkpoint manifest, or None if this run predates manifests."""
    path = checkpoint_dir / CHECKPOINT_MANIFEST_FILE
    if not path.exists():
        return None
    manifest = json.loads(path.read_text())
    for field in ("iteration", "zarr", "key_mapping", "action_signatures"):
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
        print(
            f"[resume] Run metadata reports iteration {metadata_iterations} but the "
            f"checkpoint manifest reports {manifest_iteration}; trusting the manifest "
            "(metadata was likely not flushed before the previous leg died).",
            flush=True,
        )
    return manifest_iteration if manifest_iteration > 0 else None


def commit_checkpoint_manifest(
    checkpoint_dir: Path, iteration: int, paths: CheckpointPaths
) -> None:
    """Atomically make a fully-written snapshot current, then prune superseded artifacts."""
    manifest = {
        "iteration": iteration,
        "zarr": paths.checkpoint_zarr.name,
        "key_mapping": paths.key_mapping.name,
        "action_signatures": paths.action_signatures.name,
    }
    tmp = checkpoint_dir / (CHECKPOINT_MANIFEST_FILE + ".tmp")
    tmp.write_text(json.dumps(manifest))
    os.replace(tmp, checkpoint_dir / CHECKPOINT_MANIFEST_FILE)
    _prune_superseded_snapshots(checkpoint_dir, keep=manifest)


def _prune_superseded_snapshots(checkpoint_dir: Path, keep: dict) -> None:
    """Delete snapshots the manifest no longer references. Never fails a checkpoint."""
    keep_names = {keep["zarr"], keep["key_mapping"], keep["action_signatures"]}
    doomed: list[Path] = [
        checkpoint_dir / CHECKPOINT_ZARR_DIR,
        checkpoint_dir / KEY_MAPPING_FILE,
        checkpoint_dir / ACTION_SIGNATURES_FILE,
    ]
    for pattern in ("checkpoint-*.zarr", "key_mapping-*.pkl", "action_signatures-*.pkl"):
        doomed.extend(p for p in checkpoint_dir.glob(pattern) if p.name not in keep_names)
    for path in doomed:
        try:
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        except OSError as exc:
            print(f"Warning: could not prune superseded checkpoint {path.name}: {exc}")


def get_missing_checkpoint_files(checkpoint_dir: Path) -> list[str]:
    """Check for missing checkpoint files."""
    paths = CheckpointPaths.from_dir(checkpoint_dir)
    return [
        p.name
        for p in (paths.checkpoint_zarr, paths.key_mapping, paths.action_signatures)
        if not p.exists()
    ]


def load_key_mapping(paths: CheckpointPaths) -> dict:
    with open(paths.key_mapping, "rb") as f:
        mapping_data = pickle.load(f)
    if not isinstance(mapping_data, dict):
        raise ValueError("Invalid checkpoint format: key_mapping is not a dict")
    return mapping_data


def load_action_signatures(paths: CheckpointPaths) -> dict[int, list[tuple[str, int]]]:
    with open(paths.action_signatures, "rb") as f:
        action_sigs = pickle.load(f)
    if not isinstance(action_sigs, dict):
        raise ValueError("Invalid checkpoint format: action_signatures is not a dict")
    return action_sigs


def load_checkpoint_arrays(checkpoint_dir: Path) -> dict[str, np.ndarray]:
    """Load checkpoint arrays from Zarr format (directory store for performance)."""
    zarr_path = CheckpointPaths.from_dir(checkpoint_dir).checkpoint_zarr
    if not zarr_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {zarr_path}")

    store = zarr.DirectoryStore(zarr_path)
    root = zarr.open(store, mode="r")

    # Load arrays ([:] triggers decompression)
    return {spec.checkpoint_key: root[spec.checkpoint_key][:] for spec in ARRAY_SPECS}


def _validate_checkpoint_mapping(mapping_data: dict, context: str) -> tuple[int, int]:
    if "owned_keys" not in mapping_data:
        raise ValueError(f"{context}: key_mapping missing owned_keys")

    owned_keys = mapping_data["owned_keys"]
    if not isinstance(owned_keys, dict):
        raise ValueError(f"{context}: key_mapping owned_keys must be a dict")

    num_infosets = len(owned_keys)
    max_id = 0
    for mapped_id in owned_keys.values():
        if not isinstance(mapped_id, int) or mapped_id < 0:
            raise ValueError(f"{context}: key_mapping contains invalid infoset IDs")
        if mapped_id >= max_id:
            max_id = mapped_id + 1

    return num_infosets, max_id


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


def _reconstruct_action(action_type_name: str, amount: int) -> Action:
    try:
        action_type = ActionType[action_type_name]
    except KeyError as exc:
        raise ValueError(f"Invalid action type name: {action_type_name}") from exc

    return Action(type=action_type, amount=amount)


def _validate_action_signatures(
    action_counts: np.ndarray, action_sigs: dict[int, list[tuple[str, int]]], context: str
) -> None:
    ids = list(action_sigs.keys())
    unique = len(set(ids))
    if unique != len(ids):
        raise ValueError(
            f"{context}: duplicate infoset IDs in action_signatures "
            f"({len(ids) - unique} duplicates)"
        )

    mismatches: list[tuple[int, int, int | None, str]] = []
    for infoset_id, sigs in action_sigs.items():
        if infoset_id >= len(action_counts):
            mismatches.append((infoset_id, len(sigs), None, "id_out_of_range"))
            continue
        n_actions = int(action_counts[infoset_id])
        if len(sigs) != n_actions:
            mismatches.append((infoset_id, len(sigs), n_actions, "len_mismatch"))

    if mismatches:
        preview = "; ".join(
            f"id {mid}: sig_len {slen} vs count {cnt}" for mid, slen, cnt, _ in mismatches[:5]
        )
        raise ValueError(
            f"{context}: {len(mismatches)} action signature/count mismatches. Examples: {preview}"
        )


def build_legal_actions(
    action_sigs: dict[int, list[tuple[str, int]]], infoset_id: int, context: str
) -> list[Action]:
    try:
        sigs = action_sigs[infoset_id]
    except KeyError as exc:
        raise ValueError(
            f"{context}: missing action signatures for infoset ID {infoset_id}"
        ) from exc
    return [_reconstruct_action(action_type_name, amount) for action_type_name, amount in sigs]


@dataclass(frozen=True)
class CheckpointData:
    max_id: int
    max_actions: int
    mapping_data: dict
    action_signatures: dict[int, list[tuple[str, int]]]
    arrays: dict[str, np.ndarray]

    @property
    def owned_keys(self) -> dict:
        return self.mapping_data["owned_keys"]


def load_checkpoint_data(checkpoint_dir: Path, *, context: str) -> CheckpointData:
    paths = CheckpointPaths.from_dir(checkpoint_dir)
    mapping_data = load_key_mapping(paths)
    _, max_id = _validate_checkpoint_mapping(mapping_data, context)
    arrays = load_checkpoint_arrays(checkpoint_dir)
    max_actions = _validate_checkpoint_arrays(arrays, max_id, context)
    action_sigs = load_action_signatures(paths)
    _validate_action_signatures(arrays["action_counts"], action_sigs, context)
    return CheckpointData(
        max_id=max_id,
        max_actions=max_actions,
        mapping_data=mapping_data,
        action_signatures=action_sigs,
        arrays=arrays,
    )
