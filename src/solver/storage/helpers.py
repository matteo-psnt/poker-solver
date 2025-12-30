import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from src.game.actions import Action, ActionType

# Checkpoint file names
CHECKPOINT_ZARR_FILE = "checkpoint.zarr.zip"
KEY_MAPPING_FILE = "key_mapping.pkl"
ACTION_SIGNATURES_FILE = "action_signatures.pkl"

CHECKPOINT_REQUIRED_FILES = (
    CHECKPOINT_ZARR_FILE,
    KEY_MAPPING_FILE,
    ACTION_SIGNATURES_FILE,
)


@dataclass(frozen=True)
class CheckpointPaths:
    base: Path
    checkpoint_zarr: Path
    key_mapping: Path
    action_signatures: Path

    @classmethod
    def from_dir(cls, checkpoint_dir: Path) -> "CheckpointPaths":
        return cls(
            base=checkpoint_dir,
            checkpoint_zarr=checkpoint_dir / CHECKPOINT_ZARR_FILE,
            key_mapping=checkpoint_dir / KEY_MAPPING_FILE,
            action_signatures=checkpoint_dir / ACTION_SIGNATURES_FILE,
        )


def get_missing_checkpoint_files(checkpoint_dir: Path) -> list[str]:
    """Check for missing checkpoint files."""
    return [f for f in CHECKPOINT_REQUIRED_FILES if not (checkpoint_dir / f).exists()]


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
    """Load checkpoint arrays from Zarr format."""
    zarr_path = checkpoint_dir / CHECKPOINT_ZARR_FILE
    if not zarr_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {zarr_path}")

    store = zarr.ZipStore(zarr_path, mode="r")
    try:
        root = zarr.open(store, mode="r")
        return {
            "regrets": root["regrets"][:],
            "strategies": root["strategies"][:],
            "action_counts": root["action_counts"][:],
            "reach_counts": root["reach_counts"][:],
            "cumulative_utility": root["cumulative_utility"][:],
        }
    finally:
        store.close()


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
    regrets = arrays["regrets"]
    strategies = arrays["strategies"]
    action_counts = arrays["action_counts"]
    reach_counts = arrays["reach_counts"]
    cumulative_utility = arrays["cumulative_utility"]

    if regrets.shape[0] != max_id:
        raise ValueError(
            f"{context}: max_id does not match regrets shape ({max_id} vs {regrets.shape[0]})"
        )
    if strategies.shape[0] != max_id:
        raise ValueError(
            f"{context}: max_id does not match strategies shape ({max_id} vs {strategies.shape[0]})"
        )
    if action_counts.shape[0] != max_id:
        raise ValueError(
            f"{context}: max_id does not match action_counts shape "
            f"({max_id} vs {action_counts.shape[0]})"
        )
    if reach_counts.shape[0] != max_id:
        raise ValueError(
            f"{context}: max_id does not match reach_counts shape "
            f"({max_id} vs {reach_counts.shape[0]})"
        )
    if cumulative_utility.shape[0] != max_id:
        raise ValueError(
            f"{context}: max_id does not match cumulative_utility shape "
            f"({max_id} vs {cumulative_utility.shape[0]})"
        )
    max_actions = regrets.shape[1]
    if strategies.shape[1] != max_actions:
        raise ValueError(
            f"{context}: max_actions does not match strategies shape "
            f"({max_actions} vs {strategies.shape[1]})"
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
