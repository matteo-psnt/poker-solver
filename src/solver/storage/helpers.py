from pathlib import Path

import numpy as np

from src.game.actions import Action, ActionType

CHECKPOINT_REQUIRED_FILES = (
    "regrets.npy",
    "strategies.npy",
    "action_counts.npy",
    "reach_counts.npy",
    "cumulative_utility.npy",
    "key_mapping.pkl",
    "action_signatures.pkl",
)


def get_missing_checkpoint_files(checkpoint_dir: Path) -> list[str]:
    return [name for name in CHECKPOINT_REQUIRED_FILES if not (checkpoint_dir / name).exists()]


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
