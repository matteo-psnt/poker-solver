"""Checkpoint key collection and persistence operations."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict, cast

from src.core.game.actions import Action
from src.engine.solver.infoset import InfoSetKey
from src.pipeline.training.parallel_protocol import JobType
from src.shared import checkpoint_profile

from .gather import gather_worker_results

if TYPE_CHECKING:
    from .manager import SharedArrayWorkerManager


class CollectedKeys(TypedDict):
    """Collected key and legal-action mappings from all workers."""

    owned_keys: dict[InfoSetKey, int]
    legal_actions_cache: dict[int, Sequence[Action]]


def collect_keys(
    manager: SharedArrayWorkerManager,
    timeout: float = 60.0,
) -> CollectedKeys:
    """
    Collect owned keys from all workers for checkpointing.

    Workers ship only entries added since the previous collect; this merges them
    into the coordinator storage's accumulated view (``state.owned_keys`` /
    ``state.legal_actions_cache``), which is what checkpoints are written from.
    Returns the accumulated view.
    """
    for worker_id in range(manager.num_workers):
        manager.job_queue.put(
            {
                "type": JobType.COLLECT_KEYS.value,
                "target_worker": worker_id,
            }
        )

    responses, _ = gather_worker_results(
        manager,
        accept=lambda r: r.get("type") == "keys_collected",
        expected=manager.num_workers,
        timeout=timeout,
        description="key collection",
    )

    accumulated_keys = cast("dict[InfoSetKey, int]", manager.storage.state.owned_keys)
    accumulated_actions = cast(
        "dict[int, Sequence[Action]]", manager.storage.state.legal_actions_cache
    )
    id_owners = manager.checkpoint_id_owners
    worker_ranges: dict[int, tuple[int, int]] = {}

    for result in responses:
        owned_keys = cast(dict[InfoSetKey, int], result["owned_keys"])
        legal_actions = cast("dict[int, Sequence[Action]]", result["legal_actions_cache"])
        worker_id = cast(int, result["worker_id"])
        worker_ranges[worker_id] = (
            cast(int, result["id_range_start"]),
            cast(int, result["id_range_end"]),
        )

        for key, infoset_id in owned_keys.items():
            prev = id_owners.get(infoset_id)
            if prev is not None and prev[1] != key:
                prev_worker, prev_key = prev
                raise RuntimeError(
                    f"Duplicate infoset_id {infoset_id} across workers "
                    f"(prev worker {prev_worker}, key {prev_key} vs "
                    f"worker {worker_id}, key {key}). "
                    "ID ranges likely overlapping or num_workers changed."
                )
            id_owners[infoset_id] = (worker_id, key)
            accumulated_keys[key] = infoset_id

        accumulated_actions.update(legal_actions)

    ranges = sorted(worker_ranges.items(), key=lambda item: item[1][0])
    for (wid_a, (start_a, end_a)), (wid_b, (start_b, end_b)) in itertools.pairwise(ranges):
        if start_b < end_a:
            raise RuntimeError(
                f"Worker ID ranges overlap: worker {wid_a} [{start_a},{end_a}) "
                f"and worker {wid_b} [{start_b},{end_b}). "
                "This will corrupt checkpoints; ensure consistent num_workers/initial_capacity."
            )

    return {
        "owned_keys": accumulated_keys,
        "legal_actions_cache": accumulated_actions,
    }


def checkpoint(manager: SharedArrayWorkerManager, iteration: int) -> None:
    """Save checkpoint to disk after collecting key mappings from workers."""
    with checkpoint_profile.phase("collect_keys"):
        collected = collect_keys(manager)

    # collect_keys merges into the coordinator storage's own dicts and returns
    # them; these assignments are identity-preserving (no copies of 10M+ entries).
    manager.storage.state.owned_keys = collected["owned_keys"]
    manager.storage.state.legal_actions_cache = collected["legal_actions_cache"]

    if manager.storage.state.owned_keys:
        max_id = max(manager.storage.state.owned_keys.values())
        manager.storage.state.next_local_id = max_id + 1

    if manager.config.training.verbose:
        print(
            f"[Master] Checkpointing iter={iteration}: "
            f"{len(manager.storage.state.owned_keys):,} infosets "
            f"(max_id={manager.storage.state.next_local_id})...",
            flush=True,
        )

    with checkpoint_profile.phase("storage_write"):
        manager.storage.checkpoint(iteration)
