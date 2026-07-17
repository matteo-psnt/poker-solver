"""Checkpoint key collection and persistence operations."""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, TypedDict, cast

from src.core.game.actions import Action
from src.engine.solver.infoset import InfoSetKey
from src.pipeline.training.parallel_protocol import JobType

from .gather import gather_worker_results

if TYPE_CHECKING:
    from .manager import SharedArrayWorkerManager


class CollectedKeys(TypedDict):
    """Collected key and legal-action mappings from all workers."""

    owned_keys: dict[InfoSetKey, int]
    legal_actions_cache: dict[int, list[Action]]


def collect_keys(
    manager: SharedArrayWorkerManager,
    timeout: float = 60.0,
) -> CollectedKeys:
    """
    Collect owned keys from all workers for checkpointing.

    Workers send key→ID mappings to the coordinator so complete checkpoints can be written.
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

    all_owned_keys: dict[InfoSetKey, int] = {}
    all_legal_actions: dict[int, list[Action]] = {}
    worker_ranges: dict[int, tuple[int, int]] = {}
    id_owners: dict[int, tuple[int, InfoSetKey]] = {}

    for result in responses:
        owned_keys = cast(dict[InfoSetKey, int], result["owned_keys"])
        legal_actions = cast(dict[int, list[Action]], result["legal_actions_cache"])
        worker_id = cast(int, result["worker_id"])
        worker_ranges[worker_id] = (
            cast(int, result["id_range_start"]),
            cast(int, result["id_range_end"]),
        )

        for key, infoset_id in owned_keys.items():
            if infoset_id in id_owners:
                prev_worker, prev_key = id_owners[infoset_id]
                raise RuntimeError(
                    f"Duplicate infoset_id {infoset_id} across workers "
                    f"(prev worker {prev_worker}, key {prev_key} vs "
                    f"worker {worker_id}, key {key}). "
                    "ID ranges likely overlapping or num_workers changed."
                )
            id_owners[infoset_id] = (worker_id, key)
            all_owned_keys[key] = infoset_id

        all_legal_actions.update(legal_actions)

    ranges = sorted(worker_ranges.items(), key=lambda item: item[1][0])
    for (wid_a, (start_a, end_a)), (wid_b, (start_b, end_b)) in itertools.pairwise(ranges):
        if start_b < end_a:
            raise RuntimeError(
                f"Worker ID ranges overlap: worker {wid_a} [{start_a},{end_a}) "
                f"and worker {wid_b} [{start_b},{end_b}). "
                "This will corrupt checkpoints; ensure consistent num_workers/initial_capacity."
            )

    return {
        "owned_keys": all_owned_keys,
        "legal_actions_cache": all_legal_actions,
    }


def checkpoint(manager: SharedArrayWorkerManager, iteration: int) -> None:
    """Save checkpoint to disk after collecting key mappings from workers."""
    collected = collect_keys(manager)

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

    manager.storage.checkpoint(iteration)
