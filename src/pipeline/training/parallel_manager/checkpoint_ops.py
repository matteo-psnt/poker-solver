"""Checkpoint key collection and persistence operations."""

from __future__ import annotations

import itertools
import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, TypedDict, cast

from src.core.game.actions import Action
from src.engine.solver.infoset import InfoSetKey
from src.pipeline.training.parallel_protocol import JobType
from src.shared import checkpoint_profile

from .gather import gather_worker_results

logger = logging.getLogger(__name__)

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
    Returns the accumulated view — the live storage-owned dicts, not copies.

    Cross-worker id collisions are structurally impossible (ids come only from
    worker-exclusive ranges, both at allocation and resume-load); the range
    overlap check below is the tripwire for a partitioning bug, and each worker
    checks its own shipped delta for duplicates.
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
    worker_ranges: dict[int, tuple[int, int]] = {}

    for result in responses:
        owned_keys = cast(dict[InfoSetKey, int], result["owned_keys"])
        legal_actions = cast("dict[int, Sequence[Action]]", result["legal_actions_cache"])
        worker_id = cast(int, result["worker_id"])
        worker_ranges[worker_id] = (
            cast(int, result["id_range_start"]),
            cast(int, result["id_range_end"]),
        )

        accumulated_keys.update(owned_keys)
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
    """Save checkpoint to disk after collecting key mappings from workers.

    The collect timeout must cover the WORST-case delta, not the typical one:
    the first collect of a run ships every key allocated so far (minutes of
    pickling per worker at production tree sizes — few workers and a late
    first checkpoint compound), and the default 60s killed exactly such runs.
    """
    with checkpoint_profile.phase("collect_keys"):
        collected = collect_keys(manager, timeout=1800.0)

    # collect_keys merges into the coordinator storage's own dicts and returns
    # them; these assignments are identity-preserving (no copies of 10M+ entries).
    manager.storage.state.owned_keys = collected["owned_keys"]
    manager.storage.state.legal_actions_cache = collected["legal_actions_cache"]

    if manager.storage.state.owned_keys:
        max_id = max(manager.storage.state.owned_keys.values())
        manager.storage.state.next_local_id = max_id + 1

    if manager.config.training.verbose:
        logger.info(
            f"[Master] Checkpointing iter={iteration}: "
            f"{len(manager.storage.state.owned_keys):,} infosets "
            f"(max_id={manager.storage.state.next_local_id})...",
        )

    # Per-street convergence snapshot: the coordinator only holds the full key→id
    # map here (post-collect), so per-street quality is recorded at checkpoint
    # cadence rather than per batch. Sampled + defensive — must never fail a write.
    _record_street_metrics(manager, iteration)

    with checkpoint_profile.phase("storage_write"):
        manager.storage.checkpoint(iteration)


def _record_street_metrics(manager: SharedArrayWorkerManager, iteration: int) -> None:
    """Append one per-street quality row to ``run_dir/street_metrics.jsonl``."""
    import json
    import random

    from src.pipeline.training.metrics import compute_per_street_quality

    sample_size = 20_000
    try:
        storage = manager.storage
        run_dir = storage.checkpoint_dir
        if run_dir is None:
            return
        owned = storage.state.owned_keys
        if not owned:
            return
        # Reservoir sample so streets are represented in proportion (dict order is
        # allocation order — early streets first — so a prefix would be biased).
        # O(N) iteration, O(sample_size) memory; cheap beside the O(N) checkpoint
        # write it rides alongside, and never materializes the whole key map.
        rng = random.Random(iteration)
        sample: list[tuple[object, int]] = []
        for i, item in enumerate(owned.items()):
            if i < sample_size:
                sample.append(item)
            else:
                j = rng.randint(0, i)
                if j < sample_size:
                    sample[j] = item
        per_street = compute_per_street_quality(
            sample,
            storage.shared_regrets,
            storage.shared_action_counts,
        )
        row = {"iteration": iteration, "num_infosets": len(owned), "per_street": per_street}
        with open(run_dir / "street_metrics.jsonl", "a") as fh:
            fh.write(json.dumps(row) + "\n")
    except Exception as exc:  # pragma: no cover - telemetry must never fail a checkpoint
        logger.warning(f"[street-metrics] skipped at iter={iteration}: {exc}")
