"""Per-infoset learning must be independent of num_workers.

Regression test for the worker-sharding bug where every worker computed
regret/strategy updates for infosets it did not own and then discarded them,
so each infoset was effectively trained on num_iterations / num_workers hands
(docs/WORKER_SHARDING_DROPPED_UPDATES.md).

The invariant: every hand reaches the first preflop decision, and the first
actor traverses in exactly half the iterations (traversing_player alternates
with the iteration counter). So across the root preflop infosets, the number
of surviving average-strategy updates in a batch must be ~batch_iterations / 2
regardless of worker count. Under the bug, a worker only kept updates for
infosets it owned by hash, giving ~batch_iterations / (2 * num_workers).

The measurement runs a few warmup batches first, then measures one batch
restricted to root infosets that already existed: cross-worker infoset IDs
are only learned via ID exchange, so early-batch updates to remote infosets
are legitimately dropped while IDs are unknown. That discovery lag decays per
batch (measured: 0.62 -> 0.81 -> 0.92 of the maximum at N=2) and is why the
warmup exists; the bug this guards against sits at ~1/num_workers regardless
of warmup.
"""

import pickle

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.state import Street
from src.pipeline.training.parallel_manager import SharedArrayWorkerManager
from src.shared.config import Config
from tests.test_helpers import DummyCardAbstraction

BATCH_ITERATIONS = 800
WARMUP_BATCHES = 3


def _measured_batch_root_update_count(num_workers: int) -> float:
    """Warm up, then return the root-infoset reach delta over one more batch."""
    assert BATCH_ITERATIONS % (2 * num_workers) == 0, "need even per-worker parity split"

    config = Config.default().merge({"training": {"verbose": False}})
    per_worker = [BATCH_ITERATIONS // num_workers] * num_workers

    with SharedArrayWorkerManager(
        num_workers=num_workers,
        config=config,
        serialized_action_model=pickle.dumps(ActionModel(config)),
        serialized_card_abstraction=pickle.dumps(DummyCardAbstraction()),
        base_seed=42,
        initial_capacity=200_000,
        max_actions=10,
    ) as manager:
        for batch_id in range(WARMUP_BATCHES):
            manager.run_batch(
                iterations_per_worker=per_worker,
                batch_id=batch_id,
                start_iteration=batch_id * BATCH_ITERATIONS,
                verbose=False,
            )
            manager.exchange_ids(verbose=False)

        collected = manager.collect_keys()
        root_ids = np.array(
            [
                infoset_id
                for key, infoset_id in collected["owned_keys"].items()
                if key.street == Street.PREFLOP and key.betting_sequence == ""
            ],
            dtype=np.int64,
        )
        assert root_ids.size > 0, "warmup should have created root preflop infosets"

        storage = manager.get_storage()
        before = int(storage.shared_reach_counts[root_ids].sum())

        manager.run_batch(
            iterations_per_worker=per_worker,
            batch_id=WARMUP_BATCHES,
            start_iteration=WARMUP_BATCHES * BATCH_ITERATIONS,
            verbose=False,
        )
        after = int(storage.shared_reach_counts[root_ids].sum())

    return float(after - before)


@pytest.mark.slow
@pytest.mark.timeout(180)
@pytest.mark.parametrize("num_workers", [1, 2])
def test_root_infoset_updates_invariant_to_num_workers(num_workers: int) -> None:
    """Surviving root-infoset updates per batch must not scale as 1/num_workers.

    Expected updates = BATCH_ITERATIONS / 2 (root actor traverses half the
    iterations). Tolerance covers hands unseen in batch 0 (their root infosets
    are excluded from the snapshot), per-worker unknown remote IDs, and rare
    lost lock-free writes. The broken behavior sits at ~expected / num_workers,
    far below the threshold.
    """
    expected = BATCH_ITERATIONS / 2
    delta = _measured_batch_root_update_count(num_workers)

    assert delta <= expected, (
        f"root updates {delta} exceed the per-batch maximum {expected}; "
        "the root-invariant measurement itself is broken"
    )
    assert delta >= 0.75 * expected, (
        f"root infosets received {delta} updates in a {BATCH_ITERATIONS}-iteration batch "
        f"with num_workers={num_workers}, expected ~{expected}; per-infoset learning "
        "is scaling with 1/num_workers (dropped cross-partition updates)"
    )
