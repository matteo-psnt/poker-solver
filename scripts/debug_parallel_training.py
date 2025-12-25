"""
Debug script for parallel training behavior.

Runs sequential and parallel training with a dummy card abstraction and
basic sanity checks (iteration counts, shapes, NaNs).
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.bucketing.base import BucketingStrategy
from src.solver.storage import InMemoryStorage
from src.training import components
from src.training.trainer import TrainingSession
from src.utils.config import Config


class DummyCardAbstraction(BucketingStrategy):
    """Minimal card abstraction for fast, deterministic testing."""

    def get_bucket(self, hole_cards, board, street):
        return 0

    def num_buckets(self, street):
        return 1


def _patch_card_abstraction() -> None:
    """Force TrainingSession to use DummyCardAbstraction."""

    def _builder(config, prompt_user=False, auto_compute=False):
        return DummyCardAbstraction()

    components.build_card_abstraction = _builder


def _run_session(tmp_dir: Path, use_parallel: bool) -> TrainingSession:
    config = Config.default()
    config.set("training.runs_dir", str(tmp_dir / "runs"))
    config.set("training.num_iterations", 12)
    config.set("training.verbose", True)
    config.set("storage.backend", "memory")

    session = TrainingSession(config)
    results = session.train(
        num_iterations=12,
        use_parallel=use_parallel,
        num_workers=2,
        batch_size=4,
    )

    print("\nResults:", results)

    # Basic sanity checks
    assert results["total_iterations"] == 12, "Iteration count mismatch"
    assert results["final_infosets"] > 0, "No infosets discovered"

    storage = session.solver.storage
    assert isinstance(storage, InMemoryStorage)
    for infoset in storage.infosets.values():
        assert infoset.regrets.shape == (infoset.num_actions,)
        assert infoset.strategy_sum.shape == (infoset.num_actions,)
        assert not np.any(np.isnan(infoset.regrets)), "NaNs in regrets"
        assert not np.any(np.isnan(infoset.strategy_sum)), "NaNs in strategy_sum"

    return session


def main() -> None:
    _patch_card_abstraction()

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        print("=== Sequential run ===")
        _run_session(tmp_dir, use_parallel=False)

        print("\n=== Parallel run ===")
        _run_session(tmp_dir, use_parallel=True)


if __name__ == "__main__":
    main()
