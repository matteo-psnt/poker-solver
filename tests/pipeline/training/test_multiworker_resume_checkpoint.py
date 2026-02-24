"""A multi-worker resume must not lose keys from the checkpoint it resumed from.

Workers ship only *newly allocated* keys at COLLECT_KEYS (``unshipped_keys``) and the
coordinator accumulates across collects, so a checkpoint is no longer self-contained in
one exchange -- it is the sum of every collect since the run started. That makes two
failure modes silent rather than loud:

- A worker that fails to seed ``unshipped_keys`` with the shard it loaded on resume
  never ships those keys, and they vanish from the next checkpoint even though training
  keeps updating their rows.
- Anything that clears the coordinator's accumulated view mid-run drops every key
  collected before it, with no error.

Both shrink a checkpoint instead of crashing, so only an end-to-end key-set comparison
catches them. The existing resume tests all run ``num_workers=1``, which never exercises
the ship/accumulate path at all.
"""

import itertools
import json
import shutil
import tempfile
from pathlib import Path

import pytest

from src.engine.solver.storage import key_table
from src.engine.solver.storage.helpers import CheckpointPaths
from src.pipeline.training import components
from src.pipeline.training.parallel_manager import checkpoint_ops
from src.pipeline.training.trainer import TrainingSession
from src.shared.config import Config
from tests.test_helpers import DummyCardAbstraction

NUM_WORKERS = 2


@pytest.fixture
def temp_run_dir():
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture(autouse=True)
def mock_card_abstraction(monkeypatch):
    monkeypatch.setattr(
        components, "build_card_abstraction", lambda *_a, **_k: DummyCardAbstraction()
    )
    monkeypatch.setattr(
        components, "resolve_card_abstraction_hash", lambda config: "dummy-abstraction-hash"
    )


def _build_config(temp_run_dir: Path, checkpoint_frequency: int = 100) -> Config:
    """Config is frozen, so variants are built rather than mutated."""
    return Config.from_dict(
        {
            "system": {"config_name": "test_multiworker_resume", "seed": 42},
            "game": {"starting_stack": 200, "small_blind": 1, "big_blind": 2},
            "action_model": {
                "preflop_templates": {"sb_first_in": ["fold", "call", 2.0]},
                "postflop_templates": {"first_aggressive": [1.0]},
                "jam_spr_threshold": 2.0,
            },
            "resolver": {"max_raises_per_street": 2},
            "card_abstraction": {"config": "default"},
            "training": {
                "runs_dir": str(temp_run_dir.parent),
                "num_iterations": 50,
                "checkpoint_frequency": checkpoint_frequency,
                "verbose": False,
            },
        }
    )


@pytest.fixture
def test_config(temp_run_dir):
    return _build_config(temp_run_dir)


def _checkpointed_keys(run_dir: Path) -> set:
    """The key set of the run's current checkpoint, read back off disk."""
    paths = CheckpointPaths.from_dir(run_dir)
    return set(key_table.read_all_rows(paths.key_table).keys)


@pytest.mark.timeout(120)
def test_resumed_multiworker_checkpoint_keeps_the_keys_it_resumed_from(test_config, temp_run_dir):
    session1 = TrainingSession(test_config, run_id=temp_run_dir.name)
    session1.train(num_iterations=4, num_workers=NUM_WORKERS)

    keys_before = _checkpointed_keys(temp_run_dir)
    assert keys_before, "guard: the first run must have checkpointed some keys"

    session2 = TrainingSession.resume(temp_run_dir)
    session2.train(num_iterations=4, num_workers=NUM_WORKERS)

    keys_after = _checkpointed_keys(temp_run_dir)

    dropped = keys_before - keys_after
    assert not dropped, (
        f"{len(dropped)} of {len(keys_before)} pre-resume keys are missing from the "
        f"post-resume checkpoint; workers did not re-ship their loaded shard"
    )


@pytest.mark.timeout(120)
def test_successive_multiworker_checkpoints_accumulate(temp_run_dir):
    """Within one run, a later checkpoint is a superset of an earlier one.

    Guards the accumulation itself: with incremental shipping, a coordinator that
    reset its view between collects would still write a plausible-looking (but
    truncated) checkpoint.
    """
    config = _build_config(temp_run_dir, checkpoint_frequency=2)
    session = TrainingSession(config, run_id=temp_run_dir.name)
    session.train(num_iterations=2, num_workers=NUM_WORKERS)
    keys_first = _checkpointed_keys(temp_run_dir)
    assert keys_first, "guard: the first checkpoint must contain keys"

    session.train(num_iterations=4, num_workers=NUM_WORKERS)
    keys_second = _checkpointed_keys(temp_run_dir)

    dropped = keys_first - keys_second
    assert not dropped, f"{len(dropped)} keys from the earlier checkpoint were dropped"


@pytest.mark.timeout(120)
def test_multi_collect_ships_deltas_and_accumulates(temp_run_dir, monkeypatch):
    """Exercise the actual incremental mechanic: several COLLECT_KEYS within ONE
    manager lifetime, so later collects ship only the delta and truncate
    ``unshipped_keys``.

    The other tests in this file span manager lifetimes (one collect each), so a
    coordinator that shipped the delta but failed to accumulate, or a worker that
    truncated too much, would pass them.
    """
    config = _build_config(temp_run_dir, checkpoint_frequency=1).merge(
        {"storage": {"max_checkpoint_overhead": 0.99}}
    )

    collect_sizes: list[int] = []
    real_collect = checkpoint_ops.collect_keys

    def recording_collect(manager, timeout=60.0):
        collected = real_collect(manager, timeout=timeout)
        collect_sizes.append(len(collected["owned_keys"]))
        return collected

    monkeypatch.setattr(checkpoint_ops, "collect_keys", recording_collect)

    session = TrainingSession(config, run_id=temp_run_dir.name)
    session.train(num_iterations=12, num_workers=NUM_WORKERS, batch_size=2)

    # checkpoint_frequency=1 with batch_size=2 checkpoints after every batch, the
    # coordinator waits out the pending checkpoint at each batch start, and
    # max_checkpoint_overhead=0.99 defeats back-pressure deferral: 6 collects.
    assert len(collect_sizes) == 6, f"expected 6 collects, got {len(collect_sizes)}"
    assert all(a <= b for a, b in itertools.pairwise(collect_sizes)), (
        f"accumulated view shrank between collects: {collect_sizes}"
    )
    assert collect_sizes[0] < collect_sizes[-1], "guard: later batches must allocate new keys"

    # The key table on disk must match the workers' id-derived infoset count
    # (reported per batch, independent of the ship/accumulate machinery) — catches
    # both a coordinator that failed to accumulate and a worker that over-truncated.
    metadata = json.loads((temp_run_dir / ".run.json").read_text())
    keys_on_disk = _checkpointed_keys(temp_run_dir)
    assert len(keys_on_disk) == collect_sizes[-1]
    assert len(keys_on_disk) == metadata["num_infosets"]
