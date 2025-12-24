"""
Tests for TrainingSession resume functionality.

Verifies that resuming from checkpoints correctly restores solver state,
storage mappings, and iteration counters.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from src.training.trainer import TrainingSession
from src.utils.config import Config


@pytest.fixture
def temp_run_dir():
    """Create a temporary run directory."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    if temp_dir.exists():
        shutil.rmtree(temp_dir)


@pytest.fixture
def test_config(temp_run_dir):
    """Create a minimal test configuration."""
    config_dict = {
        "system": {
            "config_name": "test_resume",
            "seed": 42,
        },
        "game": {
            "starting_stack": 200,
            "small_blind": 1,
            "big_blind": 2,
        },
        "action_abstraction": {
            "raise_sizes": {
                "preflop": [1.0],
                "flop": [1.0],
                "turn": [1.0],
                "river": [1.0],
            },
        },
        "card_abstraction": {
            "config": "fast_test",  # Uses a small precomputed abstraction
        },
        "storage": {
            "backend": "disk",  # Must use disk storage for resume
            "cache_size": 1000,
            "flush_frequency": 10000,  # Flush very rarely to avoid performance issues in tests
        },
        "solver": {
            "type": "mccfr",
        },
        "training": {
            "runs_dir": str(temp_run_dir.parent),
            "num_iterations": 50,
            "checkpoint_frequency": 100,  # Checkpoint at end only
            "log_frequency": 10,
            "verbose": False,
        },
    }
    return Config.from_dict(config_dict)


def test_resume_basic(test_config, temp_run_dir):
    """
    Test basic resume functionality.

    1. Run training for a few iterations and checkpoint
    2. Create new session via resume
    3. Verify state was restored correctly

    Note: This is a minimal smoke test. Full integration testing with
    long-running training should be done manually due to performance constraints.
    """
    # Phase 1: Initial training with very few iterations
    print("\n=== Phase 1: Initial Training (2 iterations) ===")
    session1 = TrainingSession(test_config, run_id=temp_run_dir.name)

    # Train for just 1 iteration to minimize infoset creation
    initial_iterations = 1
    results1 = session1.train(num_iterations=initial_iterations, use_parallel=False)

    initial_infosets = results1["final_infosets"]
    print(
        f"Initial training complete: {initial_infosets} infosets, {initial_iterations} iterations"
    )

    # Verify some basic outcomes
    assert initial_infosets > 0, "No infosets created during initial training"
    assert results1["total_iterations"] == initial_iterations

    # Phase 2: Resume and verify state
    print("\n=== Phase 2: Resume and Verify ===")
    session2 = TrainingSession.resume(temp_run_dir)

    # Verify state was restored
    assert session2.solver.iteration == initial_iterations, (
        f"Solver iteration not restored: expected {initial_iterations}, got {session2.solver.iteration}"
    )
    restored_infosets = session2.storage.num_infosets()
    assert restored_infosets == initial_infosets, (
        f"Infosets not restored: expected {initial_infosets}, got {restored_infosets}"
    )

    print("✅ Resume verification passed!")
    print(f"   Solver iteration: {session2.solver.iteration} (expected {initial_iterations})")
    print(f"   Infosets: {restored_infosets} (expected {initial_infosets})")


def test_resume_nonexistent_dir():
    """Test that resume fails gracefully for non-existent directory."""
    with pytest.raises(FileNotFoundError, match="Run directory not found"):
        TrainingSession.resume("/nonexistent/path")


def test_resume_no_checkpoint(temp_run_dir):
    """Test that resume fails if no checkpoint exists."""
    # Create run directory but no checkpoint files
    temp_run_dir.mkdir(parents=True, exist_ok=True)
    (temp_run_dir / ".run.json").write_text('{"run_id": "test", "iterations": 0}')

    with pytest.raises(FileNotFoundError, match="No checkpoint found"):
        TrainingSession.resume(temp_run_dir)


def test_resume_incomplete_checkpoint(test_config, temp_run_dir):
    """Test that resume fails if checkpoint is incomplete."""
    # Create a session and run minimal training
    session = TrainingSession(test_config, run_id=temp_run_dir.name)
    session.train(num_iterations=1, use_parallel=False)

    # Delete one of the checkpoint files to make it incomplete
    (temp_run_dir / "regrets.h5").unlink()

    # Should fail with clear error message
    with pytest.raises(ValueError, match="Checkpoint is incomplete"):
        TrainingSession.resume(temp_run_dir)


def test_resume_metadata_tracking(test_config, temp_run_dir):
    """Test that resume correctly updates run metadata."""
    # Initial training
    session1 = TrainingSession(test_config, run_id=temp_run_dir.name)
    session1.train(num_iterations=1, use_parallel=False)

    # Resume
    session2 = TrainingSession.resume(temp_run_dir)

    # Check metadata was updated
    metadata = session2.run_tracker.metadata
    assert metadata["resumed_at"] is not None, "resumed_at should be set"
    assert metadata["started_at"] is not None, "started_at should still be set"
    assert metadata["status"] == "running", "Status should be 'running' after resume"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
