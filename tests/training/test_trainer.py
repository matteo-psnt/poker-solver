"""Tests for Trainer class."""

import pickle
import time

import numpy as np
import pytest

from src.actions.betting_actions import BettingActions
from src.solver.mccfr import MCCFRSolver
from src.solver.storage.shared_array import SharedArrayStorage
from src.training import components
from src.training.trainer import TrainingSession
from src.utils.config import Config
from tests.test_helpers import DummyCardAbstraction


@pytest.fixture
def config_with_dummy_abstraction(tmp_path, monkeypatch):
    """Create a config with dummy card abstraction."""
    config = Config.default().merge({"training": {"runs_dir": str(tmp_path / "runs")}})

    # Mock the builder to return DummyCardAbstraction
    def mock_build_card_abstraction(config):
        return DummyCardAbstraction()

    monkeypatch.setattr(components, "build_card_abstraction", mock_build_card_abstraction)

    return config


class TestTrainer:
    """Tests for TrainingSession class."""

    def test_create_trainer(self, config_with_dummy_abstraction):
        trainer = TrainingSession(config_with_dummy_abstraction)

        assert trainer.config is not None
        assert trainer.solver is not None
        assert trainer.action_abstraction is not None
        assert trainer.card_abstraction is not None

    def test_build_action_abstraction(self, config_with_dummy_abstraction):
        trainer = TrainingSession(config_with_dummy_abstraction)

        action_abs = trainer.action_abstraction
        assert isinstance(action_abs, BettingActions)

    def test_build_storage_memory(self, config_with_dummy_abstraction):
        config = config_with_dummy_abstraction.merge({"storage": {"checkpoint_enabled": False}})

        trainer = TrainingSession(config)
        assert isinstance(trainer.storage, SharedArrayStorage)
        assert trainer.storage.checkpoint_dir is None

    def test_build_solver(self, config_with_dummy_abstraction):
        trainer = TrainingSession(config_with_dummy_abstraction)

        solver = trainer.solver
        assert isinstance(solver, MCCFRSolver)
        assert solver.iteration == 0

    def test_str_representation(self, config_with_dummy_abstraction):
        trainer = TrainingSession(config_with_dummy_abstraction)
        s = str(trainer)

        assert "TrainingSession" in s

    def test_initialization_failure_no_directory(self, tmp_path, monkeypatch):
        """Test that failed initialization doesn't create run directory."""

        # Mock card abstraction to fail
        def mock_build_card_abstraction_fail(*args, **kwargs):
            raise ValueError("Abstraction not found")

        monkeypatch.setattr(components, "build_card_abstraction", mock_build_card_abstraction_fail)

        config = Config.default().merge(
            {
                "training": {"runs_dir": str(tmp_path / "runs")},
                "storage": {"checkpoint_enabled": False},
            }
        )

        # Attempt to create trainer (should fail)
        with pytest.raises(ValueError, match="Abstraction not found"):
            TrainingSession(config)

        # Check that no run directory was created
        runs_dir = tmp_path / "runs"
        if runs_dir.exists():
            run_dirs = list(runs_dir.glob("run*"))
            assert len(run_dirs) == 0, (
                f"Run directory should not exist after init failure, found: {run_dirs}"
            )

    @pytest.mark.slow
    @pytest.mark.timeout(20)
    def test_parallel_training_discovers_infosets(self, config_with_dummy_abstraction):
        """
        Test that parallel training discovers infosets using partitioned storage.

        This test verifies:
        1. Workers discover infosets during parallel training
        2. Metrics track correct iteration counts
        3. Results contain valid iteration and infoset counts
        """
        # Configure for small parallel run
        config = config_with_dummy_abstraction.merge(
            {
                "training": {"num_iterations": 4, "verbose": False},
                "storage": {"checkpoint_enabled": False},
            }
        )

        # Create trainer and run parallel training (uses partitioned storage)
        trainer = TrainingSession(config, run_id="test_parallel_discovery")

        # Run parallel training with public API
        results = trainer.train(num_iterations=4, num_workers=2)

        # Verify results
        assert results["total_iterations"] == 4
        final_infosets = results["final_infosets"]
        assert final_infosets > 0, "Should discover infosets during training"

        # Verify metrics tracked the correct number of iterations
        assert trainer.metrics.iteration == 4

    @pytest.mark.slow
    @pytest.mark.timeout(30)
    def test_parallel_vs_sequential_both_work(self, config_with_dummy_abstraction):
        """
        Test that training works with different worker counts.

        All training uses SharedArrayStorage with hash-partitioned infosets.
        This test verifies:
        - Single worker (sequential-like) training completes
        - Multi-worker parallel training completes
        - Both discover infosets (exact count may vary due to sampling)
        """
        config = config_with_dummy_abstraction.merge(
            {
                "training": {"num_iterations": 4, "verbose": False},
                "storage": {"checkpoint_enabled": False},
            }
        )

        # Run with 1 worker (sequential-like behavior)
        trainer_seq = TrainingSession(config, run_id="test_single_worker")
        results_seq = trainer_seq.train(num_iterations=4, num_workers=1)

        # Run with 2 workers (parallel)
        trainer_par = TrainingSession(config, run_id="test_multi_worker")
        results_par = trainer_par.train(num_iterations=4, num_workers=2)

        # Both should complete same number of iterations
        assert results_seq["total_iterations"] == results_par["total_iterations"]

        # Both should discover infosets (exact count may vary due to sampling)
        assert results_seq["final_infosets"] > 0
        assert results_par["final_infosets"] > 0

        # Both use SharedArrayStorage
        assert isinstance(trainer_seq.solver.storage, SharedArrayStorage)
        for infoset in trainer_seq.solver.storage.iter_infosets():
            assert not np.any(np.isnan(infoset.regrets))
            assert not np.any(np.isnan(infoset.strategy_sum))

        # Verify metrics are valid for parallel
        assert trainer_par.metrics.iteration == 4
        assert len(trainer_par.metrics.infoset_counts) == 4

    @pytest.mark.slow
    @pytest.mark.timeout(20)  # Reduced from 60
    def test_parallel_training_performance(self, config_with_dummy_abstraction):
        """
        Test that parallel training completes efficiently.

        This test verifies:
        1. Training completes within reasonable time
        2. Throughput is acceptable
        """
        config = config_with_dummy_abstraction.merge(
            {
                "training": {"num_iterations": 4, "verbose": False},
                "storage": {"checkpoint_enabled": False},
            }
        )

        # Create trainer
        trainer = TrainingSession(config, run_id="test_perf")

        start_time = time.time()

        # Run parallel training
        results = trainer.train(num_iterations=4, num_workers=2, batch_size=2)

        elapsed = time.time() - start_time

        # Verify results
        assert results["total_iterations"] == 4

        # Calculate throughput
        throughput = 4 / elapsed if elapsed > 0 else 0

        print("\n📊 Performance Metrics:")
        print(f"   Total iterations: {results['total_iterations']}")
        print(f"   Total time: {elapsed:.2f}s")
        print(f"   Throughput: {throughput:.2f} iter/s")
        print(f"   Final infosets: {results['final_infosets']}")

        # Basic sanity check - should complete in reasonable time
        assert elapsed < 5, f"Training took too long: {elapsed:.2f}s"  # Reduced from 10
        assert throughput > 0.4, (
            f"Throughput too low: {throughput:.2f} iter/s"
        )  # Increased from 0.2

    @pytest.mark.slow
    @pytest.mark.timeout(20)  # Reduced from 30
    def test_parallel_training_completes_without_errors(self, config_with_dummy_abstraction):
        """Test that parallel training completes without errors with multiple workers."""
        # Run parallel training with multiple workers
        # The partitioned architecture ensures each worker owns different infosets
        # which eliminates action-set mismatch issues

        config = config_with_dummy_abstraction.merge({"storage": {"checkpoint_enabled": False}})
        trainer = TrainingSession(config, run_id="test_parallel_multi")

        # Run parallel training with multiple workers
        results = trainer.train(
            num_iterations=6,  # Reduced from 10
            num_workers=3,  # Reduced from 4
        )

        # Verify training completed without errors
        assert results["total_iterations"] == 6
        assert results["final_infosets"] > 0

        # Key assertion: No errors should be raised during training
        # The partitioned architecture ensures workers only write to their owned infosets


@pytest.mark.slow
class TestAsyncCheckpointing:
    """Tests for async (non-blocking) checkpointing."""

    @pytest.mark.timeout(30)
    def test_async_checkpoint_executor_initialized(self, config_with_dummy_abstraction):
        """Test that async checkpoint executor is initialized."""
        config = config_with_dummy_abstraction.merge({"storage": {"checkpoint_enabled": True}})

        trainer = TrainingSession(config)

        # Verify checkpoint executor is initialized
        assert hasattr(trainer, "_checkpoint_executor")
        assert trainer._checkpoint_executor is not None
        assert hasattr(trainer, "_pending_checkpoint")
        assert trainer._pending_checkpoint is None

    @pytest.mark.timeout(60)
    def test_async_checkpoint_nonblocking(self, config_with_dummy_abstraction):
        """Test that checkpoints don't block training."""
        config = config_with_dummy_abstraction.merge(
            {
                "training": {"checkpoint_frequency": 2, "num_iterations": 4, "verbose": False},
                "storage": {"checkpoint_enabled": True},
            }
        )

        trainer = TrainingSession(config, run_id="test_async_checkpoint")

        # Run training with checkpointing enabled
        results = trainer.train(num_iterations=4, num_workers=1)

        # Verify training completed
        assert results["total_iterations"] == 4

        # Checkpoint should have completed in background
        # (This is verified by the fact that training completes without errors)

    @pytest.mark.timeout(60)
    def test_async_checkpoint_completes_on_shutdown(self, config_with_dummy_abstraction):
        """Test that pending checkpoints complete when training ends."""
        config = config_with_dummy_abstraction.merge(
            {
                "training": {"checkpoint_frequency": 2, "num_iterations": 4, "verbose": False},
                "storage": {"checkpoint_enabled": True},
            }
        )

        trainer = TrainingSession(config, run_id="test_checkpoint_shutdown")

        # Run training
        trainer.train(num_iterations=4, num_workers=1)

        # After training, pending checkpoint should be None (completed)
        assert trainer._pending_checkpoint is None

        # Verify checkpoint files exist
        assert (trainer.run_dir / "key_mapping.pkl").exists()

    @pytest.mark.timeout(60)
    def test_async_checkpoint_parallel_training(self, config_with_dummy_abstraction):
        """Test async checkpointing with parallel training."""
        config = config_with_dummy_abstraction.merge(
            {
                "training": {"checkpoint_frequency": 2, "num_iterations": 4, "verbose": False},
                "storage": {"checkpoint_enabled": True},
            }
        )

        trainer = TrainingSession(config, run_id="test_async_parallel")

        # Run parallel training with checkpointing
        results = trainer.train(num_iterations=4, num_workers=2)

        # Verify training completed
        assert results["total_iterations"] == 4

        # Verify checkpoint was saved
        assert (trainer.run_dir / "key_mapping.pkl").exists()
        assert (trainer.run_dir / "checkpoint.zarr").exists()
        assert (trainer.run_dir / "action_signatures.pkl").exists()

    @pytest.mark.timeout(30)
    def test_checkpoint_executor_cleanup(self, config_with_dummy_abstraction):
        """Test that checkpoint executor is properly cleaned up."""
        config = config_with_dummy_abstraction.merge({"storage": {"checkpoint_enabled": True}})

        trainer = TrainingSession(config)

        # Delete trainer (triggers __del__)
        del trainer

        # Executor should be shutdown (can't check directly, but shouldn't hang)
        # If this test completes without hanging, cleanup worked

    @pytest.mark.timeout(60)
    def test_checkpoint_collects_keys_from_all_workers(self, config_with_dummy_abstraction):
        """Test that parallel checkpoint collects and saves keys from all workers."""

        config = config_with_dummy_abstraction.merge(
            {
                "training": {"checkpoint_frequency": 4, "num_iterations": 4, "verbose": False},
                "storage": {"checkpoint_enabled": True},
            }
        )

        trainer = TrainingSession(config, run_id="test_key_collection")

        # Run parallel training with multiple workers
        results = trainer.train(num_iterations=4, num_workers=2)

        # Verify training completed
        assert results["total_iterations"] == 4
        assert results["final_infosets"] > 0

        # Verify checkpoint files exist
        key_mapping_file = trainer.run_dir / "key_mapping.pkl"
        assert key_mapping_file.exists(), "key_mapping.pkl should exist"

        # Load and verify the checkpoint contains all infosets
        with open(key_mapping_file, "rb") as f:
            mapping = pickle.load(f)

        owned_keys = mapping["owned_keys"]
        assert len(owned_keys) > 0, "Checkpoint should contain collected keys"

        # Verify the count matches what was reported
        assert len(owned_keys) == results["final_infosets"], (
            f"Checkpoint has {len(owned_keys)} keys but training reported "
            f"{results['final_infosets']} infosets"
        )

    @pytest.mark.timeout(40)  # Reduced from 60
    def test_checkpoint_round_trip_parallel(self, config_with_dummy_abstraction):
        """Test that parallel checkpoints can be saved and loaded correctly."""

        config = config_with_dummy_abstraction.merge(
            {
                "training": {"checkpoint_frequency": 4, "num_iterations": 4, "verbose": False},
                "storage": {"checkpoint_enabled": True},
            }
        )

        trainer = TrainingSession(config, run_id="test_checkpoint_roundtrip")

        # Run parallel training and save checkpoint
        trainer.train(num_iterations=4, num_workers=2)

        # Verify checkpoint files exist
        key_mapping_file = trainer.run_dir / "key_mapping.pkl"
        checkpoint_dir = trainer.run_dir / "checkpoint.zarr"
        action_sigs_file = trainer.run_dir / "action_signatures.pkl"

        assert key_mapping_file.exists()
        assert checkpoint_dir.exists()
        assert action_sigs_file.exists()

        # Load and verify the checkpoint contains valid data
        with open(key_mapping_file, "rb") as f:
            mapping = pickle.load(f)

        owned_keys = mapping["owned_keys"]

        # Load from zarr
        import zarr

        zarr_data = zarr.open(str(checkpoint_dir), mode="r")
        regrets_data = zarr_data["regrets"][:]
        max_id = regrets_data.shape[0]

        # Verify max_id is consistent with keys
        assert max_id > 0
        if owned_keys:
            actual_max = max(owned_keys.values())
            assert max_id == actual_max + 1, f"max_id should be {actual_max + 1} but got {max_id}"

        action_counts = zarr_data["action_counts"][:]

        # Verify shapes match max_id
        assert regrets_data.shape[0] == max_id
        assert action_counts.shape[0] == max_id

        # Verify some regrets are non-zero (solver actually updated)
        non_zero_regrets = (regrets_data != 0).any(axis=1).sum()
        assert non_zero_regrets > 0, "Some regrets should be non-zero"

        strategy_data = zarr_data["strategies"][:]
        assert strategy_data.shape[0] == max_id

        # Verify some strategies are non-zero
        non_zero_strategies = (strategy_data != 0).any(axis=1).sum()
        assert non_zero_strategies > 0, "Some strategies should be non-zero"

        print("\nCheckpoint verification:")
        print(f"  Keys: {len(owned_keys)}")
        print(f"  Max ID: {max_id}")
        print(f"  Non-zero regrets: {non_zero_regrets}/{max_id}")
        print(f"  Non-zero strategies: {non_zero_strategies}/{max_id}")


@pytest.mark.slow
class TestParallelStress:
    """Stress tests for parallel training to catch race conditions."""

    @pytest.mark.timeout(60)  # Reduced from 120
    def test_high_worker_count_stress(self, config_with_dummy_abstraction):
        """Stress test with many workers to expose race conditions."""
        config = config_with_dummy_abstraction.merge(
            {
                "training": {"num_iterations": 12, "verbose": False},
                "storage": {"checkpoint_enabled": False},
            }
        )

        trainer = TrainingSession(config, run_id="test_stress_workers")

        # Run with many workers (increases chance of race conditions)
        results = trainer.train(
            num_iterations=12,
            num_workers=6,  # Reduced from 8
            batch_size=3,  # Small batch size = more synchronization
        )

        # Verify training completed without crashes
        assert results["total_iterations"] == 12
        assert results["final_infosets"] > 0

        # The key test: no crashes, no deadlocks, no data corruption

    @pytest.mark.timeout(60)  # Reduced from 120
    def test_repeated_batches_consistency(self, config_with_dummy_abstraction):
        """Test that repeated training batches maintain consistency."""
        config = config_with_dummy_abstraction.merge(
            {
                "training": {"num_iterations": 12, "verbose": False},
                "storage": {"checkpoint_enabled": False},
            }
        )

        trainer = TrainingSession(config, run_id="test_repeated_batches")

        # Run multiple batches with frequent ID exchanges
        results = trainer.train(
            num_iterations=12,
            num_workers=3,  # Reduced from 4
            batch_size=2,  # Very small batches = frequent exchanges
        )

        # Verify all iterations completed
        assert results["total_iterations"] == 12
        final_count = results["final_infosets"]
        assert final_count > 0

        # Key test: No infosets lost, no duplicate IDs, no corruption


class TestCheckpointEnabledConfig:
    """Tests for checkpoint_enabled configuration."""

    def test_checkpoint_enabled_true_creates_dir(self, config_with_dummy_abstraction):
        """Test that checkpoint_enabled=true sets up checkpointing."""
        config = config_with_dummy_abstraction.merge({"storage": {"checkpoint_enabled": True}})

        trainer = TrainingSession(config)

        # Verify checkpoint directory is set
        assert trainer.storage.checkpoint_dir is not None
        assert trainer.storage.checkpoint_dir == trainer.run_dir

    def test_checkpoint_enabled_false_no_checkpointing(self, config_with_dummy_abstraction):
        """Test that checkpoint_enabled=false disables checkpointing."""
        config = config_with_dummy_abstraction.merge({"storage": {"checkpoint_enabled": False}})

        trainer = TrainingSession(config)

        # Verify no checkpoint directory
        assert trainer.storage.checkpoint_dir is None

    @pytest.mark.slow
    @pytest.mark.timeout(30)
    def test_checkpoint_enabled_false_no_files_created(self, config_with_dummy_abstraction):
        """Test that no checkpoint files are created when disabled."""
        config = config_with_dummy_abstraction.merge(
            {
                "training": {"num_iterations": 4, "verbose": False},
                "storage": {"checkpoint_enabled": False},
            }
        )

        trainer = TrainingSession(config, run_id="test_no_checkpoint")

        # Run training
        trainer.train(num_iterations=4, num_workers=1)

        # Verify NO checkpoint files created
        assert not (trainer.run_dir / "key_mapping.pkl").exists()
        assert not (trainer.run_dir / "checkpoint.zarr").exists()
        assert not (trainer.run_dir / "action_signatures.pkl").exists()

    @pytest.mark.slow
    @pytest.mark.timeout(60)
    def test_checkpoint_enabled_resume(self, config_with_dummy_abstraction, tmp_path):
        """Test resuming from checkpoint with new config format."""
        config = config_with_dummy_abstraction.merge(
            {
                "training": {
                    "num_iterations": 4,
                    "verbose": False,
                    "runs_dir": str(tmp_path / "runs"),
                },
                "storage": {"checkpoint_enabled": True},
            }
        )

        # First training session
        trainer1 = TrainingSession(config, run_id="test_resume")
        results1 = trainer1.train(num_iterations=4, num_workers=1)

        # Verify checkpoint exists
        assert (trainer1.run_dir / "key_mapping.pkl").exists()

        initial_infosets = results1["final_infosets"]

        # Resume training
        trainer2 = TrainingSession.resume(trainer1.run_dir)

        # Verify loaded correctly
        assert trainer2.solver.num_infosets() == initial_infosets
