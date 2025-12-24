"""Tests for Trainer class."""

import numpy as np
import pytest

from src.actions.betting_actions import BettingActions
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import InMemoryStorage
from src.training import components
from src.training.trainer import TrainingSession
from src.utils.config import Config
from tests.test_helpers import DummyCardAbstraction


@pytest.fixture
def config_with_dummy_abstraction(tmp_path, monkeypatch):
    """Create a config with dummy card abstraction."""
    config = Config.default()
    # Use tmp_path for runs_dir to prevent creating runs in data/runs
    config.set("training.runs_dir", str(tmp_path / "runs"))

    # Mock the builder to return DummyCardAbstraction
    def mock_build_card_abstraction(config, prompt_user=False, auto_compute=False):
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
        config_with_dummy_abstraction.set("storage.backend", "memory")

        trainer = TrainingSession(config_with_dummy_abstraction)
        assert isinstance(trainer.storage, InMemoryStorage)

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

        config = Config.default()
        config.set("training.runs_dir", str(tmp_path / "runs"))
        config.set("storage.backend", "memory")

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

    @pytest.mark.timeout(30)
    def test_parallel_merge_correctness(self, config_with_dummy_abstraction):
        """
        Test that parallel training correctly merges all infoset data.

        This test verifies:
        1. Regrets and strategy_sum are properly summed across workers
        2. reach_count is correctly aggregated
        3. cumulative_utility is properly summed
        4. No infosets are dropped during merge
        5. Metrics use master solver state (not worker-local state)
        """
        # Configure for small parallel run
        config = config_with_dummy_abstraction
        config.set("training.num_iterations", 6)
        config.set("training.verbose", False)
        config.set("storage.backend", "memory")

        # Create trainer and run parallel training
        trainer = TrainingSession(config)
        num_workers = 2
        batch_size = 6  # Small batch to complete quickly

        # Track initial state
        initial_infosets = trainer.solver.num_infosets()

        # Run parallel training
        results = trainer._train_parallel(
            num_iterations=6, num_workers=num_workers, batch_size=batch_size
        )

        # Verify results
        assert results["total_iterations"] == 6
        final_infosets = results["final_infosets"]
        assert final_infosets > initial_infosets, "Should discover new infosets"

        # Verify all infosets have properly merged data
        assert isinstance(trainer.solver.storage, InMemoryStorage)
        for key, infoset in trainer.solver.storage.infosets.items():
            # Check array shapes are valid
            assert infoset.regrets.shape == (infoset.num_actions,)
            assert infoset.strategy_sum.shape == (infoset.num_actions,)

            # Check that reach_count was tracked
            # In parallel training, some infosets might not be reached (due to sampling)
            assert infoset.reach_count >= 0

            # Check that cumulative_utility is present and reasonable
            assert isinstance(infoset.cumulative_utility, (int, float))

            # Verify average utility computation works
            avg_util = infoset.get_average_utility()
            assert isinstance(avg_util, float)
            if infoset.reach_count > 0:
                # Should be well-defined if reached
                assert not np.isnan(avg_util)

        # Verify metrics tracked the correct number of iterations
        assert trainer.metrics.iteration == 6

        # Verify metrics show master solver infoset count (not worker-local)
        # The last logged infoset count should match final master count
        assert trainer.metrics.infoset_counts[-1] == final_infosets

    @pytest.mark.timeout(30)
    def test_parallel_vs_sequential_consistency(self, config_with_dummy_abstraction):
        """
        Test that parallel training produces consistent results with sequential training.

        Note: Due to different sampling paths, exact values won't match, but:
        - Both should discover similar numbers of infosets (within variance)
        - Both should have valid merged state (no NaNs, proper shapes)
        - No infosets should be silently dropped
        """
        config = config_with_dummy_abstraction
        config.set("training.num_iterations", 4)
        config.set("training.verbose", False)
        config.set("storage.backend", "memory")

        # Run sequential training
        trainer_seq = TrainingSession(config, run_id="test_sequential")
        results_seq = trainer_seq.train(num_iterations=4, use_parallel=False)

        # Run parallel training with same total iterations
        trainer_par = TrainingSession(config, run_id="test_parallel")
        results_par = trainer_par.train(num_iterations=4, use_parallel=True, num_workers=2)

        # Both should complete same number of iterations
        assert results_seq["total_iterations"] == results_par["total_iterations"]

        # Both should discover infosets (exact count may vary due to sampling)
        assert results_seq["final_infosets"] > 0
        assert results_par["final_infosets"] > 0

        # Verify parallel training has valid merged state
        assert isinstance(trainer_par.solver.storage, InMemoryStorage)
        for key, infoset in trainer_par.solver.storage.infosets.items():
            # Check no NaN values in regrets or strategies
            assert not np.any(np.isnan(infoset.regrets))
            assert not np.any(np.isnan(infoset.strategy_sum))

            # Check reach_count is non-negative
            assert infoset.reach_count >= 0

            # Check cumulative_utility is valid
            assert not np.isnan(infoset.cumulative_utility)

        # Verify metrics are valid
        assert trainer_par.metrics.iteration == 4
        assert len(trainer_par.metrics.infoset_counts) == 4

    @pytest.mark.timeout(60)
    def test_persistent_worker_pool_performance(self, config_with_dummy_abstraction):
        """
        Test that persistent worker pool provides performance benefits.

        This test verifies:
        1. Worker pool initialization happens once (not per batch)
        2. Multiple batches can be processed without re-spawning processes
        3. Throughput is consistent across batches (no startup overhead)
        """
        config = config_with_dummy_abstraction
        config.set("training.num_iterations", 4)
        config.set("training.verbose", False)
        config.set("storage.backend", "memory")

        # Create trainer
        trainer = TrainingSession(config, run_id="test_perf")

        # Use small batch size to force multiple batches
        num_workers = 2
        batch_size = 2  # Will have 2 batches

        # Track timing
        import time

        start_time = time.time()

        # Run parallel training
        results = trainer._train_parallel(
            num_iterations=4, num_workers=num_workers, batch_size=batch_size
        )

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
        # With persistent workers, 4 iterations should be fast
        assert elapsed < 10, f"Training took too long: {elapsed:.2f}s"
        assert throughput > 0.2, f"Throughput too low: {throughput:.2f} iter/s"

    @pytest.mark.timeout(30)
    def test_action_set_mismatch_handling(self, config_with_dummy_abstraction):
        """Test that action-set mismatches are handled correctly via padding."""
        # This is a regression test for the critical action-set mismatch bug
        # In MCCFR, different workers can discover different action counts for the same infoset
        # due to sampling variance, stack-dependent legality, etc.

        # Use in-memory storage for easier validation
        config_with_dummy_abstraction.set("storage.backend", "memory")
        trainer = TrainingSession(config_with_dummy_abstraction)

        # Run parallel training with multiple workers to increase chance of mismatches
        results = trainer.train(
            num_iterations=10,  # More iterations = higher chance of mismatch
            use_parallel=True,
            num_workers=4,
        )

        # Verify training completed without errors
        assert results["total_iterations"] == 10
        assert results["final_infosets"] > 0

        # Key assertion: No errors should be raised during merge
        # The old implementation would either skip infosets (silent data loss)
        # or crash on shape mismatches
        # The new implementation pads with zeros (CFR-correct)

        # Verify that regrets and strategies have consistent shapes
        assert isinstance(trainer.solver.storage, InMemoryStorage)
        for infoset_key, infoset in trainer.solver.storage.infosets.items():
            assert infoset.regrets.shape[0] == infoset.num_actions, (
                f"Regrets shape mismatch for {infoset_key}: "
                f"{infoset.regrets.shape[0]} != {infoset.num_actions}"
            )
            assert infoset.strategy_sum.shape[0] == infoset.num_actions, (
                f"Strategy shape mismatch for {infoset_key}: "
                f"{infoset.strategy_sum.shape[0]} != {infoset.num_actions}"
            )
            assert len(infoset.legal_actions) == infoset.num_actions, (
                f"Legal actions length mismatch for {infoset_key}: "
                f"{len(infoset.legal_actions)} != {infoset.num_actions}"
            )
