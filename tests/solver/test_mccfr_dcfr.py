"""
Integration tests for DCFR and regret-based pruning.

Tests that DCFR and pruning work correctly with the MCCFR solver.
"""

import pytest

from src.actions.action_model import ActionModel
from src.solver.mccfr import MCCFRSolver
from src.solver.storage.shared_array import SharedArrayStorage
from tests.test_helpers import DummyCardAbstraction, make_test_config


class TestDCFR:
    """Tests for DCFR (Discounted CFR) integration."""

    @pytest.mark.slow
    def test_dcfr_training_runs(self):
        """DCFR should complete training iterations without error."""
        action_abs = ActionModel()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test_dcfr", is_coordinator=True
        )

        config = make_test_config(
            seed=42,
            enable_dcfr=True,
            dcfr_alpha=1.5,
            dcfr_beta=0.0,
            dcfr_gamma=2.0,
        )
        solver = MCCFRSolver(action_abs, card_abs, storage, config=config)

        # Run multiple iterations
        for _ in range(10):
            solver.train_iteration()

        assert solver.iteration == 10
        assert solver.num_infosets() > 0

    @pytest.mark.slow
    def test_dcfr_disables_linear_cfr(self):
        """Enabling DCFR should disable Linear CFR with warning."""
        action_abs = ActionModel()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test_dcfr", is_coordinator=True
        )

        config = make_test_config(
            seed=42,
            linear_cfr=True,
            enable_dcfr=True,
        )

        with pytest.warns(UserWarning, match="DCFR and Linear CFR"):
            solver = MCCFRSolver(action_abs, card_abs, storage, config=config)

        # Linear CFR should be disabled
        assert not solver.linear_cfr
        assert solver.enable_dcfr

    @pytest.mark.slow
    def test_dcfr_convergence(self):
        """DCFR should converge (strategies should update)."""
        action_abs = ActionModel()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test_dcfr", is_coordinator=True
        )

        config = make_test_config(
            seed=42,
            enable_dcfr=True,
            cfr_plus=True,
        )
        solver = MCCFRSolver(action_abs, card_abs, storage, config=config)

        # Train for enough iterations
        for _ in range(100):
            solver.train_iteration()

        # Check that strategies have been updated
        updated_count = 0
        for infoset in storage.iter_infosets():
            if infoset.strategy_sum.sum() > 0:
                updated_count += 1

        assert updated_count > 0


class TestPruning:
    """Tests for regret-based action pruning."""

    @pytest.mark.slow
    def test_pruning_training_runs(self):
        """Pruning should complete training iterations without error."""
        action_abs = ActionModel()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test_pruning", is_coordinator=True
        )

        config = make_test_config(
            seed=42,
            enable_pruning=True,
            pruning_threshold=300.0,
            prune_start_iteration=10,
            prune_reactivate_frequency=20,
            sampling_method="external",
        )
        solver = MCCFRSolver(action_abs, card_abs, storage, config=config)

        for _ in range(50):
            solver.train_iteration()

        assert solver.iteration == 50
        assert solver.num_infosets() > 0

    @pytest.mark.slow
    def test_pruning_disabled_with_outcome_sampling(self):
        """Pruning should be disabled with outcome sampling."""
        action_abs = ActionModel()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test_pruning", is_coordinator=True
        )

        config = make_test_config(
            seed=42,
            enable_pruning=True,
            sampling_method="outcome",
        )

        with pytest.warns(UserWarning, match="only supported with external sampling"):
            solver = MCCFRSolver(action_abs, card_abs, storage, config=config)

        assert not solver.enable_pruning

    @pytest.mark.slow
    def test_pruning_state_exists(self):
        """Pruning state should exist in infosets when pruning is enabled."""
        action_abs = ActionModel()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test_pruning", is_coordinator=True
        )

        config = make_test_config(
            seed=42,
            enable_pruning=True,
            pruning_threshold=200.0,
            prune_start_iteration=10,
        )
        solver = MCCFRSolver(action_abs, card_abs, storage, config=config)

        # Run iterations
        for _ in range(100):
            solver.train_iteration()

        # Check that infosets have pruned arrays
        for infoset in storage.iter_infosets():
            assert hasattr(infoset, "pruned")
            assert len(infoset.pruned) == infoset.num_actions
            # At least one action should be unpruned (safety mechanism)
            assert not all(infoset.pruned)


class TestDCFRWithPruning:
    """Tests for DCFR combined with pruning."""

    @pytest.mark.slow
    def test_dcfr_and_pruning_together(self):
        """DCFR and pruning should work together without errors."""
        action_abs = ActionModel()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test_both", is_coordinator=True
        )

        config = make_test_config(
            seed=42,
            enable_dcfr=True,
            dcfr_alpha=1.5,
            dcfr_beta=0.0,
            dcfr_gamma=2.0,
            enable_pruning=True,
            pruning_threshold=300.0,
            prune_start_iteration=10,
            sampling_method="external",
        )
        solver = MCCFRSolver(action_abs, card_abs, storage, config=config)

        # Run training
        for _ in range(50):
            solver.train_iteration()

        assert solver.iteration == 50
        assert solver.num_infosets() > 0
        assert solver.enable_dcfr
        assert solver.enable_pruning

    @pytest.mark.slow
    def test_dcfr_and_pruning_convergence(self):
        """DCFR + pruning should produce valid strategies."""
        action_abs = ActionModel()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test_both", is_coordinator=True
        )

        config = make_test_config(
            seed=42,
            enable_dcfr=True,
            enable_pruning=True,
            cfr_plus=True,
        )
        solver = MCCFRSolver(action_abs, card_abs, storage, config=config)

        # Train for significant iterations
        for _ in range(100):
            solver.train_iteration()

        # Verify strategies are valid (probabilities sum to 1)
        for infoset in storage.iter_infosets():
            strategy = infoset.get_average_strategy()
            # Check that strategy sums to approximately 1
            assert 0.99 <= strategy.sum() <= 1.01
            # Check no negative probabilities
            assert all(p >= 0 for p in strategy)
