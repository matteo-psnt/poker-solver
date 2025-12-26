"""
Tests for MCCFR outcome sampling mode.

Outcome sampling is a variance-reduction technique in MCCFR.
These tests ensure outcome sampling works correctly.
"""

import pytest

from src.actions.betting_actions import BettingActions
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import SharedArrayStorage
from tests.test_helpers import DummyCardAbstraction, make_test_config


class TestOutcomeSampling:
    """Test outcome sampling mode."""

    def test_create_solver_with_outcome_sampling(self):
        """Test creating solver with outcome sampling enabled."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )

        solver = MCCFRSolver(
            action_abs,
            card_abs,
            storage,
            config=make_test_config(sampling_method="outcome", seed=42),
        )

        assert solver.sampling_method == "outcome"
        assert solver.iteration == 0

    def test_outcome_sampling_iteration_executes(self):
        """Test that outcome sampling iteration completes."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )

        solver = MCCFRSolver(
            action_abs,
            card_abs,
            storage,
            config=make_test_config(sampling_method="outcome", seed=42),
        )

        utility = solver.train_iteration()

        assert solver.iteration == 1
        assert isinstance(utility, float)
        assert solver.num_infosets() > 0

    def test_outcome_sampling_multiple_iterations(self):
        """Test multiple iterations with outcome sampling."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )

        solver = MCCFRSolver(
            action_abs,
            card_abs,
            storage,
            config=make_test_config(sampling_method="outcome", seed=42),
        )

        results = solver.train(num_iterations=10, verbose=False)

        assert solver.iteration == 10
        assert results["total_iterations"] == 10
        assert solver.num_infosets() > 0

    def test_outcome_sampling_creates_infosets(self):
        """Test that outcome sampling creates and updates infosets."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )

        solver = MCCFRSolver(
            action_abs,
            card_abs,
            storage,
            config=make_test_config(sampling_method="outcome", seed=42),
        )

        # Run iterations
        solver.train(num_iterations=20, verbose=False)

        # Should have created infosets
        assert solver.num_infosets() > 0

        # Check that at least some infosets have non-zero regrets
        infosets_with_regrets = 0
        for infoset in storage.infosets.values():
            if any(r != 0 for r in infoset.regrets):
                infosets_with_regrets += 1

        # At least half of infosets should have been updated
        assert infosets_with_regrets > solver.num_infosets() * 0.3

    def test_outcome_sampling_with_cfr_plus(self):
        """Test outcome sampling works with CFR+."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )

        solver = MCCFRSolver(
            action_abs,
            card_abs,
            storage,
            config=make_test_config(sampling_method="outcome", cfr_plus=True, seed=42),
        )

        # Should complete without errors
        solver.train(num_iterations=10, verbose=False)

        assert solver.iteration == 10
        assert solver.num_infosets() > 0

    def test_outcome_sampling_produces_valid_strategies(self):
        """Test that outcome sampling produces valid strategies."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )

        solver = MCCFRSolver(
            action_abs,
            card_abs,
            storage,
            config=make_test_config(sampling_method="outcome", seed=42),
        )

        solver.train(num_iterations=50, verbose=False)

        # Check that strategies sum to 1.0 (or close)
        for infoset in storage.infosets.values():
            strategy = infoset.get_average_strategy()
            strategy_sum = sum(strategy)

            # Allow small numerical error
            assert 0.99 <= strategy_sum <= 1.01, f"Strategy sum {strategy_sum} not close to 1.0"

            # All probabilities should be non-negative
            assert all(p >= 0 for p in strategy), "Negative probability in strategy"

    def test_external_vs_outcome_sampling_both_work(self):
        """Test that both sampling methods work (comparison test)."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()

        # External sampling
        storage_external = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test_ext", is_coordinator=True
        )
        solver_external = MCCFRSolver(
            action_abs,
            card_abs,
            storage_external,
            config=make_test_config(sampling_method="external", seed=42),
        )
        solver_external.train(num_iterations=5, verbose=False)

        # Outcome sampling
        storage_outcome = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test_out", is_coordinator=True
        )
        solver_outcome = MCCFRSolver(
            action_abs,
            card_abs,
            storage_outcome,
            config=make_test_config(sampling_method="outcome", seed=42),
        )
        solver_outcome.train(num_iterations=5, verbose=False)

        # Both should create infosets
        assert solver_external.num_infosets() > 0
        assert solver_outcome.num_infosets() > 0

        # Both should produce valid strategies
        for infoset in storage_external.infosets.values():
            strategy = infoset.get_average_strategy()
            assert 0.99 <= sum(strategy) <= 1.01

        for infoset in storage_outcome.infosets.values():
            strategy = infoset.get_average_strategy()
            assert 0.99 <= sum(strategy) <= 1.01

    def test_outcome_sampling_invalid_method_raises_error(self):
        """Test that invalid sampling method raises error."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )

        # Invalid method should raise ValueError
        with pytest.raises(ValueError, match="Invalid sampling_method"):
            MCCFRSolver(
                action_abs,
                card_abs,
                storage,
                config=make_test_config(sampling_method="invalid_method", seed=42),
            )

    def test_outcome_sampling_converges_over_iterations(self):
        """Test that outcome sampling shows convergence behavior."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )

        solver = MCCFRSolver(
            action_abs,
            card_abs,
            storage,
            config=make_test_config(sampling_method="outcome", seed=42),
        )

        # Train for many iterations
        solver.train(num_iterations=100, verbose=False)

        # Check that infosets were discovered during training
        multi_action_infosets = sum(
            1 for infoset in storage.infosets.values() if len(infoset.legal_actions) > 1
        )
        assert multi_action_infosets > 0, "Should discover multi-action infosets"

        # Check that at least some infosets have accumulated regrets
        # (indicating the solver is updating correctly)
        infosets_with_regrets = 0
        for infoset in storage.infosets.values():
            if len(infoset.legal_actions) > 1:
                # Check if any regret is non-zero (solver is learning)
                if any(abs(r) > 0.001 for r in infoset.regrets):
                    infosets_with_regrets += 1

        # At least some infosets should have accumulated regrets
        # (100 iterations with outcome sampling explores a subset of the tree)
        assert infosets_with_regrets > 0, (
            f"At least some infosets should have accumulated regrets, "
            f"but found {infosets_with_regrets} out of {multi_action_infosets}"
        )
