"""
Integration tests for DCFR and regret-based pruning.

Tests that DCFR and pruning work correctly with the MCCFR solver.
"""

import pytest
from pydantic import ValidationError

from src.core.actions.action_model import ActionModel
from src.engine.solver.mccfr import MCCFRSolver
from tests.test_helpers import DummyCardAbstraction, build_test_storage, make_test_config


class TestDCFR:
    """Tests for DCFR (Discounted CFR) integration."""

    @pytest.mark.slow
    def test_dcfr_training_runs(self):
        """DCFR should complete training iterations without error."""
        action_abs = ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        storage = build_test_storage(
            num_workers=1, worker_id=0, session_id="test_dcfr", is_coordinator=True
        )

        config = make_test_config(
            seed=42,
            iteration_weighting="dcfr",
            dcfr_alpha=1.5,
            dcfr_beta=0.0,
            dcfr_gamma=2.0,
        )
        solver = MCCFRSolver(action_abs, card_abs, storage, config=config)

        for _ in range(10):
            solver.train_iteration()

        assert solver.iteration == 10
        assert solver.num_infosets() > 0

    @pytest.mark.slow
    def test_dcfr_weighting_reflected_in_solver(self):
        """iteration_weighting='dcfr' should be reflected in solver config."""
        action_abs = ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        storage = build_test_storage(
            num_workers=1, worker_id=0, session_id="test_dcfr", is_coordinator=True
        )

        config = make_test_config(seed=42, iteration_weighting="dcfr")
        solver = MCCFRSolver(action_abs, card_abs, storage, config=config)

        assert solver.config.solver.iteration_weighting == "dcfr"

    @pytest.mark.slow
    def test_linear_weighting_reflected_in_solver(self):
        """iteration_weighting='linear' should be reflected in solver config."""
        action_abs = ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        storage = build_test_storage(
            num_workers=1, worker_id=0, session_id="test_linear", is_coordinator=True
        )

        config = make_test_config(seed=42, iteration_weighting="linear")
        solver = MCCFRSolver(action_abs, card_abs, storage, config=config)

        assert solver.config.solver.iteration_weighting == "linear"

    @pytest.mark.slow
    def test_dcfr_convergence(self):
        """DCFR should converge (strategies should update)."""
        action_abs = ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        storage = build_test_storage(
            num_workers=1, worker_id=0, session_id="test_dcfr", is_coordinator=True
        )

        config = make_test_config(
            seed=42,
            iteration_weighting="dcfr",
            cfr_plus=True,
        )
        solver = MCCFRSolver(action_abs, card_abs, storage, config=config)

        for _ in range(100):
            solver.train_iteration()

        updated_count = 0
        for infoset in storage.iter_infosets():
            if infoset.strategy_sum.sum() > 0:
                updated_count += 1

        assert updated_count > 0


class TestPruning:
    """Regret-based pruning, derived live from the persistent regrets each visit
    (``InfoSet.pruned_mask``) — no stored mask. The mask logic is unit-tested in
    ``test_infoset.py``; here we check the config guard and end-to-end training."""

    def test_pruning_requires_external_sampling(self):
        """Pruning skips the traverser's own dominated actions, which only external
        sampling enumerates; outcome sampling is rejected at load."""
        with pytest.raises(ValidationError, match="requires sampling_method='external'"):
            make_test_config(seed=42, enable_pruning=True, sampling_method="outcome")

    @pytest.mark.slow
    def test_pruning_training_produces_valid_strategies(self):
        """Training with pruning enabled runs to completion and yields valid averages."""
        action_abs = ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        storage = build_test_storage(
            num_workers=1, worker_id=0, session_id="test_pruning", is_coordinator=True
        )
        config = make_test_config(
            seed=42,
            enable_pruning=True,
            pruning_threshold=1.0,  # aggressive so pruning actually fires at small scale
            prune_start_iteration=10,
            prune_reactivate_frequency=25,
            sampling_method="external",
        )
        solver = MCCFRSolver(action_abs, card_abs, storage, config=config)

        for _ in range(100):
            solver.train_iteration()

        assert solver.iteration == 100
        assert solver.num_infosets() > 0
        for infoset in storage.iter_infosets():
            strategy = infoset.get_average_strategy()
            assert 0.99 <= strategy.sum() <= 1.01
            assert all(p >= 0 for p in strategy)
