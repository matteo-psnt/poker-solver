"""
Integration tests for DCFR weighting on the MCCFR solver.
"""

import pytest

from src.core.actions.action_model import ActionModel
from tests.test_helpers import DummyCardAbstraction, build_test_solver, make_test_config


class TestDCFR:
    """Tests for DCFR (Discounted CFR) integration."""

    @pytest.mark.slow
    def test_dcfr_training_runs(self):
        """DCFR should complete training iterations without error."""
        ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        config = make_test_config(
            seed=42,
            iteration_weighting="dcfr",
            dcfr_alpha=1.5,
            dcfr_beta=0.0,
            dcfr_gamma=2.0,
        )
        solver, _storage = build_test_solver(config, card_abs)

        for _ in range(10):
            solver.train_iteration()

        assert solver.iteration == 10
        assert solver.num_infosets() > 0

    @pytest.mark.slow
    def test_dcfr_weighting_reflected_in_solver(self):
        """iteration_weighting='dcfr' should be reflected in solver config."""
        ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        config = make_test_config(seed=42, iteration_weighting="dcfr")
        solver, _storage = build_test_solver(config, card_abs)

        assert solver.config.solver.iteration_weighting == "dcfr"

    @pytest.mark.slow
    def test_linear_weighting_reflected_in_solver(self):
        """iteration_weighting='linear' should be reflected in solver config."""
        ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        config = make_test_config(seed=42, iteration_weighting="linear")
        solver, _storage = build_test_solver(config, card_abs)

        assert solver.config.solver.iteration_weighting == "linear"

    @pytest.mark.slow
    def test_dcfr_convergence(self):
        """DCFR should converge (strategies should update)."""
        ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        config = make_test_config(
            seed=42,
            iteration_weighting="dcfr",
            cfr_plus=True,
        )
        solver, storage = build_test_solver(config, card_abs)

        for _ in range(100):
            solver.train_iteration()

        updated_count = 0
        for infoset in storage.iter_infosets():
            if infoset.strategy_sum.sum() > 0:
                updated_count += 1

        assert updated_count > 0
