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
    """Regret-based pruning is unimplemented on shared storage and rejected at load.

    The per-action pruned mask lives only on the per-visit ``InfoSet`` view and has
    no shared-array backing, so a mark never survives to the next visit. The knob is
    refused at config load rather than silently doing nothing (see ``SolverConfig``).
    """

    @pytest.mark.parametrize("sampling_method", ["external", "outcome"])
    def test_pruning_rejected_at_config(self, sampling_method):
        with pytest.raises(ValidationError, match="enable_pruning=True is not supported"):
            make_test_config(seed=42, enable_pruning=True, sampling_method=sampling_method)
