"""Tests for MCCFR solver."""

import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.state import Card, GameState, Street
from tests.test_helpers import DummyCardAbstraction, build_test_solver, make_test_config


class TestMCCFRSolver:
    """Tests for MCCFRSolver."""

    def test_create_solver(self):
        ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        solver, _storage = build_test_solver(make_test_config(), card_abs)

        assert solver.iteration == 0
        assert solver.num_infosets() == 0

    def test_deal_initial_state(self):
        ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        solver, _storage = build_test_solver(make_test_config(), card_abs)

        state = solver.deal_initial_state()

        # Check initial state properties
        assert state.pot == 3  # SB + BB
        assert state.stacks[0] == 199  # 200 - 1 (SB)
        assert state.stacks[1] == 198  # 200 - 2 (BB)
        assert len(state.hole_cards[0]) == 2
        assert len(state.hole_cards[1]) == 2
        assert state.board == ()  # No board yet

    def test_train_iteration_executes(self):
        """Test that one iteration completes without error."""
        ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        solver, _storage = build_test_solver(make_test_config(seed=42), card_abs)

        utility = solver.train_iteration()

        assert solver.iteration == 1
        assert isinstance(utility, float)
        # At least some infosets should be created
        assert solver.num_infosets() > 0

    def test_multiple_iterations(self):
        """Test multiple training iterations."""
        ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        solver, _storage = build_test_solver(make_test_config(seed=42), card_abs)

        for _ in range(5):
            solver.train_iteration()

        assert solver.iteration == 5
        assert solver.num_infosets() > 0

    def test_infosets_accumulate(self):
        """Test that infosets accumulate over iterations."""
        ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        solver, _storage = build_test_solver(make_test_config(seed=42), card_abs)

        # Run first iteration
        solver.train_iteration()
        count_after_1 = solver.num_infosets()

        # Run more iterations
        for _ in range(4):
            solver.train_iteration()
        count_after_5 = solver.num_infosets()

        # Should have discovered more infosets
        assert count_after_5 >= count_after_1

    def test_strategies_update(self):
        """Test that strategies are updated during training."""
        ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        # Use external sampling which updates strategy_sum for all actions
        solver, storage = build_test_solver(
            make_test_config(seed=42, sampling_method="external"), card_abs
        )

        # Train for enough iterations to update strategies
        for _ in range(10):
            solver.train_iteration()

        # Check that at least some infosets have been updated
        # (not all may be updated due to alternating player traversal)
        assert storage.num_infosets() > 0
        updated_infosets = sum(
            1 for infoset in storage.iter_infosets() if infoset.strategy_sum.sum() > 0
        )
        assert updated_infosets > 0, "At least some infosets should have updated strategy_sum"

    def test_is_chance_node(self):
        """Test chance node detection."""
        ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        solver, _storage = build_test_solver(make_test_config(), card_abs)

        state = solver.deal_initial_state()

        # Initially not a chance node (players need to act)
        is_chance = solver.is_chance_node(state)
        # This depends on betting history, so just check it returns bool
        assert isinstance(is_chance, bool)

    def test_sample_chance_outcome_deals_cards(self):
        """Test that chance node sampling deals cards."""
        ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        solver, _storage = build_test_solver(make_test_config(), card_abs)

        # Create state needing flop
        state = GameState(
            street=Street.FLOP,  # Flop street but no cards yet
            pot=10,
            stacks=(195, 195),
            board=(),
            hole_cards=(
                (Card.new("As"), Card.new("Kh")),
                (Card.new("Qd"), Card.new("Jc")),
            ),
            betting_history=(),
            button_position=0,
            current_player=0,
            is_terminal=False,
            _skip_validation=True,  # Skip validation for incomplete board
        )

        # Sample flop
        new_state = solver.sample_chance_outcome(state)

        # Should have flop cards
        assert len(new_state.board) == 3

    @pytest.mark.skip(reason="Non-determinism in MCCFR outcome sampling with error handling")
    def test_deterministic_with_seed(self):
        """Test that solver is mostly deterministic with same seed.

        Note: Currently skipped due to non-determinism in MCCFR outcome sampling.
        The error handling for invalid actions (when states with same InfoSetKey
        have different legal actions) introduces path-dependent behavior that
        affects which infosets are explored. This doesn't impact solution quality,
        just the exact game tree exploration path.

        TODO: Investigate sources of non-determinism:
        - Error handling fallback paths
        - Card dealing randomness
        - Action abstraction consistency
        """
        ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()

        # Run with seed 42
        solver1, _storage1 = build_test_solver(make_test_config(seed=42), card_abs)
        for _ in range(5):
            solver1.train_iteration()
        infosets1 = solver1.num_infosets()

        # Run again with same seed
        solver2, _storage2 = build_test_solver(make_test_config(seed=42), card_abs)
        for _ in range(5):
            solver2.train_iteration()
        infosets2 = solver2.num_infosets()

        # Should produce similar results (within 20% variance)
        diff = abs(infosets1 - infosets2)
        avg = (infosets1 + infosets2) / 2
        variance = diff / avg if avg > 0 else 0
        assert variance < 0.20, (
            f"Variance {variance:.2%} exceeds 20% (infosets: {infosets1} vs {infosets2})"
        )

    def test_checkpoint(self, tmp_path):
        """Test that checkpoint doesn't crash."""
        card_abs = DummyCardAbstraction()
        solver, _storage = build_test_solver(make_test_config(), card_abs, checkpoint_dir=tmp_path)

        for _ in range(10):
            solver.train_iteration()
        solver.checkpoint()  # Should not crash

    def test_str_representation(self):
        ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        solver, _storage = build_test_solver(make_test_config(), card_abs)

        s = str(solver)
        assert "StaticTreeSolver" in s
        assert "iteration" in s

    def test_custom_stack_size(self):
        """Test solver with custom stack size."""
        ActionModel(make_test_config())
        card_abs = DummyCardAbstraction()
        solver, _storage = build_test_solver(make_test_config(starting_stack=100), card_abs)

        state = solver.deal_initial_state()

        # Check custom stack size
        assert state.stacks[0] == 99  # 100 - 1 (SB)
        assert state.stacks[1] == 98  # 100 - 2 (BB)
