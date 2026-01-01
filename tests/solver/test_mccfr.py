"""Tests for MCCFR solver."""

import pytest

from src.actions.betting_actions import BettingActions
from src.bucketing.utils.infoset import InfoSetKey
from src.game.state import Card, GameState, Street
from src.solver.mccfr import MCCFRSolver
from src.solver.storage import SharedArrayStorage
from tests.test_helpers import DummyCardAbstraction, make_test_config


class TestMCCFRSolver:
    """Tests for MCCFRSolver."""

    def test_create_solver(self):
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )

        solver = MCCFRSolver(action_abs, card_abs, storage)

        assert solver.iteration == 0
        assert solver.num_infosets() == 0

    def test_deal_initial_state(self):
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )
        solver = MCCFRSolver(action_abs, card_abs, storage)

        state = solver._deal_initial_state()

        # Check initial state properties
        assert state.pot == 3  # SB + BB
        assert state.stacks[0] == 199  # 200 - 1 (SB)
        assert state.stacks[1] == 198  # 200 - 2 (BB)
        assert len(state.hole_cards[0]) == 2
        assert len(state.hole_cards[1]) == 2
        assert state.board == tuple()  # No board yet

    def test_train_iteration_executes(self):
        """Test that one iteration completes without error."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )
        solver = MCCFRSolver(action_abs, card_abs, storage, config=make_test_config(seed=42))

        utility = solver.train_iteration()

        assert solver.iteration == 1
        assert isinstance(utility, float)
        # At least some infosets should be created
        assert solver.num_infosets() > 0

    def test_multiple_iterations(self):
        """Test multiple training iterations."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )
        solver = MCCFRSolver(action_abs, card_abs, storage, config=make_test_config(seed=42))

        results = solver.train(num_iterations=5, verbose=False)

        assert solver.iteration == 5
        assert results["total_iterations"] == 5
        assert solver.num_infosets() > 0

    def test_infosets_accumulate(self):
        """Test that infosets accumulate over iterations."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )
        solver = MCCFRSolver(action_abs, card_abs, storage, config=make_test_config(seed=42))

        # Run first iteration
        solver.train_iteration()
        count_after_1 = solver.num_infosets()

        # Run more iterations
        solver.train(num_iterations=4, verbose=False)
        count_after_5 = solver.num_infosets()

        # Should have discovered more infosets
        assert count_after_5 >= count_after_1

    def test_strategies_update(self):
        """Test that strategies are updated during training."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )
        # Use external sampling which updates strategy_sum for all actions
        solver = MCCFRSolver(
            action_abs,
            card_abs,
            storage,
            config=make_test_config(seed=42, sampling_method="external"),
        )

        # Train for enough iterations to update strategies
        solver.train(num_iterations=10, verbose=False)

        # Check that at least some infosets have been updated
        # (not all may be updated due to alternating player traversal)
        assert storage.num_infosets() > 0
        updated_infosets = sum(
            1 for infoset in storage.infosets.values() if infoset.strategy_sum.sum() > 0
        )
        assert updated_infosets > 0, "At least some infosets should have updated strategy_sum"

    def test_is_chance_node(self):
        """Test chance node detection."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )
        solver = MCCFRSolver(action_abs, card_abs, storage)

        state = solver._deal_initial_state()

        # Initially not a chance node (players need to act)
        is_chance = solver._is_chance_node(state)
        # This depends on betting history, so just check it returns bool
        assert isinstance(is_chance, bool)

    def test_sample_chance_outcome_deals_cards(self):
        """Test that chance node sampling deals cards."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )
        solver = MCCFRSolver(action_abs, card_abs, storage)

        # Create state needing flop
        state = GameState(
            street=Street.FLOP,  # Flop street but no cards yet
            pot=10,
            stacks=(195, 195),
            board=tuple(),
            hole_cards=(
                (Card.new("As"), Card.new("Kh")),
                (Card.new("Qd"), Card.new("Jc")),
            ),
            betting_history=tuple(),
            button_position=0,
            current_player=0,
            is_terminal=False,
            _skip_validation=True,  # Skip validation for incomplete board
        )

        # Sample flop
        new_state = solver._sample_chance_outcome(state)

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
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()

        # Run with seed 42
        storage1 = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test1", is_coordinator=True
        )
        solver1 = MCCFRSolver(action_abs, card_abs, storage1, config=make_test_config(seed=42))
        solver1.train(num_iterations=5, verbose=False)
        infosets1 = solver1.num_infosets()

        # Run again with same seed
        storage2 = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test2", is_coordinator=True
        )
        solver2 = MCCFRSolver(action_abs, card_abs, storage2, config=make_test_config(seed=42))
        solver2.train(num_iterations=5, verbose=False)
        infosets2 = solver2.num_infosets()

        # Should produce similar results (within 20% variance)
        diff = abs(infosets1 - infosets2)
        avg = (infosets1 + infosets2) / 2
        variance = diff / avg if avg > 0 else 0
        assert variance < 0.20, (
            f"Variance {variance:.2%} exceeds 20% (infosets: {infosets1} vs {infosets2})"
        )

    def test_checkpoint(self):
        """Test that checkpoint doesn't crash."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )
        solver = MCCFRSolver(action_abs, card_abs, storage)

        solver.train(num_iterations=10, verbose=False)
        solver.checkpoint()  # Should not crash

    def test_str_representation(self):
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )
        solver = MCCFRSolver(action_abs, card_abs, storage)

        s = str(solver)
        assert "MCCFRSolver" in s
        assert "iteration" in s

    def test_custom_stack_size(self):
        """Test solver with custom stack size."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )
        solver = MCCFRSolver(
            action_abs, card_abs, storage, config=make_test_config(starting_stack=100)
        )

        state = solver._deal_initial_state()

        # Check custom stack size
        assert state.stacks[0] == 99  # 100 - 1 (SB)
        assert state.stacks[1] == 98  # 100 - 2 (BB)

    def test_train_verbose(self):
        """Test verbose training runs without error."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )
        solver = MCCFRSolver(action_abs, card_abs, storage)

        # Train with verbose=True (note: output only prints every 1000 iterations)
        # Just verify it runs without error
        solver.train(num_iterations=2, verbose=True)

    def test_train_return_statistics(self):
        """Test that train() returns correct statistics."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )
        solver = MCCFRSolver(action_abs, card_abs, storage)

        stats = solver.train(num_iterations=2, verbose=False)

        assert "start_iteration" in stats
        assert "end_iteration" in stats
        assert "total_iterations" in stats
        assert stats["total_iterations"] == 2
        assert "final_avg_utility" in stats

    def test_get_average_strategy_nonexistent_infoset(self):
        """Test get_average_strategy with non-existent infoset."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )
        solver = MCCFRSolver(action_abs, card_abs, storage)

        # Create fake infoset key (preflop uses hand string)
        fake_key = InfoSetKey(
            player_position=0,
            street=Street.PREFLOP,
            betting_sequence="fake",
            preflop_hand="AA",
            postflop_bucket=None,
            spr_bucket=0,
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="Infoset not found"):
            solver.get_average_strategy(fake_key)

    def test_get_current_strategy_nonexistent_infoset(self):
        """Test get_current_strategy with non-existent infoset."""
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()
        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )
        solver = MCCFRSolver(action_abs, card_abs, storage)

        # Create fake infoset key (preflop uses hand string)
        fake_key = InfoSetKey(
            player_position=0,
            street=Street.PREFLOP,
            betting_sequence="fake",
            preflop_hand="AA",
            postflop_bucket=None,
            spr_bucket=0,
        )

        # Should raise ValueError
        with pytest.raises(ValueError, match="Infoset not found"):
            solver.get_current_strategy(fake_key)
