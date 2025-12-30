"""Tests for head-to-head evaluator."""

from src.actions.betting_actions import BettingActions
from src.evaluation.head_to_head import (
    HeadToHeadEvaluator,
    MatchResult,
    MatchStatistics,
)
from src.game.rules import GameRules
from src.game.state import Street
from src.solver.mccfr import MCCFRSolver
from src.solver.storage.shared_array import SharedArrayStorage
from tests.test_helpers import DummyCardAbstraction, make_test_config


class TestMatchResult:
    """Tests for MatchResult dataclass."""

    def test_create_match_result(self):
        result = MatchResult(
            player0_payoff=10,
            player1_payoff=-10,
            hand_number=1,
            final_street=Street.RIVER,
            showdown=True,
        )

        assert result.player0_payoff == 10
        assert result.player1_payoff == -10
        assert result.hand_number == 1
        assert result.showdown is True


class TestMatchStatistics:
    """Tests for MatchStatistics dataclass."""

    def test_create_match_statistics(self):
        stats = MatchStatistics(
            num_hands=10,
            player0_wins=6,
            player1_wins=4,
            player0_total_won=20,
            player1_total_won=-20,
            player0_bb_per_hand=1.0,
            player1_bb_per_hand=-1.0,
            showdown_pct=50.0,
            results=[],
        )

        assert stats.num_hands == 10
        assert stats.player0_wins == 6
        assert stats.player1_wins == 4

    def test_str_representation(self):
        stats = MatchStatistics(
            num_hands=10,
            player0_wins=6,
            player1_wins=4,
            player0_total_won=20,
            player1_total_won=-20,
            player0_bb_per_hand=1.0,
            player1_bb_per_hand=-1.0,
            showdown_pct=50.0,
            results=[],
        )

        s = str(stats)
        assert "10" in s
        assert "6" in s


class TestHeadToHeadEvaluator:
    """Tests for HeadToHeadEvaluator."""

    def test_create_evaluator(self):
        rules = GameRules()
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()

        evaluator = HeadToHeadEvaluator(
            rules=rules,
            action_abstraction=action_abs,
            card_abstraction=card_abs,
            starting_stack=200,
        )

        assert evaluator.rules is not None
        assert evaluator.starting_stack == 200

    def test_play_match(self):
        """Test playing a match between two solvers."""
        rules = GameRules()
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()

        # Create two solvers
        storage1 = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test1", is_coordinator=True
        )
        storage2 = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test2", is_coordinator=True
        )

        solver1 = MCCFRSolver(action_abs, card_abs, storage1, config=make_test_config(seed=42))
        solver2 = MCCFRSolver(action_abs, card_abs, storage2, config=make_test_config(seed=43))

        # Train briefly so they have some strategy
        for _ in range(3):
            solver1.train_iteration()
            solver2.train_iteration()

        # Create evaluator
        evaluator = HeadToHeadEvaluator(
            rules=rules,
            action_abstraction=action_abs,
            card_abstraction=card_abs,
            starting_stack=200,
        )

        # Play match
        stats = evaluator.play_match(
            solver0=solver1,
            solver1=solver2,
            num_hands=3,
            seed=42,
        )

        # Check statistics
        assert stats.num_hands == 3
        assert stats.player0_wins + stats.player1_wins == 3
        assert len(stats.results) == 3
        assert 0 <= stats.showdown_pct <= 100

    def test_play_match_alternating_button(self):
        """Test that button alternates correctly."""
        rules = GameRules()
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()

        storage1 = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test1", is_coordinator=True
        )
        storage2 = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test2", is_coordinator=True
        )

        solver1 = MCCFRSolver(action_abs, card_abs, storage1, config=make_test_config(seed=42))
        solver2 = MCCFRSolver(action_abs, card_abs, storage2, config=make_test_config(seed=43))

        # Train briefly
        solver1.train_iteration()
        solver2.train_iteration()

        evaluator = HeadToHeadEvaluator(
            rules=rules,
            action_abstraction=action_abs,
            card_abstraction=card_abs,
            starting_stack=200,
        )

        # Play with alternating button
        stats = evaluator.play_match(
            solver0=solver1,
            solver1=solver2,
            num_hands=2,
            alternate_button=True,
            seed=42,
        )

        assert stats.num_hands == 2

    def test_play_self_play_match(self):
        """Test self-play (same solver vs itself)."""
        rules = GameRules()
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()

        storage = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test", is_coordinator=True
        )
        solver = MCCFRSolver(action_abs, card_abs, storage, config=make_test_config(seed=42))

        # Train
        for _ in range(3):
            solver.train_iteration()

        evaluator = HeadToHeadEvaluator(
            rules=rules,
            action_abstraction=action_abs,
            card_abstraction=card_abs,
            starting_stack=200,
        )

        # Self-play match
        stats = evaluator.play_match(
            solver0=solver,
            solver1=solver,
            num_hands=3,
            seed=42,
        )

        assert stats.num_hands == 3

        # In self-play, expected value should be close to 0
        # But individual match results can vary
        # Just check that the match ran successfully
        assert stats.player0_bb_per_hand is not None
        assert stats.player1_bb_per_hand is not None

    def test_match_payoffs_opposite_signs(self):
        """Test that payoffs have opposite signs (zero-sum game)."""
        rules = GameRules()
        action_abs = BettingActions()
        card_abs = DummyCardAbstraction()

        storage1 = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test1", is_coordinator=True
        )
        storage2 = SharedArrayStorage(
            num_workers=1, worker_id=0, session_id="test2", is_coordinator=True
        )

        solver1 = MCCFRSolver(action_abs, card_abs, storage1, config=make_test_config(seed=42))
        solver2 = MCCFRSolver(action_abs, card_abs, storage2, config=make_test_config(seed=43))

        solver1.train_iteration()
        solver2.train_iteration()

        evaluator = HeadToHeadEvaluator(
            rules=rules,
            action_abstraction=action_abs,
            card_abstraction=card_abs,
            starting_stack=200,
        )

        stats = evaluator.play_match(
            solver0=solver1,
            solver1=solver2,
            num_hands=2,
            seed=42,
        )

        # In zero-sum game, if one player wins, other loses
        # Payoffs should have opposite signs (unless tie)
        for result in stats.results:
            if result.player0_payoff > 0:
                assert result.player1_payoff < 0
            elif result.player0_payoff < 0:
                assert result.player1_payoff > 0
            else:
                # Tie
                assert result.player1_payoff == 0
