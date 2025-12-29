"""Tests for information sets."""

import numpy as np
import pytest

from src.bucketing.utils.infoset import InfoSet, InfoSetKey, create_infoset_key
from src.game.actions import bet, call, fold
from src.game.state import Street


class TestInfoSetKey:
    """Tests for InfoSetKey."""

    def test_create_infoset_key_preflop(self):
        """Test creating preflop InfoSetKey with hand string."""
        key = InfoSetKey(
            player_position=0,
            street=Street.PREFLOP,
            betting_sequence="r2.5",
            preflop_hand="AKs",
            postflop_bucket=None,
            spr_bucket=2,
        )
        assert key.player_position == 0
        assert key.street == Street.PREFLOP
        assert key.betting_sequence == "r2.5"
        assert key.preflop_hand == "AKs"
        assert key.postflop_bucket is None
        assert key.spr_bucket == 2

    def test_create_infoset_key_postflop(self):
        """Test creating postflop InfoSetKey with bucket."""
        key = InfoSetKey(
            player_position=0,
            street=Street.FLOP,
            betting_sequence="b0.75-c",
            preflop_hand=None,
            postflop_bucket=25,
            spr_bucket=1,
        )
        assert key.player_position == 0
        assert key.street == Street.FLOP
        assert key.betting_sequence == "b0.75-c"
        assert key.preflop_hand is None
        assert key.postflop_bucket == 25
        assert key.spr_bucket == 1

    def test_infoset_key_validation_preflop(self):
        """Test validation for preflop keys."""
        # Missing preflop_hand
        with pytest.raises(ValueError, match="preflop_hand must be set"):
            InfoSetKey(0, Street.PREFLOP, "r2.5", None, None, 2)

        # Has postflop_bucket on preflop
        with pytest.raises(ValueError, match="postflop_bucket must be None"):
            InfoSetKey(0, Street.PREFLOP, "r2.5", "AKs", 15, 2)

    def test_infoset_key_validation_postflop(self):
        """Test validation for postflop keys."""
        # Missing postflop_bucket
        with pytest.raises(ValueError, match="postflop_bucket must be set"):
            InfoSetKey(0, Street.FLOP, "b0.75", None, None, 1)

        # Has preflop_hand on postflop
        with pytest.raises(ValueError, match="preflop_hand must be None"):
            InfoSetKey(0, Street.FLOP, "b0.75", "AKs", 25, 1)

    def test_infoset_key_hashable(self):
        """Test that identical keys hash the same."""
        key1 = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        key2 = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        key3 = InfoSetKey(1, Street.FLOP, "b0.75", None, 25, 1)  # Different player

        # Same keys should hash the same
        assert hash(key1) == hash(key2)

        # Different keys should (usually) hash differently
        assert hash(key1) != hash(key3)

    def test_infoset_key_equality(self):
        """Test key equality."""
        key1 = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        key2 = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        key3 = InfoSetKey(1, Street.FLOP, "b0.75", None, 25, 1)

        assert key1 == key2
        assert key1 != key3

    def test_infoset_key_as_dict_key(self):
        """InfoSetKeys should work as dictionary keys."""
        key1 = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        key2 = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)

        data = {key1: "value1"}
        # Should retrieve using equivalent key
        assert data[key2] == "value1"

    def test_infoset_key_immutable(self):
        """InfoSetKeys should be immutable (frozen)."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        with pytest.raises(AttributeError):
            setattr(key, "postflop_bucket", 30)

    def test_create_infoset_key_helper_preflop(self):
        """Test helper function for preflop."""
        key = create_infoset_key(
            player=0,
            street=Street.PREFLOP,
            betting_sequence="r2.5",
            spr_bucket=2,
            preflop_hand="AKs",
        )
        assert isinstance(key, InfoSetKey)
        assert key.player_position == 0
        assert key.street == Street.PREFLOP
        assert key.preflop_hand == "AKs"

    def test_create_infoset_key_helper_postflop(self):
        """Test helper function for postflop."""
        key = create_infoset_key(
            player=0,
            street=Street.TURN,
            betting_sequence="x-b0.50",
            spr_bucket=2,
            postflop_bucket=40,
        )
        assert isinstance(key, InfoSetKey)
        assert key.player_position == 0
        assert key.street == Street.TURN
        assert key.postflop_bucket == 40

    def test_get_hand_repr(self):
        """Test get_hand_repr method."""
        # Preflop
        key_preflop = InfoSetKey(0, Street.PREFLOP, "r2.5", "AKs", None, 2)
        assert key_preflop.get_hand_repr() == "AKs"

        # Postflop
        key_postflop = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        assert key_postflop.get_hand_repr() == "B25"


class TestInfoSet:
    """Tests for InfoSet class."""

    def test_create_infoset(self):
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]

        infoset = InfoSet(key, actions)

        assert infoset.key == key
        assert infoset.legal_actions == actions
        assert infoset.num_actions == 3
        assert len(infoset.regrets) == 3
        assert len(infoset.strategy_sum) == 3

    def test_initial_strategy_uniform(self):
        """Initial strategy should be uniform."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        strategy = infoset.get_strategy()

        # All zeros -> uniform distribution
        assert len(strategy) == 3
        assert np.allclose(strategy, [1 / 3, 1 / 3, 1 / 3])
        assert np.isclose(strategy.sum(), 1.0)

    def test_regret_matching_positive_regrets(self):
        """Regret matching should prefer actions with higher regret."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        # Set some regrets
        infoset.regrets = np.array([0.0, 100.0, 200.0], dtype=np.float32)

        strategy = infoset.get_strategy()

        # Should weight toward higher regrets
        assert strategy[2] > strategy[1] > strategy[0]
        assert strategy[0] == 0.0  # Zero regret
        assert np.isclose(strategy.sum(), 1.0)

    def test_regret_matching_negative_regrets(self):
        """Negative regrets should be floored at 0."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        # Set all negative regrets
        infoset.regrets = np.array([-10.0, -20.0, -5.0], dtype=np.float32)

        strategy = infoset.get_strategy()

        # All negative -> uniform
        assert np.allclose(strategy, [1 / 3, 1 / 3, 1 / 3])

    def test_update_regret(self):
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        infoset.update_regret(0, 10.0)
        infoset.update_regret(1, -5.0)
        infoset.update_regret(2, 20.0)

        assert infoset.regrets[0] == 10.0
        assert infoset.regrets[1] == -5.0
        assert infoset.regrets[2] == 20.0

    def test_update_regret_cumulative(self):
        """Regrets should accumulate."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call()]
        infoset = InfoSet(key, actions)

        infoset.update_regret(0, 10.0)
        infoset.update_regret(0, 5.0)
        infoset.update_regret(0, -3.0)

        assert infoset.regrets[0] == 12.0  # 10 + 5 - 3

    def test_update_strategy(self):
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        # Set regrets to create non-uniform strategy
        infoset.regrets = np.array([0.0, 100.0, 200.0], dtype=np.float32)

        # Update strategy with reach prob 1.0
        infoset.update_strategy(1.0)

        assert infoset.reach_count == 1
        # strategy_sum should equal current strategy
        strategy = infoset.get_strategy()
        assert np.allclose(infoset.strategy_sum, strategy)

    def test_average_strategy(self):
        """Average strategy should track cumulative play."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        # Iteration 1: all regrets zero (uniform)
        infoset.update_strategy(1.0)

        # Iteration 2: favor action 2
        infoset.regrets = np.array([0.0, 0.0, 100.0], dtype=np.float32)
        infoset.update_strategy(1.0)

        avg = infoset.get_average_strategy()

        # Average should be between uniform and full action 2
        assert 0 < avg[2] < 1.0
        assert avg[2] > avg[0]
        assert np.isclose(avg.sum(), 1.0)

    def test_reset_regrets(self):
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call()]
        infoset = InfoSet(key, actions)

        infoset.regrets = np.array([10.0, 20.0], dtype=np.float32)
        infoset.reset_regrets()

        assert np.allclose(infoset.regrets, [0.0, 0.0])

    def test_reset_strategy_sum(self):
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call()]
        infoset = InfoSet(key, actions)

        infoset.strategy_sum = np.array([10.0, 20.0], dtype=np.float32)
        infoset.reset_strategy_sum()

        assert np.allclose(infoset.strategy_sum, [0.0, 0.0])

    def test_prune_small_values(self):
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        infoset.regrets = np.array([1e-10, 10.0, 1e-11], dtype=np.float32)
        infoset.strategy_sum = np.array([1e-12, 5.0, 1e-10], dtype=np.float32)

        infoset.prune(threshold=1e-9)

        # Small values should be pruned to 0
        assert infoset.regrets[0] == 0.0
        assert infoset.regrets[1] == 10.0
        assert infoset.regrets[2] == 0.0
        assert infoset.strategy_sum[0] == 0.0
        assert infoset.strategy_sum[2] == 0.0

    def test_update_regret_invalid_index(self):
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call()]
        infoset = InfoSet(key, actions)

        with pytest.raises(ValueError, match="Invalid action index"):
            infoset.update_regret(5, 10.0)

    def test_str_representation(self):
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call()]
        infoset = InfoSet(key, actions)

        s = str(infoset)
        assert "InfoSet" in s
        assert "FOLD" in s
        assert "CALL" in s
        assert "Strategy" in s
