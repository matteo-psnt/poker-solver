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

    # DCFR and pruning tests

    def test_update_regret_with_dcfr(self):
        """Test that DCFR applies discount to cumulative regrets (Brown & Sandholm 2019)."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        # Set initial cumulative regret
        infoset.regrets[0] = 100.0

        # Update with DCFR enabled at iteration 100
        # Should: (1) discount cumulative regret, (2) add new regret
        infoset.update_regret(
            0,
            10.0,
            iteration=100,
            enable_dcfr=True,
            dcfr_alpha=1.5,
            dcfr_beta=0.0,
        )

        # Discount factor for positive regret: 100^1.5 / (100^1.5 + 1)
        t_exp = 100.0**1.5
        discount_factor = t_exp / (t_exp + 1.0)
        expected_regret = 100.0 * discount_factor + 10.0
        assert abs(infoset.regrets[0] - expected_regret) < 1e-6

    def test_update_regret_dcfr_negative_regret(self):
        """Test DCFR with negative cumulative regret using beta parameter."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        # Set initial negative cumulative regret
        infoset.regrets[0] = -50.0

        # Update with DCFR enabled at iteration 100
        # With beta=0, negative cumulative regrets get discount factor 1.0 (no discount)
        infoset.update_regret(
            0,
            -10.0,
            iteration=100,
            enable_dcfr=True,
            dcfr_alpha=1.5,
            dcfr_beta=0.0,
        )

        # With beta=0, cumulative negative regret is not discounted
        # Result: -50.0 * 1.0 + (-10.0) = -60.0
        assert abs(infoset.regrets[0] - (-60.0)) < 1e-6

    def test_dcfr_cumulative_discounting_over_iterations(self):
        """Test that DCFR correctly discounts cumulative regrets over multiple iterations."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        # Simulate multiple iterations with constant regret updates
        # This demonstrates the exponential discounting effect
        constant_regret = 10.0
        alpha = 1.5

        for iteration in range(1, 6):  # iterations 1-5
            infoset.update_regret(
                0,
                constant_regret,
                iteration=iteration,
                enable_dcfr=True,
                dcfr_alpha=alpha,
                dcfr_beta=0.0,
            )

        # Manually compute expected regret after 5 iterations
        # Iteration 1: 0 * (discount) + 10 = 10
        # Iteration 2: 10 * (2^1.5/(2^1.5+1)) + 10 ≈ 10 * 0.738 + 10 = 17.38
        # ... and so on
        # Verify the final regret is reasonable
        assert infoset.regrets[0] > 0
        assert infoset.regrets[0] < 100  # Should not grow unbounded

    def test_dcfr_vs_vanilla_cfr_discounting(self):
        """Compare DCFR discounting to vanilla CFR (no discounting)."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]

        # DCFR infoset
        infoset_dcfr = InfoSet(key, actions)
        # Vanilla CFR infoset
        infoset_vanilla = InfoSet(key, actions)

        # Both start with same cumulative regret
        infoset_dcfr.regrets[0] = 100.0
        infoset_vanilla.regrets[0] = 100.0

        # Update both with same new regret
        new_regret = 20.0
        iteration = 100

        infoset_dcfr.update_regret(
            0, new_regret, iteration=iteration, enable_dcfr=True, dcfr_alpha=1.5
        )
        infoset_vanilla.update_regret(0, new_regret, iteration=iteration, enable_dcfr=False)

        # DCFR should have smaller cumulative regret due to discounting old regret
        assert infoset_dcfr.regrets[0] < infoset_vanilla.regrets[0]
        # Vanilla: 100 + 20 = 120
        assert abs(infoset_vanilla.regrets[0] - 120.0) < 1e-6

    def test_pruning_initialization(self):
        """Pruned array should be initialized to False."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        assert len(infoset.pruned) == 3
        assert not any(infoset.pruned)

    def test_is_pruned(self):
        """Test is_pruned method."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        # Initially not pruned
        assert not infoset.is_pruned(0)
        assert not infoset.is_pruned(1)
        assert not infoset.is_pruned(2)

        # Manually set pruned
        infoset.pruned[0] = True
        assert infoset.is_pruned(0)
        assert not infoset.is_pruned(1)

    def test_is_pruned_invalid_index(self):
        """Test is_pruned with invalid index."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call()]
        infoset = InfoSet(key, actions)

        with pytest.raises(ValueError, match="Invalid action index"):
            infoset.is_pruned(5)

    def test_update_pruning_negative_regret(self):
        """Actions with sufficiently negative regret should be pruned."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        # Set regrets: one very negative, two positive
        infoset.regrets = np.array([-500.0, 10.0, 20.0])

        # Update pruning (after start iteration, but not at reactivation point)
        infoset.update_pruning(
            iteration=201,
            pruning_threshold=300.0,
            prune_start_iteration=100,
            prune_reactivate_frequency=100,
        )

        # Action 0 should be pruned (regret -500 < -300)
        assert infoset.is_pruned(0)
        assert not infoset.is_pruned(1)
        assert not infoset.is_pruned(2)

    def test_update_pruning_reactivation_on_positive_regret(self):
        """Pruned actions should reactivate if regret becomes positive."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        # Start with negative regret and prune
        infoset.regrets = np.array([-500.0, 10.0, 20.0])
        infoset.update_pruning(
            iteration=201,
            pruning_threshold=300.0,
            prune_start_iteration=100,
            prune_reactivate_frequency=100,
        )
        assert infoset.is_pruned(0)

        # Update regret to positive
        infoset.regrets[0] = 5.0
        infoset.update_pruning(
            iteration=202,
            pruning_threshold=300.0,
            prune_start_iteration=100,
            prune_reactivate_frequency=100,
        )

        # Should be reactivated
        assert not infoset.is_pruned(0)

    def test_update_pruning_periodic_reactivation(self):
        """All actions should be reactivated at reactivation frequency."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        # Set all regrets negative and prune
        infoset.regrets = np.array([-500.0, -400.0, -600.0])
        infoset.update_pruning(
            iteration=199,
            pruning_threshold=300.0,
            prune_start_iteration=100,
            prune_reactivate_frequency=100,
        )

        # At least one should be unpruned (safety mechanism)
        assert sum(infoset.pruned) <= 2

        # At reactivation iteration, all should be unpruned
        infoset.update_pruning(
            iteration=200,
            pruning_threshold=300.0,
            prune_start_iteration=100,
            prune_reactivate_frequency=100,
        )
        assert not any(infoset.pruned)

    def test_update_pruning_disabled_early_iterations(self):
        """Pruning should be disabled before prune_start_iteration."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        # Set very negative regrets
        infoset.regrets = np.array([-500.0, -400.0, -600.0])

        # Update pruning before start iteration
        infoset.update_pruning(
            iteration=50,  # Before start iteration
            pruning_threshold=300.0,
            prune_start_iteration=100,
            prune_reactivate_frequency=100,
        )

        # No actions should be pruned
        assert not any(infoset.pruned)

    def test_update_pruning_safety_mechanism(self):
        """At least one action should always remain unpruned."""
        key = InfoSetKey(0, Street.FLOP, "b0.75", None, 25, 1)
        actions = [fold(), call(), bet(50)]
        infoset = InfoSet(key, actions)

        # All regrets deeply negative
        infoset.regrets = np.array([-500.0, -600.0, -400.0])

        infoset.update_pruning(
            iteration=200,
            pruning_threshold=300.0,
            prune_start_iteration=100,
            prune_reactivate_frequency=100,
        )

        # At least one action must be unpruned (safety mechanism)
        assert sum(infoset.pruned) < 3

        # The action with highest (least negative) regret should be unpruned
        # That's action 2 with regret -400
        assert not infoset.is_pruned(2)
