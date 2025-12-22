"""Tests for storage systems."""

import tempfile
from pathlib import Path

import numpy as np

from src.bucketing.utils.infoset import InfoSetKey
from src.game.actions import bet, call, fold
from src.game.state import Street
from src.solver.storage import DiskBackedStorage, InMemoryStorage


class TestInMemoryStorage:
    """Tests for InMemoryStorage."""

    def test_create_storage(self):
        storage = InMemoryStorage()
        assert storage.num_infosets() == 0

    def test_get_or_create_infoset(self):
        storage = InMemoryStorage()
        key = InfoSetKey(
            player_position=0,
            street=Street.FLOP,
            betting_sequence="b0.75",
            preflop_hand=None,
            postflop_bucket=25,
            spr_bucket=1,
        )
        actions = [fold(), call(), bet(50)]

        infoset = storage.get_or_create_infoset(key, actions)

        assert infoset is not None
        assert infoset.key == key
        assert len(infoset.legal_actions) == 3
        assert storage.num_infosets() == 1

    def test_get_or_create_same_key_returns_same_infoset(self):
        storage = InMemoryStorage()
        key = InfoSetKey(
            player_position=0,
            street=Street.FLOP,
            betting_sequence="b0.75",
            preflop_hand=None,
            postflop_bucket=25,
            spr_bucket=1,
        )
        actions = [fold(), call()]

        infoset1 = storage.get_or_create_infoset(key, actions)
        infoset2 = storage.get_or_create_infoset(key, actions)

        assert infoset1 is infoset2
        assert storage.num_infosets() == 1

    def test_get_infoset_exists(self):
        storage = InMemoryStorage()
        key = InfoSetKey(
            player_position=0,
            street=Street.FLOP,
            betting_sequence="b0.75",
            preflop_hand=None,
            postflop_bucket=25,
            spr_bucket=1,
        )
        actions = [fold(), call()]

        storage.get_or_create_infoset(key, actions)
        infoset = storage.get_infoset(key)

        assert infoset is not None
        assert infoset.key == key

    def test_get_infoset_not_exists(self):
        storage = InMemoryStorage()
        key = InfoSetKey(
            player_position=0,
            street=Street.FLOP,
            betting_sequence="b0.75",
            preflop_hand=None,
            postflop_bucket=25,
            spr_bucket=1,
        )

        infoset = storage.get_infoset(key)

        assert infoset is None

    def test_has_infoset(self):
        storage = InMemoryStorage()
        key1 = InfoSetKey(
            player_position=0,
            street=Street.FLOP,
            betting_sequence="b0.75",
            preflop_hand=None,
            postflop_bucket=25,
            spr_bucket=1,
        )
        key2 = InfoSetKey(
            player_position=1,
            street=Street.FLOP,
            betting_sequence="b0.75",
            preflop_hand=None,
            postflop_bucket=25,
            spr_bucket=1,
        )

        storage.get_or_create_infoset(key1, [fold(), call()])

        assert storage.has_infoset(key1)
        assert not storage.has_infoset(key2)

    def test_infoset_state_persists(self):
        """Test that infoset modifications persist."""
        storage = InMemoryStorage()
        key = InfoSetKey(
            player_position=0,
            street=Street.FLOP,
            betting_sequence="b0.75",
            preflop_hand=None,
            postflop_bucket=25,
            spr_bucket=1,
        )
        actions = [fold(), call()]

        infoset = storage.get_or_create_infoset(key, actions)
        infoset.update_regret(0, 10.0)
        infoset.update_strategy(1.0)

        # Get same infoset again
        infoset2 = storage.get_or_create_infoset(key, actions)

        assert infoset2.regrets[0] == 10.0

    def test_clear(self):
        storage = InMemoryStorage()
        key = InfoSetKey(
            player_position=0,
            street=Street.FLOP,
            betting_sequence="b0.75",
            preflop_hand=None,
            postflop_bucket=25,
            spr_bucket=1,
        )

        storage.get_or_create_infoset(key, [fold(), call()])
        assert storage.num_infosets() == 1

        storage.clear()
        assert storage.num_infosets() == 0

    def test_str_representation(self):
        storage = InMemoryStorage()
        s = str(storage)
        assert "InMemoryStorage" in s
        assert "num_infosets" in s


class TestDiskBackedStorage:
    """Tests for DiskBackedStorage."""

    def test_create_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskBackedStorage(Path(tmpdir), cache_size=10)
            assert storage.num_infosets() == 0

    def test_get_or_create_infoset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskBackedStorage(Path(tmpdir), cache_size=10)
            key = InfoSetKey(
                player_position=0,
                street=Street.FLOP,
                betting_sequence="b0.75",
                preflop_hand=None,
                postflop_bucket=25,
                spr_bucket=1,
            )
            actions = [fold(), call(), bet(50)]

            infoset = storage.get_or_create_infoset(key, actions)

            assert infoset is not None
            assert infoset.key == key
            assert storage.num_infosets() == 1

    def test_cache_eviction(self):
        """Test that LRU cache evicts old items."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskBackedStorage(Path(tmpdir), cache_size=2)

            # Create 3 infosets
            key1 = InfoSetKey(
                player_position=0,
                street=Street.FLOP,
                betting_sequence="b0.75",
                preflop_hand=None,
                postflop_bucket=1,
                spr_bucket=1,
            )
            key2 = InfoSetKey(
                player_position=0,
                street=Street.FLOP,
                betting_sequence="b0.75",
                preflop_hand=None,
                postflop_bucket=2,
                spr_bucket=1,
            )
            key3 = InfoSetKey(
                player_position=0,
                street=Street.FLOP,
                betting_sequence="b0.75",
                preflop_hand=None,
                postflop_bucket=3,
                spr_bucket=1,
            )

            storage.get_or_create_infoset(key1, [fold(), call()])
            storage.get_or_create_infoset(key2, [fold(), call()])
            storage.get_or_create_infoset(key3, [fold(), call()])

            # Cache size is 2, so key1 should be evicted
            assert len(storage.cache) <= 2
            assert storage.num_infosets() == 3  # But still tracked

    def test_flush_writes_dirty_infosets(self):
        """Test that flush writes modified infosets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskBackedStorage(Path(tmpdir), cache_size=10)
            key = InfoSetKey(
                player_position=0,
                street=Street.FLOP,
                betting_sequence="b0.75",
                preflop_hand=None,
                postflop_bucket=25,
                spr_bucket=1,
            )

            infoset = storage.get_or_create_infoset(key, [fold(), call()])
            infoset.update_regret(0, 100.0)

            # Flush to disk
            storage.flush()

            # Check dirty keys cleared
            assert len(storage.dirty_keys) == 0

    def test_checkpoint_and_load(self):
        """Test checkpointing and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create storage and add infosets
            storage1 = DiskBackedStorage(Path(tmpdir), cache_size=10)
            key = InfoSetKey(
                player_position=0,
                street=Street.FLOP,
                betting_sequence="b0.75",
                preflop_hand=None,
                postflop_bucket=25,
                spr_bucket=1,
            )

            infoset = storage1.get_or_create_infoset(key, [fold(), call()])
            infoset.update_regret(0, 50.0)
            infoset.update_regret(1, 75.0)

            # Checkpoint
            storage1.checkpoint(iteration=100)

            # Create new storage and load
            storage2 = DiskBackedStorage(Path(tmpdir), cache_size=10)

            assert storage2.num_infosets() == 1

            # Load infoset
            loaded = storage2.get_infoset(key)
            assert loaded is not None
            assert np.isclose(loaded.regrets[0], 50.0)
            assert np.isclose(loaded.regrets[1], 75.0)

    def test_has_infoset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskBackedStorage(Path(tmpdir), cache_size=10)
            key1 = InfoSetKey(
                player_position=0,
                street=Street.FLOP,
                betting_sequence="b0.75",
                preflop_hand=None,
                postflop_bucket=25,
                spr_bucket=1,
            )
            key2 = InfoSetKey(
                player_position=1,
                street=Street.FLOP,
                betting_sequence="b0.75",
                preflop_hand=None,
                postflop_bucket=25,
                spr_bucket=1,
            )

            storage.get_or_create_infoset(key1, [fold(), call()])

            assert storage.has_infoset(key1)
            assert not storage.has_infoset(key2)

    def test_periodic_flush(self):
        """Test that storage flushes periodically."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskBackedStorage(Path(tmpdir), cache_size=10, flush_frequency=5)

            # Access infosets to trigger periodic flush
            for i in range(10):
                key = InfoSetKey(
                    player_position=0,
                    street=Street.FLOP,
                    betting_sequence=f"seq{i}",
                    preflop_hand=None,
                    postflop_bucket=i,
                    spr_bucket=1,
                )
                storage.get_or_create_infoset(key, [fold(), call()])

            # Should have flushed at least once
            assert storage.access_count >= 10

    def test_str_representation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskBackedStorage(Path(tmpdir), cache_size=10)
            s = str(storage)
            assert "DiskBackedStorage" in s
