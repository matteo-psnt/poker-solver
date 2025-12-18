"""Tests for disk-backed storage."""

import tempfile
from pathlib import Path

import numpy as np

from src.abstraction.infoset import InfoSetKey
from src.game.actions import Action, ActionType
from src.game.state import Street
from src.solver.storage import DiskBackedStorage


class TestDiskBackedStorage:
    """Tests for DiskBackedStorage."""

    def test_create_storage(self):
        """Test creating disk-backed storage."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskBackedStorage(Path(tmpdir))

            assert storage.checkpoint_dir == Path(tmpdir)
            assert storage.cache_size == 100000
            assert storage.num_infosets() == 0

    def test_save_and_load_infoset(self):
        """Test saving and loading a single infoset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskBackedStorage(Path(tmpdir), cache_size=2)

            # Create infoset (preflop uses hand string)
            key = InfoSetKey(
                player_position=0,
                street=Street.PREFLOP,
                betting_sequence="",
                preflop_hand="AKs",
                postflop_bucket=None,
                spr_bucket=1,
            )

            actions = [
                Action(ActionType.FOLD, 0),
                Action(ActionType.CALL, 0),
                Action(ActionType.RAISE, 5),
            ]

            # Get or create
            infoset = storage.get_or_create_infoset(key, actions)
            assert infoset is not None
            assert len(infoset.legal_actions) == 3

            # Modify strategy
            infoset.regrets = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            infoset.strategy_sum = np.array([0.5, 0.3, 0.2], dtype=np.float32)
            infoset.reach_count = 42

            # Mark as dirty since we modified it
            storage.mark_dirty(key)

            # Force flush
            storage.flush()

            # Create new storage instance (simulates restart)
            storage2 = DiskBackedStorage(Path(tmpdir), cache_size=2)

            # Debug: check if key mappings loaded
            print(f"Storage2 has {storage2.num_infosets()} infosets")
            print(f"Key in key_to_id: {key in storage2.key_to_id}")

            # Load infoset
            infoset2 = storage2.get_infoset(key)

            assert infoset2 is not None, (
                f"Failed to load infoset. Storage has {storage2.num_infosets()} infosets, "
                f"key in mapping: {key in storage2.key_to_id}"
            )
            assert np.allclose(infoset2.regrets, [1.0, 2.0, 3.0])
            assert np.allclose(infoset2.strategy_sum, [0.5, 0.3, 0.2])

    def test_cache_eviction(self):
        """Test LRU cache eviction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Small cache size
            storage = DiskBackedStorage(Path(tmpdir), cache_size=2)

            # Create 3 infosets (more than cache size)
            hands = ["AA", "KK", "QQ"]
            keys = []
            for i in range(3):
                key = InfoSetKey(
                    player_position=0,
                    street=Street.PREFLOP,
                    betting_sequence=f"seq{i}",
                    preflop_hand=hands[i],
                    postflop_bucket=None,
                    spr_bucket=0,
                )
                keys.append(key)

                actions = [Action(ActionType.FOLD, 0)]
                infoset = storage.get_or_create_infoset(key, actions)
                infoset.regrets = np.array([float(i)], dtype=np.float32)

            # First infoset should have been evicted from cache
            assert len(storage.cache) <= 2

            # But should still be loadable from disk
            infoset0 = storage.get_infoset(keys[0])
            assert infoset0 is not None
            assert np.allclose(infoset0.regrets, [0.0])

    def test_checkpoint_and_reload(self):
        """Test checkpoint and reload functionality."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskBackedStorage(Path(tmpdir))

            # Create multiple infosets
            hands = ["AA", "KK", "QQ", "JJ", "TT", "99", "88", "77", "66", "55"]
            for i in range(10):
                key = InfoSetKey(
                    player_position=i % 2,
                    street=Street.PREFLOP,
                    betting_sequence=f"s{i}",
                    preflop_hand=hands[i],
                    postflop_bucket=None,
                    spr_bucket=0,
                )

                actions = [Action(ActionType.CALL, 0), Action(ActionType.RAISE, 5)]
                infoset = storage.get_or_create_infoset(key, actions)
                infoset.regrets = np.array([float(i), float(i + 1)], dtype=np.float32)

            # Checkpoint
            storage.checkpoint(iteration=100)

            # Verify metadata saved
            metadata_file = Path(tmpdir) / "metadata.json"
            assert metadata_file.exists()

            # Reload in new storage
            storage2 = DiskBackedStorage(Path(tmpdir))

            assert storage2.num_infosets() == 10

    def test_multiple_checkpoints(self):
        """Test multiple checkpoint saves."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskBackedStorage(Path(tmpdir))

            # Create infoset (preflop uses hand string)
            key = InfoSetKey(
                player_position=0,
                street=Street.PREFLOP,
                betting_sequence="",
                preflop_hand="AKs",
                postflop_bucket=None,
                spr_bucket=1,
            )

            actions = [Action(ActionType.FOLD, 0)]
            infoset = storage.get_or_create_infoset(key, actions)
            infoset.regrets = np.array([1.0], dtype=np.float32)

            # First checkpoint
            storage.checkpoint(iteration=10)

            # Modify
            infoset.regrets = np.array([2.0], dtype=np.float32)

            # Mark as dirty since we modified it
            storage.mark_dirty(key)

            # Second checkpoint
            storage.checkpoint(iteration=20)

            # Reload should get latest
            storage2 = DiskBackedStorage(Path(tmpdir))
            infoset2 = storage2.get_infoset(key)

            assert np.allclose(infoset2.regrets, [2.0])

    def test_num_infosets(self):
        """Test num_infosets count."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskBackedStorage(Path(tmpdir))

            assert storage.num_infosets() == 0

            # Add infosets (using postflop for buckets)
            for i in range(5):
                key = InfoSetKey(
                    player_position=0,
                    street=Street.FLOP,
                    betting_sequence=str(i),
                    preflop_hand=None,
                    postflop_bucket=i,
                    spr_bucket=0,
                )
                actions = [Action(ActionType.FOLD, 0)]
                storage.get_or_create_infoset(key, actions)

            assert storage.num_infosets() == 5

    def test_get_nonexistent_infoset(self):
        """Test getting infoset that doesn't exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskBackedStorage(Path(tmpdir))

            key = InfoSetKey(
                player_position=0,
                street=Street.FLOP,
                betting_sequence="nonexistent",
                preflop_hand=None,
                postflop_bucket=99,
                spr_bucket=0,
            )

            infoset = storage.get_infoset(key)
            assert infoset is None

    def test_flush_writes_dirty_infosets(self):
        """Test that flush writes all dirty infosets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskBackedStorage(Path(tmpdir))

            # Create infoset (postflop uses bucket)
            key = InfoSetKey(
                player_position=0,
                street=Street.FLOP,
                betting_sequence="",
                preflop_hand=None,
                postflop_bucket=5,
                spr_bucket=1,
            )

            actions = [Action(ActionType.FOLD, 0)]
            infoset = storage.get_or_create_infoset(key, actions)
            infoset.regrets = np.array([42.0], dtype=np.float32)

            # Flush
            storage.flush()

            # Verify written to disk
            regret_file = Path(tmpdir) / "regrets.h5"
            assert regret_file.exists()

    def test_str_representation(self):
        """Test string representation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            storage = DiskBackedStorage(Path(tmpdir))

            s = str(storage)
            assert "DiskBackedStorage" in s
