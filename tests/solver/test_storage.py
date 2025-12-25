"""Tests for storage systems."""

import tempfile
from pathlib import Path

import h5py
import numpy as np

from src.bucketing.utils.infoset import InfoSetKey
from src.game.actions import bet, call, fold
from src.game.state import Street
from src.solver.storage import InMemoryStorage


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


class TestOptimizedHDF5Checkpoint:
    """Tests for optimized HDF5 checkpoint format."""

    def test_checkpoint_and_load_single_infoset(self):
        """Test saving and loading a single infoset with optimized format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)

            # Create storage with checkpointing
            storage = InMemoryStorage(checkpoint_dir=checkpoint_dir)

            # Create infoset
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

            # Set specific values
            infoset.regrets = np.array([1.0, 2.0, 3.0], dtype=np.float32)
            infoset.strategy_sum = np.array([0.5, 0.3, 0.2], dtype=np.float32)

            # Save checkpoint
            storage.checkpoint(iteration=100)

            # Verify files created
            assert (checkpoint_dir / "key_mapping.pkl").exists()
            assert (checkpoint_dir / "regrets.h5").exists()
            assert (checkpoint_dir / "strategies.h5").exists()

            # Load in new storage
            storage2 = InMemoryStorage(checkpoint_dir=checkpoint_dir)

            assert storage2.num_infosets() == 1

            # Verify loaded data
            loaded = storage2.get_infoset(key)
            assert loaded is not None
            assert np.allclose(loaded.regrets, [1.0, 2.0, 3.0])
            assert np.allclose(loaded.strategy_sum, [0.5, 0.3, 0.2])

    def test_checkpoint_multiple_infosets_varying_actions(self):
        """Test checkpoint with varying action counts (tests padding logic)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            storage = InMemoryStorage(checkpoint_dir=checkpoint_dir)

            # Create infosets with different action counts
            infosets_data = [
                (2, [1.0, 2.0], [0.6, 0.4]),
                (3, [3.0, 4.0, 5.0], [0.5, 0.3, 0.2]),
                (4, [6.0, 7.0, 8.0, 9.0], [0.25, 0.25, 0.25, 0.25]),
            ]

            keys = []
            for i, (num_actions, regrets, strategies) in enumerate(infosets_data):
                key = InfoSetKey(
                    player_position=0,
                    street=Street.FLOP,
                    betting_sequence=f"seq{i}",
                    preflop_hand=None,
                    postflop_bucket=i,
                    spr_bucket=1,
                )
                keys.append(key)

                actions = [fold()] * num_actions
                infoset = storage.get_or_create_infoset(key, actions)
                infoset.regrets = np.array(regrets, dtype=np.float32)
                infoset.strategy_sum = np.array(strategies, dtype=np.float32)

            # Checkpoint
            storage.checkpoint(iteration=200)

            # Load and verify
            storage2 = InMemoryStorage(checkpoint_dir=checkpoint_dir)
            assert storage2.num_infosets() == 3

            for i, (num_actions, regrets, strategies) in enumerate(infosets_data):
                loaded = storage2.get_infoset(keys[i])
                assert loaded is not None
                assert len(loaded.regrets) == num_actions
                assert np.allclose(loaded.regrets, regrets)
                assert np.allclose(loaded.strategy_sum, strategies)

    def test_hdf5_format_uses_compression(self):
        """Verify that HDF5 files use gzip compression."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            storage = InMemoryStorage(checkpoint_dir=checkpoint_dir)

            # Create many infosets
            for i in range(100):
                key = InfoSetKey(
                    player_position=0,
                    street=Street.FLOP,
                    betting_sequence=f"seq{i}",
                    preflop_hand=None,
                    postflop_bucket=i,
                    spr_bucket=1,
                )
                actions = [fold(), call()]
                infoset = storage.get_or_create_infoset(key, actions)
                infoset.regrets = np.array([float(i), float(i + 1)], dtype=np.float32)
                infoset.strategy_sum = np.array([0.5, 0.5], dtype=np.float32)

            storage.checkpoint(iteration=300)

            # Check HDF5 file for compression
            regrets_file = checkpoint_dir / "regrets.h5"
            with h5py.File(regrets_file, "r") as f:
                assert "regrets" in f
                dataset = f["regrets"]
                assert dataset.compression == "gzip"
                assert dataset.compression_opts == 4

    def test_checkpoint_matrix_format_structure(self):
        """Verify the exact structure of the optimized matrix format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            storage = InMemoryStorage(checkpoint_dir=checkpoint_dir)

            # Create infosets
            for i in range(5):
                key = InfoSetKey(
                    player_position=0,
                    street=Street.FLOP,
                    betting_sequence=f"seq{i}",
                    preflop_hand=None,
                    postflop_bucket=i,
                    spr_bucket=1,
                )
                actions = [fold(), call(), bet(50)]
                infoset = storage.get_or_create_infoset(key, actions)
                infoset.regrets = np.array([1.0, 2.0, 3.0], dtype=np.float32)

            storage.checkpoint(iteration=400)

            # Verify HDF5 structure
            regrets_file = checkpoint_dir / "regrets.h5"
            strategies_file = checkpoint_dir / "strategies.h5"

            with h5py.File(regrets_file, "r") as f:
                # Should have 2 datasets: regrets and action_counts
                assert "regrets" in f
                assert "action_counts" in f

                # Regrets should be (num_infosets, max_actions)
                regrets_data = f["regrets"][:]
                action_counts_data = f["action_counts"][:]

                assert regrets_data.shape == (5, 3)  # 5 infosets, 3 actions each
                assert action_counts_data.shape == (5,)
                assert np.all(action_counts_data == 3)

            with h5py.File(strategies_file, "r") as f:
                # Should have 1 dataset: strategies
                assert "strategies" in f
                strategies_data = f["strategies"][:]
                assert strategies_data.shape == (5, 3)

    def test_empty_storage_checkpoint(self):
        """Test checkpointing empty storage (edge case)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            storage = InMemoryStorage(checkpoint_dir=checkpoint_dir)

            # Checkpoint empty storage
            storage.checkpoint(iteration=0)

            # Should only create key_mapping file
            assert (checkpoint_dir / "key_mapping.pkl").exists()
            # HDF5 files may or may not exist for empty storage

            # Load should work without errors
            storage2 = InMemoryStorage(checkpoint_dir=checkpoint_dir)
            assert storage2.num_infosets() == 0

    def test_multiple_checkpoints_overwrite(self):
        """Test that multiple checkpoints overwrite correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            storage = InMemoryStorage(checkpoint_dir=checkpoint_dir)

            # Create infoset
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

            # First checkpoint
            infoset.regrets = np.array([1.0, 2.0], dtype=np.float32)
            storage.checkpoint(iteration=100)

            # Modify and checkpoint again
            infoset.regrets = np.array([10.0, 20.0], dtype=np.float32)
            storage.checkpoint(iteration=200)

            # Load should get latest values
            storage2 = InMemoryStorage(checkpoint_dir=checkpoint_dir)
            loaded = storage2.get_infoset(key)
            assert np.allclose(loaded.regrets, [10.0, 20.0])

    def test_large_scale_checkpoint_performance(self):
        """Test checkpoint performance with many infosets."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            storage = InMemoryStorage(checkpoint_dir=checkpoint_dir)

            # Create 1000 infosets (simulates moderate scale)
            import time

            for i in range(1000):
                key = InfoSetKey(
                    player_position=i % 2,
                    street=Street.FLOP,
                    betting_sequence=f"seq{i}",
                    preflop_hand=None,
                    postflop_bucket=i % 100,
                    spr_bucket=i % 3,
                )
                actions = [fold(), call()]
                infoset = storage.get_or_create_infoset(key, actions)
                infoset.regrets = np.array([float(i), float(i + 1)], dtype=np.float32)

            # Time the checkpoint
            start = time.time()
            storage.checkpoint(iteration=500)
            checkpoint_time = time.time() - start

            # Should be reasonably fast (< 2 seconds for 1000 infosets)
            assert checkpoint_time < 2.0, f"Checkpoint took {checkpoint_time:.2f}s (too slow)"

            # Verify all infosets loaded correctly
            storage2 = InMemoryStorage(checkpoint_dir=checkpoint_dir)
            assert storage2.num_infosets() == 1000

    def test_checkpoint_without_dir_is_noop(self):
        """Test that checkpoint without checkpoint_dir is a no-op."""
        storage = InMemoryStorage(checkpoint_dir=None)

        # Create infoset
        key = InfoSetKey(
            player_position=0,
            street=Street.FLOP,
            betting_sequence="b0.75",
            preflop_hand=None,
            postflop_bucket=25,
            spr_bucket=1,
        )
        storage.get_or_create_infoset(key, [fold(), call()])

        # Checkpoint should not raise
        storage.checkpoint(iteration=100)

        # No files should be created
        # (verified implicitly by not having a directory)

    def test_load_nonexistent_checkpoint_dir(self):
        """Test loading from nonexistent directory (should be empty)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Use nonexistent subdirectory
            checkpoint_dir = Path(tmpdir) / "nonexistent"

            # Should not raise
            storage = InMemoryStorage(checkpoint_dir=checkpoint_dir)

            # Should be empty
            assert storage.num_infosets() == 0
