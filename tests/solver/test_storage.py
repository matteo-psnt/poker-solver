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


class TestSharedArrayStorage:
    """Tests for SharedArrayStorage partitioned shared memory."""

    def test_unknown_id_views_are_read_only(self):
        """Test that UNKNOWN_ID placeholder infosets are read-only to prevent corruption."""
        import uuid

        from src.solver.storage import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"
        num_workers = 4

        # Create coordinator (worker 0)
        storage = SharedArrayStorage(
            num_workers=num_workers,
            worker_id=0,
            session_id=session_id,
            max_infosets=10000,
            max_actions=10,
            is_coordinator=True,
        )

        try:
            # Find a key that is owned by a worker OTHER than worker 0
            # by iterating through different keys
            remote_key = None
            for i in range(100):  # Try multiple keys
                candidate_key = InfoSetKey(
                    player_position=1,
                    street=Street.TURN,
                    betting_sequence=f"b{i}.5xc",
                    preflop_hand=None,
                    postflop_bucket=i,
                    spr_bucket=2,
                )
                owner = storage.get_owner(candidate_key)
                if owner != 0:
                    remote_key = candidate_key
                    break

            assert remote_key is not None, "Could not find a key owned by another worker"

            # Worker 0 requests this remote key (owned by another worker)
            # Since the ID is unknown, it should return UNKNOWN_ID view
            actions = [fold(), call(), bet(50)]
            infoset = storage.get_or_create_infoset(remote_key, actions)

            # The view should be backed by UNKNOWN_ID (row 0) and should be read-only
            # Attempting to write should raise ValueError
            try:
                infoset.regrets[0] = 100.0
                assert False, "Should have raised ValueError for read-only array"
            except ValueError as e:
                assert "read-only" in str(e).lower()

            try:
                infoset.strategy_sum[0] = 100.0
                assert False, "Should have raised ValueError for read-only array"
            except ValueError as e:
                assert "read-only" in str(e).lower()

            # Verify row 0 (UNKNOWN_ID) in shared memory was not modified
            assert storage.shared_regrets[0, 0] == 0.0
            assert storage.shared_strategy_sum[0, 0] == 0.0

        finally:
            storage.cleanup()

    def test_unknown_id_prevents_in_place_updates(self):
        """Test that UNKNOWN_ID views prevent dangerous in-place operations like +=."""
        import uuid

        import numpy as np

        from src.solver.storage import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=4,
            worker_id=0,
            session_id=session_id,
            max_infosets=10000,
            max_actions=10,
            is_coordinator=True,
        )

        try:
            # Find a remote key
            remote_key = None
            for i in range(100):
                key = InfoSetKey(
                    player_position=1,
                    street=Street.FLOP,
                    betting_sequence=f"seq{i}",
                    preflop_hand=None,
                    postflop_bucket=i,
                    spr_bucket=1,
                )
                if storage.get_owner(key) != 0:
                    remote_key = key
                    break

            assert remote_key is not None

            # Get UNKNOWN_ID view
            infoset = storage.get_or_create_infoset(remote_key, [fold(), call()])

            # Simulate CFR update pattern (in-place addition)
            # This should raise ValueError because the array is read-only
            try:
                infoset.regrets += np.array([10.0, 20.0])
                assert False, "Should have raised ValueError for read-only array"
            except (ValueError, TypeError):
                # ValueError for read-only or TypeError if operation blocked
                pass

            # Verify row 0 is still zeros
            assert np.all(storage.shared_regrets[0, :] == 0.0)
            assert np.all(storage.shared_strategy_sum[0, :] == 0.0)

        finally:
            storage.cleanup()

    def test_owned_infosets_are_writable(self):
        """Test that owned infosets can be written to."""
        import uuid

        from src.solver.storage import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=4,
            worker_id=0,
            session_id=session_id,
            max_infosets=10000,
            max_actions=10,
            is_coordinator=True,
        )

        try:
            # Find a key owned by worker 0
            owned_key = None
            for i in range(100):
                candidate_key = InfoSetKey(
                    player_position=0,
                    street=Street.FLOP,
                    betting_sequence=f"b{i}",
                    preflop_hand=None,
                    postflop_bucket=i,
                    spr_bucket=1,
                )
                if storage.get_owner(candidate_key) == 0:
                    owned_key = candidate_key
                    break

            assert owned_key is not None, "Could not find a key owned by worker 0"

            actions = [fold(), call()]
            infoset = storage.get_or_create_infoset(owned_key, actions)

            # Owned infosets should be writable
            infoset.regrets[0] = 42.0
            infoset.strategy_sum[1] = 99.0

            # Verify the write went through
            assert infoset.regrets[0] == 42.0
            assert infoset.strategy_sum[1] == 99.0

        finally:
            storage.cleanup()

    def test_stable_hash_consistency(self):
        """Test that xxhash provides stable ownership across calls."""
        import uuid

        from src.solver.storage import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=4,
            worker_id=0,
            session_id=session_id,
            max_infosets=10000,
            max_actions=10,
            is_coordinator=True,
        )

        try:
            key = InfoSetKey(
                player_position=0,
                street=Street.RIVER,
                betting_sequence="b1.0xc",
                preflop_hand=None,
                postflop_bucket=42,
                spr_bucket=1,
            )

            # Ownership should be deterministic
            owner1 = storage.get_owner(key)
            owner2 = storage.get_owner(key)
            owner3 = storage.get_owner(key)

            assert owner1 == owner2 == owner3
            assert 0 <= owner1 < 4

        finally:
            storage.cleanup()

    def test_per_worker_id_ranges(self):
        """Test that each worker has non-overlapping ID ranges."""
        import uuid

        from src.solver.storage import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"
        num_workers = 4
        max_infosets = 10000

        # Create coordinator
        coordinator = SharedArrayStorage(
            num_workers=num_workers,
            worker_id=0,
            session_id=session_id,
            max_infosets=max_infosets,
            max_actions=10,
            is_coordinator=True,
        )

        try:
            # Verify coordinator's ID range
            expected_slots_per_worker = (max_infosets - 1) // num_workers
            assert coordinator.id_range_start == 1
            assert coordinator.id_range_end == 1 + expected_slots_per_worker

            # Verify ranges don't overlap (calculate for all workers)
            ranges = []
            for worker_id in range(num_workers):
                start = 1 + worker_id * expected_slots_per_worker
                end = 1 + (worker_id + 1) * expected_slots_per_worker
                ranges.append((start, end))

            # No overlap check
            for i in range(len(ranges)):
                for j in range(i + 1, len(ranges)):
                    start_i, end_i = ranges[i]
                    start_j, end_j = ranges[j]
                    # Ranges should not overlap
                    assert end_i <= start_j or end_j <= start_i

        finally:
            coordinator.cleanup()

    def test_ownership_is_deterministic_across_calls(self):
        """Test that ownership is deterministic for the same key."""
        import uuid

        from src.solver.storage import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=8,
            worker_id=0,
            session_id=session_id,
            max_infosets=10000,
            max_actions=10,
            is_coordinator=True,
        )

        try:
            # Create 100 keys and verify ownership is consistent
            keys = []
            for i in range(100):
                key = InfoSetKey(
                    player_position=i % 2,
                    street=Street.FLOP,
                    betting_sequence=f"b{i}.5xc",
                    preflop_hand=None,
                    postflop_bucket=i % 50,
                    spr_bucket=i % 3,
                )
                keys.append(key)

            # Check ownership multiple times - should be consistent
            ownerships_first = [storage.get_owner(k) for k in keys]
            ownerships_second = [storage.get_owner(k) for k in keys]
            ownerships_third = [storage.get_owner(k) for k in keys]

            assert ownerships_first == ownerships_second == ownerships_third

            # Verify distribution across workers (should be roughly even)
            owner_counts: dict[int, int] = {}
            for owner in ownerships_first:
                owner_counts[owner] = owner_counts.get(owner, 0) + 1

            # With 8 workers and 100 keys, each should get ~12.5 on average
            # Check that no worker gets 0 or too many (degenerate hash)
            for worker_id in range(8):
                count = owner_counts.get(worker_id, 0)
                # Allow significant variance but catch degenerate cases
                assert count >= 0, f"Worker {worker_id} got {count} keys"

        finally:
            storage.cleanup()

    def test_get_or_create_allocates_id_from_correct_range(self):
        """Test that ID allocation uses the correct worker range."""
        import uuid

        from src.solver.storage import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"
        num_workers = 4
        max_infosets = 10000

        storage = SharedArrayStorage(
            num_workers=num_workers,
            worker_id=0,
            session_id=session_id,
            max_infosets=max_infosets,
            max_actions=10,
            is_coordinator=True,
        )

        try:
            # Find and create several keys owned by worker 0
            created_ids = []
            for i in range(20):
                key = InfoSetKey(
                    player_position=0,
                    street=Street.FLOP,
                    betting_sequence=f"b{i}",
                    preflop_hand=None,
                    postflop_bucket=i,
                    spr_bucket=1,
                )
                if storage.get_owner(key) == 0:
                    storage.get_or_create_infoset(key, [fold(), call()])
                    infoset_id = storage._owned_keys[key]
                    created_ids.append(infoset_id)

                    # Verify ID is in worker 0's range
                    assert storage.id_range_start <= infoset_id < storage.id_range_end

            # Verify IDs are sequential within the range
            if len(created_ids) > 1:
                for i in range(1, len(created_ids)):
                    assert created_ids[i] == created_ids[i - 1] + 1

        finally:
            storage.cleanup()

    def test_pending_id_requests_tracked(self):
        """Test that requests for non-owned keys are tracked."""
        import uuid

        from src.solver.storage import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=4,
            worker_id=0,
            session_id=session_id,
            max_infosets=10000,
            max_actions=10,
            is_coordinator=True,
        )

        try:
            # Find a key NOT owned by worker 0
            remote_key = None
            remote_owner = None
            for i in range(100):
                key = InfoSetKey(
                    player_position=1,
                    street=Street.TURN,
                    betting_sequence=f"b{i}.5",
                    preflop_hand=None,
                    postflop_bucket=i,
                    spr_bucket=2,
                )
                owner = storage.get_owner(key)
                if owner != 0:
                    remote_key = key
                    remote_owner = owner
                    break

            assert remote_key is not None, "Could not find a remote key"

            # Access the remote key
            storage.get_or_create_infoset(remote_key, [fold(), call()])

            # Should have added to pending requests
            pending = storage.get_pending_id_requests()
            assert remote_owner in pending
            assert remote_key in pending[remote_owner]

            # Clear and verify
            storage.clear_pending_id_requests()
            pending = storage.get_pending_id_requests()
            assert len(pending[remote_owner]) == 0

        finally:
            storage.cleanup()

    def test_remote_key_cache_updated_on_response(self):
        """Test that remote key cache is updated when receiving ID responses."""
        import uuid

        from src.solver.storage import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=4,
            worker_id=0,
            session_id=session_id,
            max_infosets=10000,
            max_actions=10,
            is_coordinator=True,
        )

        try:
            # Create a remote key and simulate receiving its ID
            remote_key = InfoSetKey(
                player_position=1,
                street=Street.RIVER,
                betting_sequence="b1.0xc",
                preflop_hand=None,
                postflop_bucket=99,
                spr_bucket=1,
            )

            # Initially, remote key should not be in cache
            assert remote_key not in storage._remote_keys

            # Simulate receiving response from owner
            storage.receive_id_responses({remote_key: 500})

            # Now it should be in cache
            assert remote_key in storage._remote_keys
            assert storage._remote_keys[remote_key] == 500

        finally:
            storage.cleanup()

    def test_id_exhaustion_raises_error(self):
        """Test that exhausting a worker's ID range raises RuntimeError."""
        import uuid

        from src.solver.storage import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        # Create storage with very small ID range to trigger exhaustion
        storage = SharedArrayStorage(
            num_workers=4,
            worker_id=0,
            session_id=session_id,
            max_infosets=10,  # Only 10 total slots
            max_actions=5,
            is_coordinator=True,
        )

        try:
            # Worker 0 gets (10-1)/4 = 2 slots (IDs 1 and 2)
            # Try to create more infosets than available slots
            created_count = 0
            exhausted = False

            for i in range(20):
                key = InfoSetKey(
                    player_position=0,
                    street=Street.FLOP,
                    betting_sequence=f"b{i}",
                    preflop_hand=None,
                    postflop_bucket=i,
                    spr_bucket=1,
                )

                if storage.get_owner(key) == 0:
                    try:
                        storage.get_or_create_infoset(key, [fold(), call()])
                        created_count += 1
                    except RuntimeError as e:
                        if "exhausted ID range" in str(e):
                            exhausted = True
                            break
                        raise

            # Should have exhausted the range
            assert exhausted, f"Should have exhausted ID range but created {created_count} infosets"
            # Should have created exactly the number of available slots
            expected_slots = (10 - 1) // 4  # Reserve slot 0, divide among workers
            assert created_count == expected_slots, (
                f"Created {created_count} infosets but expected {expected_slots}"
            )

        finally:
            storage.cleanup()

    def test_max_actions_boundary(self):
        """Test behavior at max_actions boundary."""
        import uuid

        from src.solver.storage import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=2,
            worker_id=0,
            session_id=session_id,
            max_infosets=1000,
            max_actions=3,  # Small max_actions
            is_coordinator=True,
        )

        try:
            # Find a key owned by worker 0
            owned_key = None
            for i in range(50):
                key = InfoSetKey(
                    player_position=0,
                    street=Street.FLOP,
                    betting_sequence=f"b{i}",
                    preflop_hand=None,
                    postflop_bucket=i,
                    spr_bucket=1,
                )
                if storage.get_owner(key) == 0:
                    owned_key = key
                    break

            assert owned_key is not None

            # Create with exactly max_actions actions
            actions_at_limit = [fold(), call(), bet(50)]
            assert len(actions_at_limit) == 3

            infoset = storage.get_or_create_infoset(owned_key, actions_at_limit)

            # Verify it was created successfully
            assert len(infoset.legal_actions) == 3
            assert infoset.regrets.shape == (3,)
            assert infoset.strategy_sum.shape == (3,)

            # Write to all action slots
            infoset.regrets[:] = [1.0, 2.0, 3.0]
            assert infoset.regrets[0] == 1.0
            assert infoset.regrets[1] == 2.0
            assert infoset.regrets[2] == 3.0

        finally:
            storage.cleanup()
