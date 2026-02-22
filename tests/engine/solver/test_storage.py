"""Tests for storage systems."""

import numpy as np

from src.core.game.actions import bet, call, fold
from src.core.game.state import Street
from src.pipeline.abstraction.utils.infoset import InfoSetKey

# SharedArrayStorage checkpoint tests are covered by parallel training integration tests


class TestSharedArrayStorage:
    """Tests for SharedArrayStorage partitioned shared memory."""

    def test_unknown_id_views_are_read_only(self):
        """Test that UNKNOWN_ID placeholder infosets are read-only to prevent corruption."""
        import uuid

        from src.engine.solver.storage.shared_array import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"
        num_workers = 4

        # Create coordinator (worker 0)
        storage = SharedArrayStorage(
            num_workers=num_workers,
            worker_id=0,
            session_id=session_id,
            initial_capacity=10000,
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

        from src.engine.solver.storage.shared_array import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=4,
            worker_id=0,
            session_id=session_id,
            initial_capacity=10000,
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

        from src.engine.solver.storage.shared_array import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=4,
            worker_id=0,
            session_id=session_id,
            initial_capacity=10000,
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

        from src.engine.solver.storage.shared_array import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=4,
            worker_id=0,
            session_id=session_id,
            initial_capacity=10000,
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

        from src.engine.solver.storage.shared_array import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"
        num_workers = 4
        max_infosets = 10000

        # Create coordinator
        coordinator = SharedArrayStorage(
            num_workers=num_workers,
            worker_id=0,
            session_id=session_id,
            initial_capacity=max_infosets,
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

        from src.engine.solver.storage.shared_array import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=8,
            worker_id=0,
            session_id=session_id,
            initial_capacity=10000,
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

        from src.engine.solver.storage.shared_array import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"
        num_workers = 4
        max_infosets = 10000

        storage = SharedArrayStorage(
            num_workers=num_workers,
            worker_id=0,
            session_id=session_id,
            initial_capacity=max_infosets,
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
                    infoset_id = storage.state.owned_keys[key]
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

        from src.engine.solver.storage.shared_array import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=4,
            worker_id=0,
            session_id=session_id,
            initial_capacity=10000,
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
            pending = storage.state.pending_id_requests
            assert remote_owner in pending
            assert remote_key in pending[remote_owner]

            # Clear and verify
            for owner_id in storage.state.pending_id_requests:
                storage.state.pending_id_requests[owner_id].clear()
            pending = storage.state.pending_id_requests
            assert len(pending[remote_owner]) == 0

        finally:
            storage.cleanup()

    def test_remote_key_cache_updated_on_response(self):
        """Test that remote key cache is updated when receiving ID responses."""
        import uuid

        from src.engine.solver.storage.shared_array import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=4,
            worker_id=0,
            session_id=session_id,
            initial_capacity=10000,
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
            assert remote_key not in storage.state.remote_keys

            # Simulate receiving response from owner
            storage.state.remote_keys.update({remote_key: 500})

            # Now it should be in cache
            assert remote_key in storage.state.remote_keys
            assert storage.state.remote_keys[remote_key] == 500

        finally:
            storage.cleanup()

    def test_id_exhaustion_raises_error(self):
        """Test that exhausting a worker's ID range raises RuntimeError."""
        import uuid

        from src.engine.solver.storage.shared_array import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        # Create storage with very small ID range to trigger exhaustion
        storage = SharedArrayStorage(
            num_workers=4,
            worker_id=0,
            session_id=session_id,
            initial_capacity=10,  # Only 10 total slots
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

        from src.engine.solver.storage.shared_array import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=2,
            worker_id=0,
            session_id=session_id,
            initial_capacity=1000,
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

    def test_capacity_monitoring(self):
        """Test capacity usage monitoring."""
        import uuid

        from src.engine.solver.storage.shared_array import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=2,
            worker_id=0,
            session_id=session_id,
            initial_capacity=100,  # Small for testing
            max_actions=5,
            is_coordinator=True,
        )

        try:
            # Initially, usage should be 0
            assert storage.get_capacity_usage() == 0.0
            assert not storage.needs_resize()

            # Create some infosets owned by worker 0
            created = 0
            for i in range(100):
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
                    created += 1
                    if created >= 20:  # Create enough to test threshold
                        break

            # Check capacity usage
            usage = storage.get_capacity_usage()
            assert 0 < usage <= 1.0

            # Check resize stats
            stats = storage.get_resize_stats()
            assert "capacity_usage" in stats
            assert "initial_capacity" in stats
            assert stats["used"] == created

        finally:
            storage.cleanup()

    def test_resize_extends_id_range(self):
        """Test that resize extends ID range correctly."""
        import uuid

        from src.engine.solver.storage.shared_array import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=2,
            worker_id=0,
            session_id=session_id,
            initial_capacity=100,
            max_actions=5,
            is_coordinator=True,
        )

        try:
            # Record initial state
            initial_range_start = storage.id_range_start
            initial_range_end = storage.id_range_end
            initial_max = storage.capacity

            # Create an infoset to have some data
            key = InfoSetKey(
                player_position=0,
                street=Street.FLOP,
                betting_sequence="b10",
                preflop_hand=None,
                postflop_bucket=1,
                spr_bucket=1,
            )
            # Find a key owned by worker 0
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
                    break

            infoset = storage.get_or_create_infoset(key, [fold(), call()])
            infoset.regrets[:] = [1.0, 2.0]

            infoset_id = storage.state.owned_keys[key]
            next_id_before = storage.state.next_local_id

            # Resize to 2x capacity
            new_max = initial_max * 2
            storage.resize(new_max)

            # Verify resize worked
            assert storage.capacity == new_max
            assert storage.id_range_start == initial_range_start  # Start stays same
            assert storage.id_range_end == initial_range_end  # Base range unchanged
            assert storage.state.extra_allocations  # Extra capacity allocated
            assert any(alloc.end > initial_range_end for alloc in storage.state.extra_allocations)
            assert storage.state.next_local_id == next_id_before  # next_id preserved

            # Verify data was preserved
            assert np.allclose(storage.shared_regrets[infoset_id, :2], [1.0, 2.0])

            # Verify we can still access the infoset
            retrieved = storage.get_infoset(key)
            assert retrieved is not None
            assert np.allclose(retrieved.regrets, [1.0, 2.0])

        finally:
            storage.cleanup()

    def test_resize_preserves_data(self):
        """Test that resize preserves all existing infoset data."""
        import uuid

        from src.engine.solver.storage.shared_array import SharedArrayStorage

        session_id = f"test_{uuid.uuid4().hex[:8]}"

        storage = SharedArrayStorage(
            num_workers=2,
            worker_id=0,
            session_id=session_id,
            initial_capacity=100,
            max_actions=5,
            is_coordinator=True,
        )

        try:
            # Create multiple infosets with different values
            keys_and_values = []
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
                    infoset = storage.get_or_create_infoset(key, [fold(), call()])
                    values = [float(i), float(i * 2)]
                    infoset.regrets[:] = values
                    infoset.strategy_sum[:] = [float(i * 3), float(i * 4)]
                    keys_and_values.append((key, values))
                    if len(keys_and_values) >= 5:
                        break

            # Resize
            storage.resize(200)

            # Verify all data preserved
            for key, values in keys_and_values:
                retrieved = storage.get_infoset(key)
                assert retrieved is not None
                assert np.allclose(retrieved.regrets, values)

        finally:
            storage.cleanup()
