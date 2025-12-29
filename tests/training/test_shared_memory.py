"""Tests for shared memory communication in parallel training."""

from typing import Any, Dict

import numpy as np

from src.bucketing.utils.infoset import InfoSetKey
from src.game.state import Street
from src.training.parallel import _deserialize_infosets_from_shm, _serialize_infosets_to_shm


class TestSharedMemorySerialization:
    """Tests for shared memory serialization/deserialization."""

    def test_serialize_deserialize_single_infoset(self):
        """Test serializing and deserializing a single infoset."""
        key = InfoSetKey(
            player_position=0,
            street=Street.FLOP,
            betting_sequence="b0.75",
            preflop_hand=None,
            postflop_bucket=25,
            spr_bucket=1,
        )

        infoset_data = {
            key: {
                "regrets": np.array([1.0, 2.0, 3.0], dtype=np.float32),
                "strategy_sum": np.array([0.5, 0.3, 0.2], dtype=np.float32),
                "reach_count": 42,
                "cumulative_utility": 123.456,
            }
        }

        # Serialize
        shm, shm_size = _serialize_infosets_to_shm(infoset_data, "test_single")

        try:
            # Deserialize
            result = _deserialize_infosets_from_shm("test_single", shm_size)

            # Verify
            assert len(result) == 1
            assert key in result
            assert np.allclose(result[key]["regrets"], [1.0, 2.0, 3.0])
            assert np.allclose(result[key]["strategy_sum"], [0.5, 0.3, 0.2])
            assert result[key]["reach_count"] == 42
            assert abs(result[key]["cumulative_utility"] - 123.456) < 1e-6

        finally:
            # Manual cleanup since we're not using deserialize's auto-cleanup
            shm.close()

    def test_serialize_deserialize_multiple_infosets(self):
        """Test serializing multiple infosets with varying action counts."""
        infosets = {}

        for i in range(5):
            key = InfoSetKey(
                player_position=i % 2,
                street=Street.FLOP,
                betting_sequence=f"seq{i}",
                preflop_hand=None,
                postflop_bucket=i,
                spr_bucket=1,
            )
            num_actions = 2 + i  # Varying action counts: 2, 3, 4, 5, 6
            infosets[key] = {
                "regrets": np.random.randn(num_actions).astype(np.float32),
                "strategy_sum": np.random.rand(num_actions).astype(np.float32),
                "reach_count": i * 10,
                "cumulative_utility": float(i) * 1.5,
            }

        # Serialize
        shm, shm_size = _serialize_infosets_to_shm(infosets, "test_multiple")

        try:
            # Deserialize
            result = _deserialize_infosets_from_shm("test_multiple", shm_size)

            # Verify
            assert len(result) == len(infosets)
            for key, expected_data in infosets.items():
                assert key in result
                assert np.allclose(result[key]["regrets"], expected_data["regrets"])
                assert np.allclose(result[key]["strategy_sum"], expected_data["strategy_sum"])
                assert result[key]["reach_count"] == expected_data["reach_count"]
                assert (
                    abs(result[key]["cumulative_utility"] - expected_data["cumulative_utility"])
                    < 1e-6
                )

        finally:
            shm.close()

    def test_serialize_empty_infosets(self):
        """Test serializing empty infoset dictionary."""
        infoset_data: Dict[InfoSetKey, Dict[str, Any]] = {}

        # Serialize
        shm, shm_size = _serialize_infosets_to_shm(infoset_data, "test_empty")

        try:
            # Deserialize
            result = _deserialize_infosets_from_shm("test_empty", shm_size)

            # Verify
            assert len(result) == 0
            assert isinstance(result, dict)

        finally:
            shm.close()

    def test_serialize_large_batch(self):
        """Test serializing a large batch of infosets (performance test)."""
        import time

        num_infosets = 1000
        infosets = {}

        for i in range(num_infosets):
            key = InfoSetKey(
                player_position=i % 2,
                street=Street.FLOP,
                betting_sequence=f"seq{i}",
                preflop_hand=None,
                postflop_bucket=i % 100,
                spr_bucket=i % 3,
            )
            infosets[key] = {
                "regrets": np.random.randn(3).astype(np.float32),
                "strategy_sum": np.random.rand(3).astype(np.float32),
                "reach_count": i,
                "cumulative_utility": float(i) * 0.5,
            }

        # Serialize
        start = time.time()
        shm, shm_size = _serialize_infosets_to_shm(infosets, "test_large")
        serialize_time = time.time() - start

        try:
            # Deserialize
            start = time.time()
            result = _deserialize_infosets_from_shm("test_large", shm_size)
            deserialize_time = time.time() - start

            # Verify basic correctness
            assert len(result) == num_infosets

            # Verify performance (should be fast)
            # With shared memory, both should complete in < 100ms for 1000 infosets
            assert serialize_time < 0.5, f"Serialization too slow: {serialize_time:.2f}s"
            assert deserialize_time < 0.5, f"Deserialization too slow: {deserialize_time:.2f}s"

            print(f"Serialized {num_infosets} infosets in {serialize_time:.3f}s")
            print(f"Deserialized {num_infosets} infosets in {deserialize_time:.3f}s")
            print(f"Total shared memory size: {shm_size / 1024:.1f} KB")

        finally:
            shm.close()

    def test_data_types_preserved(self):
        """Test that data types are correctly preserved through serialization."""
        key = InfoSetKey(
            player_position=0,
            street=Street.TURN,
            betting_sequence="b1.0",
            preflop_hand=None,
            postflop_bucket=50,
            spr_bucket=2,
        )

        infoset_data = {
            key: {
                "regrets": np.array([1.5, -2.3, 0.0, 100.7], dtype=np.float32),
                "strategy_sum": np.array([0.25, 0.25, 0.25, 0.25], dtype=np.float32),
                "reach_count": 999,
                "cumulative_utility": -456.789,
            }
        }

        # Serialize
        shm, shm_size = _serialize_infosets_to_shm(infoset_data, "test_types")

        try:
            # Deserialize
            result = _deserialize_infosets_from_shm("test_types", shm_size)

            # Verify types
            assert result[key]["regrets"].dtype == np.float32
            assert result[key]["strategy_sum"].dtype == np.float32
            assert isinstance(result[key]["reach_count"], int)
            assert isinstance(result[key]["cumulative_utility"], float)

            # Verify values including negatives and zeros
            assert np.allclose(result[key]["regrets"], [1.5, -2.3, 0.0, 100.7])
            assert result[key]["reach_count"] == 999
            assert abs(result[key]["cumulative_utility"] - (-456.789)) < 1e-6

        finally:
            shm.close()

    def test_cleanup_on_deserialization(self):
        """Test that shared memory is properly handled during deserialization.

        Workers should only close() but NOT unlink() - only the creator should unlink.
        """
        key = InfoSetKey(
            player_position=0,
            street=Street.RIVER,
            betting_sequence="c",
            preflop_hand=None,
            postflop_bucket=75,
            spr_bucket=0,
        )

        infoset_data = {
            key: {
                "regrets": np.array([1.0], dtype=np.float32),
                "strategy_sum": np.array([1.0], dtype=np.float32),
                "reach_count": 1,
                "cumulative_utility": 1.0,
            }
        }

        # Serialize (creator creates it)
        shm, shm_size = _serialize_infosets_to_shm(infoset_data, "test_cleanup")
        shm.close()  # Creator closes after writing

        # Deserialize (worker reads it)
        # Worker should only close(), NOT unlink()
        result = _deserialize_infosets_from_shm("test_cleanup", shm_size)

        # Verify data was received
        assert len(result) == 1

        # Shared memory should STILL EXIST because worker only closed, didn't unlink
        from multiprocessing import shared_memory

        try:
            shm_test = shared_memory.SharedMemory(name="test_cleanup")
            # Success! Shared memory still exists (correct behavior)
            # Now cleanup (simulating what master would do)
            shm_test.close()
            shm_test.unlink()
        except FileNotFoundError:
            assert False, (
                "Shared memory should still exist after worker read (worker should only close, not unlink)"
            )

    def test_cleanup_parameter(self):
        """Test that cleanup=True properly deletes shared memory after reading."""
        key = InfoSetKey(
            player_position=0,
            street=Street.RIVER,
            betting_sequence="c",
            preflop_hand=None,
            postflop_bucket=75,
            spr_bucket=0,
        )

        infoset_data = {
            key: {
                "regrets": np.array([1.0, 2.0], dtype=np.float32),
                "strategy_sum": np.array([3.0, 4.0], dtype=np.float32),
                "reach_count": 10,
                "cumulative_utility": 5.0,
            }
        }

        # Serialize
        shm, shm_size = _serialize_infosets_to_shm(infoset_data, "test_cleanup_param")
        shm.close()

        # Deserialize with cleanup=True (simulating master reading worker results)
        result = _deserialize_infosets_from_shm("test_cleanup_param", shm_size, cleanup=True)

        # Verify data was received
        assert len(result) == 1

        # Shared memory should be DELETED because cleanup=True
        from multiprocessing import shared_memory

        try:
            shm_test = shared_memory.SharedMemory(name="test_cleanup_param")
            shm_test.close()
            shm_test.unlink()
            assert False, "Shared memory should have been deleted by cleanup=True"
        except FileNotFoundError:
            # Expected - cleanup=True deleted it
            pass

    def test_different_streets(self):
        """Test serialization with keys from different streets."""
        keys = [
            InfoSetKey(
                player_position=0,
                street=Street.PREFLOP,
                betting_sequence="r3",
                preflop_hand="AKs",
                postflop_bucket=None,
                spr_bucket=0,
            ),
            InfoSetKey(
                player_position=1,
                street=Street.FLOP,
                betting_sequence="b0.5",
                preflop_hand=None,
                postflop_bucket=25,
                spr_bucket=1,
            ),
            InfoSetKey(
                player_position=0,
                street=Street.TURN,
                betting_sequence="c",
                preflop_hand=None,
                postflop_bucket=50,
                spr_bucket=2,
            ),
            InfoSetKey(
                player_position=1,
                street=Street.RIVER,
                betting_sequence="b1.0",
                preflop_hand=None,
                postflop_bucket=75,
                spr_bucket=1,
            ),
        ]

        infosets = {}
        for i, key in enumerate(keys):
            infosets[key] = {
                "regrets": np.array([float(i)], dtype=np.float32),
                "strategy_sum": np.array([1.0], dtype=np.float32),
                "reach_count": i + 1,
                "cumulative_utility": float(i) * 2.0,
            }

        # Serialize
        shm, shm_size = _serialize_infosets_to_shm(infosets, "test_streets")

        try:
            # Deserialize
            result = _deserialize_infosets_from_shm("test_streets", shm_size)

            # Verify all streets preserved
            assert len(result) == 4
            for i, key in enumerate(keys):
                assert key in result
                assert result[key]["regrets"][0] == float(i)
                assert result[key]["reach_count"] == i + 1

        finally:
            shm.close()
