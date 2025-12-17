"""Tests for Card pickling/hashing across multiprocessing."""

import multiprocessing as mp
import pickle
from typing import Set

import pytest

from src.game.state import Card

# Module-level functions for multiprocessing (must be picklable)


def worker_hash_cards(card_strs):
    """Worker function that creates cards and returns their hashes."""
    hashes = []
    for card_str in card_strs:
        card = Card.new(card_str)
        hashes.append((card_str, hash(card)))
    return hashes


def worker_use_card_in_set(card_str):
    """Worker function that uses cards in sets (tests __hash__ and __eq__)."""
    card1 = Card.new(card_str)
    card2 = Card.new(card_str)

    # Test set operations
    card_set: Set[Card] = {card1, card2}

    # Both should hash to same value
    assert len(card_set) == 1, f"Expected 1 card in set, got {len(card_set)}"
    assert card1 in card_set
    assert card2 in card_set

    return hash(card1), hash(card2), len(card_set)


def worker_pickle_and_hash(card_str):
    """Worker function that pickles a card and hashes it."""
    # Create card
    card = Card.new(card_str)

    # Get hash before pickle
    hash_before = hash(card)

    # Pickle and unpickle
    pickled = pickle.dumps(card)
    unpickled_card = pickle.loads(pickled)

    # Get hash after pickle
    hash_after = hash(unpickled_card)

    # They should be equal
    return hash_before, hash_after, hash_before == hash_after


def check_conflicts(args):
    """Check if cards conflict (module-level for pickling)."""
    hole_cards_strs, board_strs = args
    hole_cards = tuple(Card.new(c) for c in hole_cards_strs)
    board = tuple(Card.new(c) for c in board_strs)

    # Check for conflicts
    all_cards = set(hole_cards) | set(board)
    has_conflict = len(all_cards) < len(hole_cards) + len(board)

    return has_conflict


def compute_mock_equity(args):
    """Compute mock equity (module-level for pickling)."""
    hand_str, board_strs = args

    # Create cards (similar to parallel_equity._get_example_hand)
    if len(hand_str) == 2:  # Pair
        rank = hand_str[0]
        hole_cards = (Card.new(f"{rank}h"), Card.new(f"{rank}d"))
    else:
        high_rank = hand_str[0]
        low_rank = hand_str[1]
        suited = hand_str[2] == "s"
        if suited:
            hole_cards = (Card.new(f"{high_rank}s"), Card.new(f"{low_rank}s"))
        else:
            hole_cards = (Card.new(f"{high_rank}h"), Card.new(f"{low_rank}d"))

    # Create board
    board = tuple(Card.new(b) for b in board_strs)

    # Use cards in set operations (this is where hash is called)
    all_cards = set(hole_cards) | set(board)
    has_conflict = len(all_cards) < len(hole_cards) + len(board)

    # Return some mock result
    return len(all_cards), has_conflict


class TestCardMultiprocessing:
    """Tests for Card behavior across process boundaries."""

    def test_card_hash_in_worker_process(self):
        """Test that Card.__hash__ works in worker processes."""
        card_strs = ["As", "Kh", "Qd", "Jc", "Ts"]

        # Run in worker process
        with mp.Pool(processes=2) as pool:
            results = pool.map(worker_hash_cards, [card_strs])

        # Check results
        for card_str, card_hash in results[0]:
            # Hash should be non-zero
            assert card_hash != 0, f"Hash for {card_str} is 0"

    def test_card_set_operations_in_worker(self):
        """Test that cards work correctly in sets across processes."""
        card_strs = ["As", "Kh", "Qd", "Jc"]

        with mp.Pool(processes=2) as pool:
            results = pool.map(worker_use_card_in_set, card_strs)

        for hash1, hash2, set_size in results:
            assert hash1 == hash2, "Same card should have same hash"
            assert set_size == 1, "Set should have exactly 1 card"

    def test_card_pickle_preserves_hash(self):
        """Test that pickling/unpickling preserves hash value."""
        card_strs = ["As", "Kh", "2d", "7c"]

        with mp.Pool(processes=2) as pool:
            results = pool.map(worker_pickle_and_hash, card_strs)

        for hash_before, hash_after, are_equal in results:
            assert are_equal, f"Hash changed after pickle: {hash_before} → {hash_after}"
            assert hash_before == hash_after

    def test_card_conflict_detection_multiprocess(self):
        """Test card conflict detection works across processes."""
        test_cases = [
            (("As", "Kh"), ("Qd", "Jc", "Ts")),  # No conflict
            (("As", "Kh"), ("As", "Qd", "Jc")),  # Conflict (As appears twice)
            (("Kh", "Kd"), ("Kh", "2h", "3s")),  # Conflict (Kh appears twice)
        ]

        with mp.Pool(processes=2) as pool:
            results = pool.map(check_conflicts, test_cases)

        assert not results[0], "Should have no conflict"
        assert results[1], "Should have conflict (As)"
        assert results[2], "Should have conflict (Kh)"

    def test_card_hash_attribute_exists(self):
        """Test that _hash attribute is properly initialized."""
        card = Card.new("As")

        # Check attribute exists
        assert hasattr(card, "_hash"), "Card should have _hash attribute"

        # After first hash call, it should be cached
        hash1 = hash(card)
        assert card._hash is not None, "_hash should be cached after first call"
        assert card._hash == hash1

        # Second call should return cached value
        hash2 = hash(card)
        assert hash2 == hash1
        assert card._hash == hash1

    def test_card_hash_after_pickle_no_cache(self):
        """Test that hash works even if _hash cache is lost during pickle."""
        card = Card.new("Kd")

        # Pickle the card
        pickled = pickle.dumps(card)
        unpickled = pickle.loads(pickled)

        # Even if _hash is not preserved, hash() should still work
        hash_value = hash(unpickled)
        assert hash_value != 0

        # And subsequent calls should work too
        hash_value2 = hash(unpickled)
        assert hash_value2 == hash_value

    def test_parallel_equity_card_usage(self):
        """Test Card usage pattern similar to parallel equity computation."""
        test_hands = [
            ("AA", ["Ks", "Qh", "Jd"]),  # No conflict
            ("AKs", ["Ah", "Kh", "2c"]),  # Conflict: Ah in hole and board
            ("72o", ["7h", "2d", "Ac"]),  # Conflict: 7h in hole (7h,2d) and board
        ]

        with mp.Pool(processes=2) as pool:
            results = pool.map(compute_mock_equity, test_hands)

        # Verify results
        num_cards_1, has_conflict_1 = results[0]
        assert num_cards_1 == 5  # AA (Ah, Ad) + board (Ks, Qh, Jd) = 5 unique
        assert not has_conflict_1

        num_cards_2, has_conflict_2 = results[1]
        # AKs = (As, Ks), board = (Ah, Kh, 2c) - no duplicates
        assert num_cards_2 == 5
        assert not has_conflict_2

        num_cards_3, has_conflict_3 = results[2]
        # 72o = (7h, 2d), board = (7h, 2d, Ac) - 7h appears in both!
        assert has_conflict_3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
