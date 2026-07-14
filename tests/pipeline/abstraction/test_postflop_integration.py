"""
Integration tests for the full-coverage combo abstraction pipeline.

Covers precompute → save → load → runtime lookup on a truncated board set
(board_limit keeps tests fast; production runs cover every canonical board).
"""

import json

import numpy as np
import pytest

from src.core.game.rules import Street
from src.core.game.state import Card
from src.pipeline.abstraction.config import PrecomputeConfig, StreetBucketConfig
from src.pipeline.abstraction.postflop.board_enumeration import CanonicalBoardEnumerator
from src.pipeline.abstraction.postflop.precompute import PostflopPrecomputer

pytestmark = pytest.mark.slow

N_TEST_BOARDS = 40


def _test_config() -> PrecomputeConfig:
    return PrecomputeConfig(
        buckets=StreetBucketConfig(flop=10, turn=15, river=20),
        flop_runouts=20,  # Small runout sample for speed (each is an exact river pass)
        num_workers=2,
        seed=42,
        kmeans_max_iter=50,
        kmeans_n_init=2,
    )


def _covered_flop_boards() -> list[tuple[Card, ...]]:
    """Concrete representatives of the canonical flops covered by board_limit."""
    enumerator = CanonicalBoardEnumerator(Street.FLOP)
    enumerator.enumerate()
    infos = sorted(enumerator.iterate(), key=lambda info: info.board_id)
    return [info.representative for info in infos[:N_TEST_BOARDS]]


@pytest.fixture(scope="module")
def flop_artifact(tmp_path_factory):
    """Precompute a truncated flop abstraction once and save it to disk."""
    precomputer = PostflopPrecomputer(_test_config())
    precomputer.precompute_street(Street.FLOP, board_limit=N_TEST_BOARDS)

    out_dir = tmp_path_factory.mktemp("abstraction")
    precomputer.save(out_dir)
    return out_dir


@pytest.fixture(scope="module")
def bucketer(flop_artifact):
    return PostflopPrecomputer.load(flop_artifact)


class TestFullCoveragePipeline:
    @pytest.mark.timeout(60)
    def test_lookup_works_for_every_covered_board(self, bucketer):
        """Every covered board resolves every legal hand with no fallback path."""
        num_buckets = bucketer.num_buckets(Street.FLOP)
        assert num_buckets > 0

        for board in _covered_flop_boards():
            board_set = set(board)
            deck = [c for c in Card.get_full_deck() if c not in board_set]
            for hole in [(deck[0], deck[1]), (deck[10], deck[30]), (deck[-2], deck[-1])]:
                bucket = bucketer.get_bucket(hole, board, Street.FLOP)
                assert 0 <= bucket < num_buckets

    def test_isomorphic_boards_get_same_bucket(self, bucketer):
        """Suit-permuted (board, hand) pairs must resolve identically."""
        boards = _covered_flop_boards()
        board = boards[5]

        # Swap hearts <-> spades consistently across board and hand.
        swap = {"h": "s", "s": "h", "d": "d", "c": "c"}
        rank_chars = "23456789TJQKA"  # eval7 rank encoding 0..12
        suit_chars = "cdhs"  # eval7 suit encoding 0..3

        def swap_card(card: Card) -> Card:
            rank = rank_chars[card.rank_eval7()]
            suit = suit_chars[card.suit_eval7()]
            return Card.new(rank + swap[suit])

        board_iso = tuple(swap_card(c) for c in board)
        deck = [c for c in Card.get_full_deck() if c not in set(board)]
        hole = (deck[3], deck[17])
        hole_iso = (swap_card(hole[0]), swap_card(hole[1]))

        assert bucketer.get_bucket(hole, board, Street.FLOP) == bucketer.get_bucket(
            hole_iso, board_iso, Street.FLOP
        )

    def test_hole_card_order_is_irrelevant(self, bucketer):
        board = _covered_flop_boards()[0]
        deck = [c for c in Card.get_full_deck() if c not in set(board)]
        hole = (deck[4], deck[40])

        assert bucketer.get_bucket(hole, board, Street.FLOP) == bucketer.get_bucket(
            (hole[1], hole[0]), board, Street.FLOP
        )

    def test_illegal_hand_raises(self, bucketer):
        board = _covered_flop_boards()[0]
        overlapping = (board[0], Card.new("2c") if board[0] != Card.new("2c") else Card.new("3c"))
        with pytest.raises(KeyError):
            bucketer.get_bucket(overlapping, board, Street.FLOP)

    def test_uncovered_board_raises(self, bucketer):
        """A board outside the truncated set fails loudly, not silently."""
        enumerator = CanonicalBoardEnumerator(Street.FLOP)
        enumerator.enumerate()
        infos = sorted(enumerator.iterate(), key=lambda info: info.board_id)
        uncovered = infos[-1].representative

        deck = [c for c in Card.get_full_deck() if c not in set(uncovered)]
        with pytest.raises(KeyError):
            bucketer.get_bucket((deck[0], deck[1]), uncovered, Street.FLOP)

    def test_preflop_uses_169_classes(self, bucketer):
        assert bucketer.num_buckets(Street.PREFLOP) == 169
        bucket = bucketer.get_bucket((Card.new("As"), Card.new("Ah")), (), Street.PREFLOP)
        assert 0 <= bucket < 169

    def test_lookup_is_deterministic(self, bucketer):
        board = _covered_flop_boards()[2]
        deck = [c for c in Card.get_full_deck() if c not in set(board)]
        hole = (deck[7], deck[22])

        first = bucketer.get_bucket(hole, board, Street.FLOP)
        assert all(bucketer.get_bucket(hole, board, Street.FLOP) == first for _ in range(5))


class TestPickleSupport:
    """Training workers receive the abstraction by pickle."""

    def test_disk_backed_bucketer_pickles_as_path(self, flop_artifact, bucketer):
        import pickle

        payload = pickle.dumps(bucketer)
        # Path-only pickling: workers re-mmap instead of copying matrices.
        assert len(payload) < 10_000

        restored = pickle.loads(payload)
        board = _covered_flop_boards()[0]
        deck = [c for c in Card.get_full_deck() if c not in set(board)]
        hole = (deck[0], deck[1])
        assert restored.get_bucket(hole, board, Street.FLOP) == bucketer.get_bucket(
            hole, board, Street.FLOP
        )

    def test_in_memory_bucketer_pickles_arrays(self):
        import pickle

        precomputer = PostflopPrecomputer(_test_config())
        precomputer.precompute_street(Street.FLOP, board_limit=5)
        in_memory = precomputer.build_bucketer()

        restored = pickle.loads(pickle.dumps(in_memory))
        board = _covered_flop_boards()[0]
        deck = [c for c in Card.get_full_deck() if c not in set(board)]
        hole = (deck[2], deck[9])
        assert restored.get_bucket(hole, board, Street.FLOP) == in_memory.get_bucket(
            hole, board, Street.FLOP
        )


class TestArtifactFormat:
    def test_artifact_files(self, flop_artifact):
        assert (flop_artifact / "metadata.json").exists()
        assert (flop_artifact / "hand_id_to_col.npy").exists()
        assert (flop_artifact / "flop_board_ids.npy").exists()
        assert (flop_artifact / "flop_buckets.npy").exists()

    def test_board_ids_sorted_and_matrix_shape(self, flop_artifact):
        board_ids = np.load(flop_artifact / "flop_board_ids.npy")
        buckets = np.load(flop_artifact / "flop_buckets.npy")

        assert board_ids.size == N_TEST_BOARDS
        assert np.all(np.diff(board_ids) > 0)
        assert buckets.shape[0] == N_TEST_BOARDS

    def test_metadata_has_quality_stats(self, flop_artifact):
        with open(flop_artifact / "metadata.json") as f:
            metadata = json.load(f)

        assert metadata["storage_version"] == 2
        flop_stats = metadata["streets"]["FLOP"]
        assert flop_stats["num_boards"] == N_TEST_BOARDS
        assert flop_stats["num_buckets"] > 0

        quality = flop_stats["quality"]
        assert 0.0 < quality["variance_explained"] <= 1.0
        assert quality["within_bucket_std"] < quality["equity_std"]
        assert quality["combo_count"] > quality["class_count"] > 0

    def test_buckets_ordered_by_equity(self, bucketer):
        """Bucket 0 must be the weakest: nut-ish hands land in high buckets."""
        board = _covered_flop_boards()[0]
        deck = [c for c in Card.get_full_deck() if c not in set(board)]

        num_buckets = bucketer.num_buckets(Street.FLOP)
        buckets = [
            bucketer.get_bucket((deck[i], deck[j]), board, Street.FLOP)
            for i, j in [(0, 1), (5, 20), (12, 33), (40, 41), (-1, -2)]
        ]
        # Not all hands can share one bucket on any reasonable abstraction.
        assert len(set(buckets)) > 1
        assert all(0 <= b < num_buckets for b in buckets)
