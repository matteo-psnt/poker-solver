"""Tests for PostflopBucketer public mutation APIs."""

from dataclasses import dataclass
from typing import cast

from src.core.game.state import Card, Street
from src.pipeline.abstraction.postflop.board_clustering import BoardClusterer
from src.pipeline.abstraction.postflop.hand_bucketing import PostflopBucketer


@dataclass
class _DummyClusterer:
    cluster_id: int

    def get_cluster(self, board_id: int, street: Street) -> int | None:
        _ = (board_id, street)
        return self.cluster_id

    def predict(self, board: tuple[Card, ...], street: Street) -> int:
        _ = (board, street)
        return self.cluster_id


def test_assign_bucket_with_public_api():
    bucketer = PostflopBucketer()
    bucketer.set_board_clusterer(cast(BoardClusterer, _DummyClusterer(cluster_id=7)))

    board = (Card.new("Ah"), Card.new("Kd"), Card.new("2c"))
    hole_cards = (Card.new("Qs"), Card.new("Jh"))
    combo = bucketer.canonicalize(hole_cards, board)

    bucketer.assign_bucket(Street.FLOP, cluster_id=7, hand_id=combo.hand_id, bucket_id=3)

    assert bucketer.get_bucket(hole_cards, board, Street.FLOP) == 3


def test_assign_buckets_bulk():
    bucketer = PostflopBucketer()
    bucketer.set_board_clusterer(cast(BoardClusterer, _DummyClusterer(cluster_id=2)))

    board = (Card.new("Ac"), Card.new("Tc"), Card.new("4d"))
    hand_a = (Card.new("Kh"), Card.new("Qh"))
    hand_b = (Card.new("7s"), Card.new("6s"))
    combo_a = bucketer.canonicalize(hand_a, board)
    combo_b = bucketer.canonicalize(hand_b, board)

    bucketer.assign_buckets(
        Street.FLOP,
        assignments={
            2: {
                combo_a.hand_id: 1,
                combo_b.hand_id: 5,
            }
        },
    )

    assert bucketer.get_bucket(hand_a, board, Street.FLOP) == 1
    assert bucketer.get_bucket(hand_b, board, Street.FLOP) == 5
