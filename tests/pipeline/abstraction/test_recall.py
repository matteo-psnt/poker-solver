"""Path recall composes the base's per-street ids and nothing else changes.

The wrapper is only useful if every consumer that keys on ``(node, bucket)``
picks it up unchanged, so the checks are on the seams: the id arithmetic, the
counts the tree allocates from, the board context the vector kernel builds,
the resolver that every process loads through, and the compiled scalar walk
that must decline it rather than read base buckets under a composite tree.
"""

from __future__ import annotations

import json
import pickle

import pytest

from src.core.game.state import FULL_DECK, Street
from src.engine.solver.mccfr.compiled_walk import CompiledContext
from src.pipeline.abstraction.recall import PathRecallBucketer
from src.pipeline.abstraction.resolver import ComboAbstractionResolver
from src.pipeline.abstraction.vector_universe import build_hand_context
from src.shared.config.loader import load_training_config
from tests.pipeline.abstraction.test_vector_universe import BOARD, StubAbstraction

F, T, R = (StubAbstraction.counts[s] for s in (Street.FLOP, Street.TURN, Street.RIVER))
HAND = (FULL_DECK[1], FULL_DECK[2])
CARDS = tuple(FULL_DECK[i] for i in BOARD)


@pytest.fixture(scope="module")
def wrapped():
    return PathRecallBucketer(StubAbstraction())


class TestComposition:
    def test_counts_multiply_along_the_path(self, wrapped):
        assert wrapped.num_buckets(Street.PREFLOP) == 169
        assert wrapped.num_buckets(Street.FLOP) == F
        assert wrapped.num_buckets(Street.TURN) == F * T
        assert wrapped.num_buckets(Street.RIVER) == F * T * R

    def test_turn_and_river_ids_are_mixed_radix_over_the_path(self, wrapped):
        base = wrapped.base
        flop = base.get_bucket(HAND, CARDS[:3], Street.FLOP)
        turn = base.get_bucket(HAND, CARDS[:4], Street.TURN)
        river = base.get_bucket(HAND, CARDS, Street.RIVER)
        assert wrapped.get_bucket(HAND, CARDS[:3], Street.FLOP) == flop
        assert wrapped.get_bucket(HAND, CARDS[:4], Street.TURN) == flop * T + turn
        assert wrapped.get_bucket(HAND, CARDS, Street.RIVER) == (flop * T + turn) * R + river

    def test_preflop_is_the_base_class(self, wrapped):
        assert wrapped.get_bucket(HAND, (), Street.PREFLOP) == wrapped.base.get_bucket(
            HAND, (), Street.PREFLOP
        )

    def test_a_later_street_recalls_an_earlier_split(self, wrapped):
        """Two hands in different flop buckets never share a turn row."""
        base = wrapped.base
        hands = [(FULL_DECK[a], FULL_DECK[b]) for a in range(4, 20) for b in range(a + 1, 20)]
        by_flop: dict[int, set[int]] = {}
        for hand in hands:
            flop = base.get_bucket(hand, CARDS[:3], Street.FLOP)
            by_flop.setdefault(flop, set()).add(wrapped.get_bucket(hand, CARDS[:4], Street.TURN))
        rows = list(by_flop.values())
        assert len(rows) > 1
        for i, left in enumerate(rows):
            for right in rows[i + 1 :]:
                assert not left & right


class TestSeams:
    def test_the_board_context_stays_within_the_composite_counts(self, wrapped):
        context = build_hand_context(BOARD, wrapped)
        for street in (Street.FLOP, Street.TURN, Street.RIVER):
            row = context.buckets_for(street)
            assert row.min() >= 0
            assert row.max() < wrapped.num_buckets(street)
        # The turn row carries the flop split: same flop bucket -> same quotient.
        flop_row = context.buckets_for(Street.FLOP)
        turn_row = context.buckets_for(Street.TURN)
        assert (turn_row // T == flop_row).all()

    def test_the_compiled_scalar_walk_declines_it(self, wrapped):
        """Duck-typing on the artifact's arrays would read BASE buckets under a
        tree sized for composite ones -- no error, wrong rows."""
        assert CompiledContext.for_abstraction(None, wrapped, FULL_DECK) is None

    def test_it_pickles_to_a_worker(self, wrapped):
        clone = pickle.loads(pickle.dumps(wrapped))
        assert clone.get_bucket(HAND, CARDS, Street.RIVER) == wrapped.get_bucket(
            HAND, CARDS, Street.RIVER
        )

    def test_the_resolver_wraps_on_request_and_only_then(self, tmp_path):
        abstractions_dir = tmp_path / "combo_abstraction"
        path = abstractions_dir / "buckets-ten"
        path.mkdir(parents=True)
        (path / "metadata.json").write_text(json.dumps({"config": {"config_name": "ten"}}))
        base = StubAbstraction()
        resolver = ComboAbstractionResolver(
            abstractions_dir=abstractions_dir,
            loader=lambda _: base,
        )
        assert resolver.load(abstraction_config="ten") is base
        wrapped = resolver.load(abstraction_config="ten", recall="path")
        assert isinstance(wrapped, PathRecallBucketer)
        assert wrapped.base is base

    def test_the_probe_configs_differ_only_in_recall(self):
        street = load_training_config("probe_ten")
        path = load_training_config("probe_ten_path")
        assert street.card_abstraction.recall == "street"
        assert path.card_abstraction.recall == "path"
        assert street.card_abstraction.config == path.card_abstraction.config == "ten"
        assert street.action_model == path.action_model
