"""Opponent Cluster Hand Strength: counting must be exact, not approximately right.

OCHS replaces scalar river equity with a per-opponent-cluster vector, which
means redoing win/tie counting with card removal eight times over subsets. That
is index arithmetic, and index arithmetic fails silently — a slightly wrong
blocker correction still produces plausible numbers in [0, 1] that cluster into
plausible buckets. So these tests pin it against two independent references:
the existing scalar engine, and brute-force pairwise enumeration.
"""

from __future__ import annotations

import eval7
import numpy as np
import pytest

from src.core.game.state import Card
from src.pipeline.abstraction.preflop.hand_classes import PreflopHandClasses
from src.pipeline.abstraction.preflop.opponent_clusters import (
    NUM_PREFLOP_CLASSES,
    opponent_cluster_assignment,
    preflop_class_equities,
    representative_combo,
)
from src.pipeline.abstraction.utils.equity import RangeEquityEngine, class_index_table

BOARD = tuple(Card.new(s) for s in ("Ah", "Kd", "7c", "2s", "9h"))
WET_BOARD = tuple(Card.new(s) for s in ("Th", "9h", "8h", "2d", "3c"))


@pytest.fixture(scope="module")
def engine():
    return RangeEquityEngine()


@pytest.fixture(scope="module")
def clusters():
    # Small sample: these tests check counting, not equity precision.
    return opponent_cluster_assignment(samples=800, cache_dir=None)


def _cluster_of(combo, clusters):
    table = class_index_table()
    a, b = (c.to_eval7() for c in combo)
    return clusters[table[a.rank * 4 + a.suit, b.rank * 4 + b.suit]]


class TestCountingIsExact:
    def test_single_cluster_reproduces_scalar_equity(self, engine):
        """With every class in one cluster, OCHS *is* equity-vs-uniform.

        The strongest available check: it must match the independently written
        scalar engine bit for bit, including card removal.
        """
        one_cluster = np.zeros(NUM_PREFLOP_CLASSES, dtype=np.int64)
        combos, ochs = engine.board_ochs(BOARD, one_cluster, 1)

        table = engine.board_equities(BOARD)
        scalar = {
            frozenset((a.mask, b.mask)): e
            for (a, b), e in zip(table.combos, table.equities, strict=True)
        }
        compared = 0
        for (card_a, card_b), value in zip(combos, ochs[:, 0], strict=True):
            reference = scalar[frozenset((card_a.mask, card_b.mask))]
            if np.isnan(reference):
                continue
            assert value == pytest.approx(reference, abs=1e-12)
            compared += 1
        assert compared > 1000, "expected ~1081 comparable combos"

    @pytest.mark.parametrize("board", [BOARD, WET_BOARD])
    def test_matches_brute_force_pairwise(self, engine, clusters, board):
        """Direct enumeration over every disjoint opponent combo, per cluster."""
        combos, ochs = engine.board_ochs(board, clusters, 8)
        board_e7 = [c.to_eval7() for c in board]
        strength = {
            (a.mask, b.mask): eval7.evaluate([*board_e7, a.to_eval7(), b.to_eval7()])
            for a, b in combos
        }
        cluster_of = [_cluster_of(c, clusters) for c in combos]

        rng = np.random.default_rng(0)
        for i in rng.choice(len(combos), size=12, replace=False):
            hero_a, hero_b = combos[i]
            hero_cards = {hero_a.mask, hero_b.mask}
            hero_strength = strength[(hero_a.mask, hero_b.mask)]
            for target in (0, 3, 7):
                wins = ties = total = 0
                for j, (opp_a, opp_b) in enumerate(combos):
                    if {opp_a.mask, opp_b.mask} & hero_cards:
                        continue
                    if cluster_of[j] != target:
                        continue
                    total += 1
                    opp_strength = strength[(opp_a.mask, opp_b.mask)]
                    if hero_strength > opp_strength:
                        wins += 1
                    elif hero_strength == opp_strength:
                        ties += 1
                expected = (wins + 0.5 * ties) / total if total else 0.0
                assert ochs[i, target] == pytest.approx(expected, abs=1e-12)

    def test_requires_a_complete_board(self, engine, clusters):
        with pytest.raises(ValueError, match="complete board"):
            engine.board_ochs(BOARD[:4], clusters, 8)

    def test_values_are_probabilities(self, engine, clusters):
        _, ochs = engine.board_ochs(BOARD, clusters, 8)
        assert ochs.shape[1] == 8
        assert (ochs >= 0.0).all() and (ochs <= 1.0).all()


class TestSeparatesWhatEquityCannot:
    def test_equal_equity_hands_can_differ_in_ochs(self, engine, clusters):
        """The reason OCHS exists, demonstrated on a real board.

        Hands with near-identical equity against a uniform range but different
        profiles against strong vs weak holdings are exactly what scalar
        bucketing merges and OCHS separates. If no such pair exists, the feature
        buys nothing and the whole change is pointless.
        """
        combos, ochs = engine.board_ochs(WET_BOARD, clusters, 8)
        table = engine.board_equities(WET_BOARD)
        scalar = {
            frozenset((a.mask, b.mask)): e
            for (a, b), e in zip(table.combos, table.equities, strict=True)
        }
        equities = np.array(
            [scalar[frozenset((a.mask, b.mask))] for a, b in combos], dtype=np.float64
        )

        order = np.argsort(equities)
        best_gap = 0.0
        for pos in range(len(order) - 1):
            i, j = order[pos], order[pos + 1]
            if abs(equities[i] - equities[j]) > 1e-4:
                continue
            best_gap = max(best_gap, float(np.abs(ochs[i] - ochs[j]).max()))

        assert best_gap > 0.05, (
            "no pair with equal scalar equity differs materially in OCHS; "
            f"largest per-cluster gap was {best_gap:.4f}"
        )


class TestOpponentClusters:
    def test_equities_are_plausible(self):
        equities = preflop_class_equities(samples=800, cache_dir=None)
        classes = PreflopHandClasses()
        index = {
            hand: classes.get_hand_index(representative_combo(hand))
            for hand in PreflopHandClasses.get_all_hands()
        }
        assert equities[index["AA"]] > equities[index["KK"]] > equities[index["22"]]
        assert equities[index["AA"]] > 0.8
        assert equities[index["32o"]] < 0.4
        assert equities[index["AKs"]] > equities[index["AKo"]]

    def test_clusters_are_ordered_weakest_first(self):
        clusters = opponent_cluster_assignment(num_clusters=8, samples=800, cache_dir=None)
        equities = preflop_class_equities(samples=800, cache_dir=None)
        means = [equities[clusters == c].mean() for c in range(8) if (clusters == c).any()]
        assert means == sorted(means), "cluster ids must increase with strength"

    def test_premium_pairs_land_in_the_top_cluster(self):
        clusters = opponent_cluster_assignment(num_clusters=8, samples=800, cache_dir=None)
        classes = PreflopHandClasses()
        top = clusters.max()
        assert clusters[classes.get_hand_index(representative_combo("AA"))] == top

    def test_every_cluster_is_used(self):
        clusters = opponent_cluster_assignment(num_clusters=8, samples=800, cache_dir=None)
        assert set(clusters.tolist()) == set(range(8))

    def test_assignment_is_deterministic(self):
        a = opponent_cluster_assignment(samples=800, cache_dir=None)
        b = opponent_cluster_assignment(samples=800, cache_dir=None)
        np.testing.assert_array_equal(a, b)
