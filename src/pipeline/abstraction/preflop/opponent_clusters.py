"""Opponent hand clusters for Opponent Cluster Hand Strength (OCHS).

OCHS (Johanson, Burch, Valenzano & Bowling, AAMAS 2013) replaces river
expected-hand-strength-against-a-uniform-range with a VECTOR of win rates, one
per cluster of opponent holdings, because scalar equity cannot express WHICH
part of the opponent's range a hand beats: a bluff-catcher and a mediocre made
hand can share an equity number while wanting opposite strategies.

This module owns only the opponent partition -- which holdings count as cluster
c. The per-board win-rate vectors live in
:mod:`src.pipeline.abstraction.utils.equity`.

Clusters are the 169 preflop hand classes grouped by preflop all-in equity
against a random hand: a fixed, game-level notion of "kind of holding" that does
not depend on the board being evaluated.

Determinism. Preflop equity has no tractable exact enumeration, so it is
estimated by Monte Carlo with a fixed seed and sample count, cached on disk and
keyed by BOTH -- a moving opponent partition would silently change every OCHS
feature and therefore every bucket.

sklearn is imported at its call site: ~0.5s that every ``poker-solver``
invocation was paying for a clusterer it never calls. The guard is
``tests/interfaces/test_import_weight.py``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import eval7
import numpy as np

from src.core.game.state import Card
from src.pipeline.abstraction.preflop.hand_classes import PreflopHandClasses
from src.shared.cache import cache_dir

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = cache_dir("preflop_equity")
DEFAULT_SAMPLES = 20_000
DEFAULT_SEED = 42
NUM_PREFLOP_CLASSES = 169

# 8 is the count used in the originating work. It is a resolution knob: more
# clusters describe the opponent range more finely at linear cost in the feature
# width that k-means then has to separate.
DEFAULT_NUM_CLUSTERS = 8

_SUITS = "shdc"


def representative_combo(hand_string: str) -> tuple[Card, Card]:
    """A concrete two-card combo standing in for a canonical class.

    Suits are chosen to realise the class: pairs and offsuit hands take
    different suits, suited hands take the same one. Which specific suits is
    irrelevant — preflop equity is suit-symmetric.
    """
    high, low = hand_string[0], hand_string[1]
    suited = hand_string.endswith("s")
    if high == low:
        return Card.new(f"{high}{_SUITS[0]}"), Card.new(f"{low}{_SUITS[1]}")
    if suited:
        return Card.new(f"{high}{_SUITS[0]}"), Card.new(f"{low}{_SUITS[0]}")
    return Card.new(f"{high}{_SUITS[0]}"), Card.new(f"{low}{_SUITS[1]}")


def _simulate_class_equity(
    hero: tuple[Card, Card], deck: list[eval7.Card], rng: np.random.Generator, samples: int
) -> float:
    """Monte-Carlo all-in equity for one class against a uniform random hand."""
    hero_e7 = [c.to_eval7() for c in hero]
    hero_masks = {c.mask for c in hero}
    available = [c for c in deck if c.mask not in hero_masks]
    n = len(available)

    evaluate = eval7.evaluate
    score = 0.0
    for _ in range(samples):
        # 2 opponent cards + 5 board cards, drawn without replacement.
        idx = rng.choice(n, size=7, replace=False)
        opp = [available[idx[0]], available[idx[1]]]
        board = [available[i] for i in idx[2:]]
        hero_value = evaluate(hero_e7 + board)
        opp_value = evaluate(opp + board)
        if hero_value > opp_value:
            score += 1.0
        elif hero_value == opp_value:
            score += 0.5
    return score / samples


def preflop_class_equities(
    *,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> np.ndarray:
    """All-in equity vs a random hand for each of the 169 classes, in class-index order."""
    cache_path = None
    if cache_dir is not None:
        cache_path = Path(cache_dir) / f"preflop_equity_s{samples}_seed{seed}.npy"
        if cache_path.exists():
            return np.load(cache_path)

    classes = PreflopHandClasses()
    hands = PreflopHandClasses.get_all_hands()
    deck = eval7.Deck().cards

    equities = np.zeros(NUM_PREFLOP_CLASSES, dtype=np.float64)
    for hand_string in hands:
        combo = representative_combo(hand_string)
        # Seed per class, not per run: the estimate for "AKs" must not depend on
        # how many classes were evaluated before it.
        rng = np.random.default_rng([seed, classes.get_hand_index(combo)])
        equities[classes.get_hand_index(combo)] = _simulate_class_equity(combo, deck, rng, samples)

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, equities)
        logger.info(f"Cached preflop class equities to {cache_path}")
    return equities


def opponent_cluster_assignment(
    *,
    num_clusters: int = DEFAULT_NUM_CLUSTERS,
    samples: int = DEFAULT_SAMPLES,
    seed: int = DEFAULT_SEED,
    cache_dir: Path | None = DEFAULT_CACHE_DIR,
) -> np.ndarray:
    """Map each preflop class index to an opponent cluster, weakest cluster first.

    Ordering by ascending equity is what makes an OCHS vector readable and
    comparable across boards: component 0 is always "win rate against the
    weakest holdings", component ``num_clusters - 1`` always "against the
    strongest". Unordered cluster ids would make the L2 distance between two
    vectors meaningless.
    """
    equities = preflop_class_equities(samples=samples, seed=seed, cache_dir=cache_dir)

    # Measured 0.85s to import `sklearn.cluster`, and only the FITTING paths
    # need it -- reading a built abstraction does not. Hoisting it would put
    # that second on `poker-solver --help` and on every node task, to serve the
    # one call below.
    from sklearn.cluster import KMeans  # noqa: PLC0415 -- 0.85s import, fit-only

    kmeans = KMeans(n_clusters=num_clusters, n_init=10, random_state=seed)
    labels = kmeans.fit_predict(equities.reshape(-1, 1))

    # Relabel so cluster index increases with mean equity.
    centers = kmeans.cluster_centers_.ravel()
    order = np.argsort(centers)
    remap = np.zeros(num_clusters, dtype=np.int64)
    remap[order] = np.arange(num_clusters)
    return remap[labels].astype(np.int64)


__all__ = (
    "DEFAULT_NUM_CLUSTERS",
    "NUM_PREFLOP_CLASSES",
    "opponent_cluster_assignment",
    "preflop_class_equities",
    "representative_combo",
)
