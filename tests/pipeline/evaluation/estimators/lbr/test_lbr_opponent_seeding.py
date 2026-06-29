"""The opponent model's randomness must be pinned per hand, not per worker.

Regression test for the deployed-opponent LBR seeding bug: the resolver's RNG was
seeded once at engine construction -- from a ``base_seed`` identical across every
worker -- and never reseeded per hand. All workers therefore drew from the same
stream, and a hand's resolver randomness depended on its position within a chunk
(``chunksize = num_hands // (num_workers*4)``), so results were not reproducible
across worker counts despite ``LBRConfig.num_workers`` promising exactly that.

The invariant: the seed the opponent receives is a pure function of
``(base_seed, hand)``, and both seats of a pair receive the same one.
"""

from types import SimpleNamespace
from typing import TYPE_CHECKING, cast

import numpy as np

from src.pipeline.evaluation.estimators.lbr import hunl_local_best_response as lbr

if TYPE_CHECKING:
    from src.pipeline.evaluation.estimators.lbr.hunl_local_best_response import (
        _HUNLLocalBestResponse,
    )


class _RecordingOpponent:
    """Captures reseed calls; stands in for the model under measurement."""

    def __init__(self) -> None:
        self.seeds: list[int] = []

    def reseed(self, seed: int) -> None:
        self.seeds.append(seed)


def _play(hand: int, base_seed: int, monkeypatch) -> list[int]:
    """Run one hand pair with stubbed play and return the seeds the opponent saw."""
    opponent = _RecordingOpponent()
    engine = SimpleNamespace(
        rng=np.random.default_rng(0),
        opponent=opponent,
        play_hand=lambda seat, state: SimpleNamespace(seat=seat),
    )
    # The real engine rebinds its RNG and its collaborators' through reseed();
    # the double has to offer the same seam or it is testing a different object.
    engine.reseed = lambda rng: setattr(engine, "rng", rng)
    monkeypatch.setattr(lbr, "_deal_initial_state", lambda *_a, **_k: object())
    lbr._play_hand_pair(cast("_HUNLLocalBestResponse", engine), hand, base_seed, starting_stack=200)
    return opponent.seeds


def test_both_seats_of_a_pair_share_one_opponent_seed(monkeypatch):
    """The seat swap is a paired sample; opponent noise must not differ across it."""
    seeds = _play(hand=7, base_seed=1, monkeypatch=monkeypatch)
    assert len(seeds) == 2, "opponent must be reseeded once per seat"
    assert seeds[0] == seeds[1]


def test_opponent_seed_depends_only_on_base_seed_and_hand(monkeypatch):
    """Worker count and chunking cannot influence the seed, so evals reproduce."""
    first = _play(hand=7, base_seed=1, monkeypatch=monkeypatch)
    again = _play(hand=7, base_seed=1, monkeypatch=monkeypatch)
    assert first == again


def test_distinct_hands_draw_distinct_opponent_streams(monkeypatch):
    """The bug's signature was one shared stream; every hand must differ."""
    seeds = {_play(hand=h, base_seed=1, monkeypatch=monkeypatch)[0] for h in range(64)}
    assert len(seeds) == 64


def test_opponent_stream_is_independent_of_the_deal_stream():
    """The opponent must not be correlated with the deal it is responding to."""
    for hand in range(32):
        assert lbr._opponent_hand_seed(1, hand) != lbr._hand_seed(1, hand)


def test_resolved_opponent_reseed_replaces_the_resolver_generator():
    """reseed() must reach the generator solve_subgame actually consumes."""
    from src.pipeline.evaluation.estimators.lbr.opponent_model import ResolvedOpponent

    opponent = object.__new__(ResolvedOpponent)
    opponent._resolver = SimpleNamespace(rng=None)  # ty: ignore[invalid-assignment]

    opponent.reseed(12345)
    drawn_first = opponent._resolver.rng.random()

    opponent.reseed(12345)
    assert opponent._resolver.rng.random() == drawn_first
