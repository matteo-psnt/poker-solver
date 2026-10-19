import numpy as np

from src.pipeline.evaluation.estimators.restricted_cfr import board_pool


def test_a_pool_is_flops_by_turns_groups_of_distinct_rivers():
    groups = board_pool(np.random.default_rng(7), flops=3, turns=2, rivers=4)
    assert len(groups) == 3 * 2
    assert all(len(group) == 4 for group in groups)
    for group in groups:
        assert len({tuple(b[:4]) for b in group}) == 1, "a group shares its flop and turn"
        assert len({int(b[4]) for b in group}) == 4, "and differs only in the river"
        assert all(len(set(b.tolist())) == 5 for b in group), "no card twice on a board"
    assert len({frozenset(g[0][:3].tolist()) for g in groups}) == 3


def test_a_pool_is_pinned_by_its_seed():
    a = board_pool(np.random.default_rng(7), flops=2, turns=2, rivers=4)
    b = board_pool(np.random.default_rng(7), flops=2, turns=2, rivers=4)
    assert all(
        np.array_equal(x, y)
        for ga, gb in zip(a, b, strict=True)
        for x, y in zip(ga, gb, strict=True)
    )
