"""How much the average strategy still MOVES, per street.

Exploitability says how far a strategy is from equilibrium; it needs a best
response and a board plan. This says something cheaper and different: whether
the strategy has stopped changing. Two rungs of one run, the total-variation
distance between their average strategies, per row, summarised per street.

THREE THINGS MAKE THE NAIVE VERSION LIE, and each has a column here.

1. A PURE ROW'S DRIFT IS EXACTLY ZERO -- it is frozen, not settled. Measured
   09-24: the bottom entropy decile of flop/turn/river rows moves 0.00000.
2. A row reached almost never is the FIRST to freeze, because regret matching
   locks it onto whichever action won its handful of traversals. So the
   unweighted median is flattered by millions of rows nobody visits: on the
   river it read 1.7-2.9x lower than the reach-weighted figure.
3. Drift falling because rows are GOING pure is not convergence. `pure` is
   reported beside the drift so the two are never confused.

Read it against the exploitability of the same rungs. Drift falling while
exploitability holds means the strategy is settling on a bad answer, which more
iterations will not fix.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from src.engine.solver.storage.static_array import StaticArrayStorage

# A row is PURE when one action holds at least this much. Same cutoff the policy
# profile uses, so the two readings of one checkpoint agree.
PURE_CUTOFF = 0.99
# "Thin" is the bottom quartile of reach on that street. A quartile rather than a
# fixed count because streets differ by four orders of magnitude in row count.
THIN_QUANTILE = 25.0


@dataclass
class StreetSettling:
    """One street's movement between two rungs."""

    rows: int
    median_tv: float
    reach_weighted_tv: float
    p90_tv: float
    pure_fraction: float
    thin_pure_fraction: float
    # Share of the street's total reach carried by its thinnest quartile. When
    # this is ~0 the frozen thin rows cost little in SELF-PLAY -- it says nothing
    # about what a best responder steers into.
    thin_reach_share: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "rows": self.rows,
            "median_tv": self.median_tv,
            "reach_weighted_tv": self.reach_weighted_tv,
            "p90_tv": self.p90_tv,
            "pure_fraction": self.pure_fraction,
            "thin_pure_fraction": self.thin_pure_fraction,
            "thin_reach_share": self.thin_reach_share,
        }


@dataclass
class SettlingResult:
    """Per-street settling between ``from_iteration`` and ``at_iteration``."""

    from_iteration: int
    at_iteration: int
    streets: dict[str, StreetSettling] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "from_iteration": self.from_iteration,
            "at_iteration": self.at_iteration,
            "by_street": {name: s.as_dict() for name, s in self.streets.items()},
        }


def _normalised(rows: np.ndarray) -> np.ndarray:
    """Rows to distributions; a zero row is uniform, as the average strategy has it."""
    totals = rows.sum(axis=1, keepdims=True)
    uniform = np.full_like(rows, 1.0 / rows.shape[1])
    return np.divide(rows, totals, out=uniform, where=totals > 0)


def _by_node(storage: StaticArrayStorage) -> dict[int, tuple[str, np.ndarray, np.ndarray]]:
    """Per node: its street, its visited rows' average strategy, and their reach.

    KEYED BY NODE, never stacked per street: action counts differ node to node
    (six at the opening, four facing a raise), so a street's rows only pair up
    node-for-node. Reach is the UN-NORMALISED row sum, which is the accumulated
    weight that row received -- `storage.visited` is a boolean flag and carries
    no count at all.
    """
    tree = storage.tree
    out: dict[int, tuple[str, np.ndarray, np.ndarray]] = {}
    for node in tree.nodes:
        visited = tree.node_row_vector(storage.visited, node.node_id).astype(bool)
        if not visited.any():
            continue
        raw = tree.node_action_matrix(storage.strategy_sum, node.node_id)[visited]
        raw = raw.astype(np.float64)
        out[node.node_id] = (str(node.street), _normalised(raw), raw.sum(axis=1))
    return out


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """The value below which half the WEIGHT sits."""
    total = weights.sum()
    if total <= 0:
        return float("nan")
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(cumulative, total / 2.0)])


def measure_settling(
    early: StaticArrayStorage,
    late: StaticArrayStorage,
    *,
    from_iteration: int,
    at_iteration: int,
) -> SettlingResult:
    """Per-street drift between two loaded rungs of one run."""
    a, b = _by_node(early), _by_node(late)
    pooled: dict[str, dict[str, list[np.ndarray]]] = {}
    for node_id, (street, rows_a, _reach_a) in a.items():
        mate = b.get(node_id)
        if mate is None:
            continue
        _street_b, rows_b, reach_b = mate
        if rows_b.shape != rows_a.shape:
            continue
        bucket = pooled.setdefault(street, {"tv": [], "reach": [], "pure": []})
        bucket["tv"].append(0.5 * np.abs(rows_a - rows_b).sum(axis=1))
        bucket["reach"].append(reach_b)
        bucket["pure"].append(rows_b.max(axis=1) >= PURE_CUTOFF)

    result = SettlingResult(from_iteration=from_iteration, at_iteration=at_iteration)
    for street, chunks in pooled.items():
        tv = np.concatenate(chunks["tv"])
        reach = np.concatenate(chunks["reach"])
        pure = np.concatenate(chunks["pure"])
        thin_cut = float(np.percentile(reach, THIN_QUANTILE))
        thin = reach <= thin_cut
        total_reach = float(reach.sum())
        result.streets[street] = StreetSettling(
            rows=int(tv.size),
            median_tv=float(np.median(tv)),
            reach_weighted_tv=_weighted_median(tv, reach),
            p90_tv=float(np.percentile(tv, 90)),
            pure_fraction=float(pure.mean()),
            thin_pure_fraction=float((pure & thin).mean()),
            thin_reach_share=(
                float(reach[thin].sum() / total_reach) if total_reach > 0 else float("nan")
            ),
        )
    return result
