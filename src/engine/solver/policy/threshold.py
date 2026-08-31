"""The eval-time policy threshold, applied to a whole stored table at once.

Small probabilities in an average strategy are mostly the residue of early
iterations that never got averaged away. Zeroing them and renormalising measured
940.1 -> 854.0 mbb/hand on the programme gate (three seeds, `--policy-threshold
0.02`); 0.10 measured WORSE, so the knob is not monotone and 0.02 is the value
that was verified.

Why here rather than at lookup: the deployed player reaches the blueprint through
the resolver, which reads it for leaf values and range inference as well as for
the fall-through decision. Transforming the table once at load reaches all three;
transforming at one call site reaches one of them.

`transform_policy_rows` in `pipeline/evaluation/estimators/public_tree_br.py` is
the same transform on a dense per-state block, and `test_threshold.py` pins the
two against each other. They differ in one way that cannot be closed: that one
thresholds the distribution over the actions LEGAL at a state, this one over
every action stored in the row. The two agree whenever the stored list is legal
in full, which is every node of a static tree built from the same rules.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src.engine.solver.betting_tree import BettingTree
    from src.engine.solver.storage.static_array import StaticArrayStorage


def threshold_strategy_sum(
    strategy_sum: np.ndarray,
    visited: np.ndarray,
    row_starts: np.ndarray,
    row_widths: np.ndarray,
    threshold: float,
) -> int:
    """Zero every stored action below ``threshold`` of its row, in place.

    Returns the number of rows changed. Untrained rows (``visited == 0``) and
    rows with no mass are left alone: their uniform fallback is what the
    blueprint actually plays there, and purifying it would field "always the
    first action" -- which is fold.

    A row whose every action is below the cut keeps its largest, so no row is
    ever emptied. Ties keep every tied action, which normalises to the same
    distribution the argmax convention would give when the tie is genuine.
    """
    if threshold <= 0.0:
        return 0
    totals = np.add.reduceat(strategy_sum, row_starts)
    row_max = np.maximum.reduceat(strategy_sum, row_starts)
    trained = visited.astype(bool)
    live = trained & (totals > 0.0)

    # Below the cut for EVERY action, so thresholding alone would empty the row.
    cut = totals * np.float32(threshold)
    emptying = live & (row_max < cut)

    per_slot_cut = np.repeat(cut, row_widths)
    drop = np.repeat(live, row_widths) & (strategy_sum < per_slot_cut)
    del per_slot_cut
    keep = np.repeat(emptying, row_widths) & (strategy_sum == np.repeat(row_max, row_widths))
    drop &= ~keep
    del keep

    changed = int(np.count_nonzero(np.add.reduceat(drop.view(np.uint8), row_starts)))
    strategy_sum[drop] = 0.0
    return changed


def apply_policy_threshold(
    storage: StaticArrayStorage, tree: BettingTree, threshold: float
) -> tuple[int, int]:
    """Threshold a loaded blueprint's average strategy. ``(rows changed, rows trained)``."""
    changed = threshold_strategy_sum(
        storage.strategy_sum,
        storage.visited,
        tree.row_slot_starts,
        tree.row_widths,
        threshold,
    )
    return changed, int(np.count_nonzero(storage.visited))
