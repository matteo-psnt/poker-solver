"""The strategy at a node, for every hand a player can hold.

The solver reasons in *buckets* -- a few hundred per street. A person reads
1326 combos. That gap is most of why a trained blueprint is hard to look at, and
closing it is what this module does: bucket every combo the board allows, read
the strategy once per bucket, and hand back a row per combo.

Buckets are the unit, and that is worth seeing
----------------------------------------------
The strategy at a node depends on the acting player's *bucket*, never on their
exact cards -- the only use of the concrete combo is to find the bucket. So every
combo sharing a bucket carries an identical row, by construction rather than by
coincidence, and this module computes one row per bucket rather than per combo
for exactly that reason. It is ~200x less work than 1326 lookups, and it is also
the honest shape: where a rendered grid goes flat is precisely where the
abstraction stopped telling hands apart. That is a property of the solver worth
showing, not an artifact worth smoothing over.

Untrained is not the same as uniform
------------------------------------
On the static backend every row is allocated before training starts, so an
unvisited infoset would answer "uniform over the legal actions" if simply read --
a confident-looking strategy the solver never learned. ``TreePolicySource``
already refuses that (it returns ``None`` for an unvisited row), and this module
carries the refusal outward as ``trained=False`` with no strategy at all, rather
than substituting the uniform that a caller would have no way to recognise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.core.game.actions import Action
from src.core.game.state import Card, GameState
from src.engine.search.range_inference import (
    ALL_COMBOS,
    COMBO_MASKS,
    NUM_COMBOS,
    replace_actor_hole_cards,
)
from src.engine.solver.policy_lookup import blueprint_action_distribution
from src.engine.solver.policy_source import ScorableBlueprint
from src.pipeline.analysis.paths import PathError, ReplayedNode, encode_action


@dataclass(frozen=True)
class BucketStrategy:
    """What the blueprint plays at one node, holding one bucket.

    ``strategy`` is ``None`` exactly when ``trained`` is false, so there is no
    representable state in which a caller reads a strategy the solver never
    learned. ``reach_count`` is how many times training visited this infoset --
    the number to weigh a row by, and small enough on most rows to be the real
    story about a blueprint.
    """

    bucket: int
    trained: bool
    strategy: tuple[float, ...] | None
    reach_count: int


@dataclass(frozen=True)
class StrategyGrid:
    """Every combo's strategy at one node, plus what it took to get there.

    There is deliberately no node id here. ``PolicySource`` promises only a
    ``Hashable`` identity, and a node id is a position in a layout that a retrain
    reshuffles -- the very reason a spot is named by its path. The path the
    caller replayed is the identifier; nothing else needs to be.

    ``combo_buckets`` is indexed by position in
    :data:`~src.engine.search.range_inference.ALL_COMBOS` and holds ``-1`` where
    the board blocks the combo, so a renderer can grey those cells rather than
    dropping them and silently shifting the grid.
    """

    street: str
    board: tuple[Card, ...]
    actor: int
    actions: tuple[str, ...]
    combo_buckets: tuple[int, ...]
    buckets: dict[int, BucketStrategy]
    blocked: int

    @property
    def trained_buckets(self) -> int:
        """How many distinct buckets here the blueprint actually visited."""
        return sum(1 for entry in self.buckets.values() if entry.trained)

    def for_combo(self, combo_index: int) -> BucketStrategy | None:
        """The row for one combo, or ``None`` when the board blocks it."""
        bucket = self.combo_buckets[combo_index]
        return None if bucket < 0 else self.buckets[bucket]


def strategy_grid(
    blueprint: ScorableBlueprint,
    node: ReplayedNode,
    *,
    use_average: bool = True,
) -> StrategyGrid:
    """Per-combo strategy at ``node``.

    ``use_average`` selects the average strategy -- the blueprint proper, and
    what converges -- over the regret-matched current strategy. They diverge
    sharply on an under-trained run, which is why this is explicit at every call
    rather than defaulted somewhere out of sight.

    Raises :class:`PathError` at a terminal node: there is no strategy at a spot
    where nobody acts, and a grid of empty rows would be a worse answer than a
    refusal.
    """
    if node.actor is None:
        raise PathError("This line ends the hand, so there is no strategy to show.")

    actor = node.actor
    state = node.state
    legal = node.legal_actions
    source = blueprint.policy_source

    dead = 0
    for card in state.board:
        dead |= card.mask
    blocked_mask = (COMBO_MASKS & dead) != 0

    combo_buckets: list[int] = []
    buckets: dict[int, BucketStrategy] = {}
    for index in range(NUM_COMBOS):
        if blocked_mask[index]:
            combo_buckets.append(-1)
            continue
        with_combo = replace_actor_hole_cards(state, actor=actor, combo=ALL_COMBOS[index])
        bucket = source.bucket_for(with_combo, actor)
        combo_buckets.append(bucket)
        if bucket not in buckets:
            buckets[bucket] = _read_bucket(blueprint, with_combo, bucket, legal, use_average)

    return StrategyGrid(
        street=str(state.street),
        board=tuple(state.board),
        actor=actor,
        actions=tuple(encode_action(action) for action in legal),
        combo_buckets=tuple(combo_buckets),
        buckets=buckets,
        blocked=int(np.count_nonzero(blocked_mask)),
    )


def _read_bucket(
    blueprint: ScorableBlueprint,
    state: GameState,
    bucket: int,
    legal: tuple[Action, ...],
    use_average: bool,
) -> BucketStrategy:
    """One bucket's row.

    ``state`` is any state whose acting combo falls in ``bucket``; which one does
    not matter, because the lookup is keyed on the bucket and the only other
    thing consulted -- whether each action is affordable -- is a function of the
    chip configuration the whole node shares.
    """
    infoset = blueprint.policy_source.infoset_at(state, bucket)
    if infoset is None:
        return BucketStrategy(bucket=bucket, trained=False, strategy=None, reach_count=0)

    distribution = blueprint_action_distribution(
        infoset, state, blueprint.rules, legal, use_average=use_average
    )
    if distribution is None:
        return BucketStrategy(bucket=bucket, trained=False, strategy=None, reach_count=0)

    return BucketStrategy(
        bucket=bucket,
        trained=True,
        strategy=tuple(float(distribution.get(action, 0.0)) for action in legal),
        reach_count=int(infoset.reach_count),
    )
