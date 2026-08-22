"""A sensible opening guess for a row nobody has visited yet.

An untouched infoset plays UNIFORM: every legal action equally likely. As poker
that is not a neutral starting point, it is a bad one -- fold a third of the
time, jam a third of the time, regardless of what you hold. And it is not a
corner case. A cold run had 15.5% of rows still unvisited at 5M iterations and
~0.8% at 100M, with a long tail beyond that visited two or three times, which is
uniform in all but name.

This supplies the fallback instead: play toward the aggression your hand
strength justifies. Strong hands lean to betting and raising, weak ones to
checking and folding, and everything in between sits in between.

It reaches the table through two INDEPENDENT channels, and seeding one does not
seed the other. Regrets steer where training goes; ``strategy_sum`` is what
evaluation actually plays. A prior written only into regrets never plays on a
row training does not reach -- exactly the rows it exists for.

Two facts make it free to compute -- no new data, no extra pass:

``bucket index IS hand strength``. The abstraction fits its clusters with
``order_by="mean"`` (river) and ``order_by="cdf"`` (flop/turn), so bucket 0 is
the weakest class on that street and the last bucket the strongest. The row
index already encodes what this needs.

``the action menu is already ordered``. fold < check/call < small bet < large
bet < all-in is a total order on aggression, and every node's legal actions can
be placed on it from their type and size alone.

Deliberately NOT a strategy anyone should ship. It is an opening guess whose
whole job is to be less wrong than uniform on rows training has not reached, and
to be overwritten wherever it has.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action, ActionType
from src.core.game.rules import GameRules
from src.engine.solver.betting_tree import BettingTree, build_betting_tree
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.engine.solver.storage.static_checkpoint import save_checkpoint
from src.pipeline.blueprint import construction

if TYPE_CHECKING:
    from src.shared.config import Config

DEFAULT_TEMPERATURE = 0.25
"""How sharply the guess commits. Low sharpens toward one action; high flattens
toward uniform, which is the thing being replaced. Chosen to be soft: this is a
prior, and a confident wrong prior is worse than a vague one -- the flat-vs-
confidence result already showed that damping the prior's own uncertainty hurt.
"""


def action_aggression(actions: tuple[Action, ...], pot: float) -> np.ndarray:
    """Place each legal action on a passive-to-aggressive axis in ``[0, 1]``.

    Fold is 0 and all-in is 1 by definition. Bets and raises land between by
    size relative to the pot, capped at one pot-sized bet so a 3x pot jam and a
    1.2x pot bet are not squashed together at the top.
    """
    scores = np.empty(len(actions), dtype=np.float64)
    for i, action in enumerate(actions):
        match action.type:
            case ActionType.FOLD:
                scores[i] = 0.0
            case ActionType.CHECK | ActionType.CALL:
                scores[i] = 0.35
            case ActionType.ALL_IN:
                scores[i] = 1.0
            case _:
                # BET.amount is the total bet; RAISE.amount is the chips ABOVE
                # the call, so this is "how much is being put in on top" in both
                # cases rather than a single consistent quantity. Good enough
                # for an ordering -- bigger is more aggressive either way -- and
                # deliberately not dressed up as a pot fraction it is not.
                added = (action.amount or 0.0) / pot if pot > 0 else 0.5
                scores[i] = 0.5 + 0.4 * min(added, 1.0)
    return scores


def strength_policy(
    strength: float, aggression: np.ndarray, temperature: float = DEFAULT_TEMPERATURE
) -> np.ndarray:
    """The guess for one row: mass on the actions this strength justifies.

    ``softmax(-|strength - aggression| / temperature)``. A hand at strength 0.9
    puts its mass near the aggressive end, one at 0.1 near fold and check, and
    the distance form means the choice degrades gracefully rather than snapping
    between actions as strength crosses a boundary.
    """
    if aggression.size == 0:
        return aggression
    logits = -np.abs(float(strength) - aggression) / max(temperature, 1e-6)
    logits -= logits.max()
    weights = np.exp(logits)
    return weights / weights.sum()


def bucket_strength(bucket: int, num_buckets: int) -> float:
    """Bucket index as a strength in ``[0, 1]``; the midpoint of its band.

    Midpoint rather than ``bucket / (n - 1)`` so the extreme buckets are not
    treated as certainties -- the strongest river bucket is a strong class, not
    the nuts, and a prior that says otherwise is wrong in exactly the spots that
    cost the most.
    """
    if num_buckets <= 1:
        return 0.5
    return (bucket + 0.5) / num_buckets


__all__ = (
    "DEFAULT_TEMPERATURE",
    "action_aggression",
    "bucket_strength",
    "build_tree",
    "seed_regrets",
    "strength_policy",
    "tree_policy",
    "tree_regrets",
    "write_checkpoint",
)


def seed_regrets(tree: BettingTree, weight: float, temperature: float = DEFAULT_TEMPERATURE):
    """Regrets over the WHOLE tree encoding the strength-aware opening guess.

    Every row, not only the ones some other solver reached: the point is to
    replace uniform everywhere, including the rows training will never visit.

    Returns ``(regrets, seeded)`` in the shape ``StaticArrayStorage`` holds, so
    this drops into the same checkpoint path a warm start uses.
    """
    regrets = np.zeros(tree.num_slots, dtype=np.float64)
    seeded = np.zeros(tree.num_rows, dtype=bool)
    for node_id, node in enumerate(tree.nodes):
        width = int(tree.num_actions[node_id])
        if width == 0:
            continue
        # Pot is not stored per node; the aggression axis only needs a scale for
        # bet sizes, and the starting stack is the one scale every node shares.
        axis = action_aggression(node.legal_actions, float(tree.starting_stack))
        buckets = int(tree.buckets_per_node[node_id])
        base = int(tree.slot_offset[node_id])
        row0 = int(tree.row_offset[node_id])
        for bucket in range(buckets):
            policy = strength_policy(bucket_strength(bucket, buckets), axis, temperature)
            start = base + bucket * width
            regrets[start : start + width] = policy * weight
            seeded[row0 + bucket] = True
    return regrets, seeded


def build_tree(config: Config) -> BettingTree:
    """The tree the guess is laid out over.

    Public because building it reloads the card abstraction, which is the
    expensive part of seeding: a caller wanting regrets, a fallback and a
    checkpoint should build ONCE and pass it back in. Three implicit builds cost
    ~50 minutes of setup before iteration 1.
    """
    return build_betting_tree(
        GameRules(config.game.small_blind, config.game.big_blind),
        ActionModel(config),
        construction.build_card_abstraction(config),
        starting_stack=config.game.starting_stack,
    )


def tree_policy(
    config: Config,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    tree: BettingTree | None = None,
) -> np.ndarray:
    """The guess as a per-row probability distribution -- every row sums to 1.

    The one quantity worth computing: regrets are this times a weight and the
    fallback is this times a mass. They land in different arrays -- regrets steer
    TRAINING, ``strategy_sum`` decides what is PLAYED -- so seeding one does not
    seed the other, but both describe this single guess.
    """
    regrets, _ = seed_regrets(tree if tree is not None else build_tree(config), 1.0, temperature)
    return regrets


def tree_regrets(
    config: Config,
    *,
    weight: int,
    temperature: float = DEFAULT_TEMPERATURE,
    tree: BettingTree | None = None,
) -> np.ndarray:
    """The guess as a regret vector, without writing anything.

    Separate from the write so a warm start can take it as a BASE and add its
    own confidence on top, rather than the two racing to write iteration 0.
    """
    if weight <= 0:
        raise ValueError("equity prior weight must be positive; it scales the guess.")
    return tree_policy(config, temperature=temperature, tree=tree) * float(weight)


def write_checkpoint(
    config: Config,
    *,
    run_dir,
    regrets: np.ndarray,
    abstraction_hash: str | None,
    fallback: np.ndarray | None = None,
    tree: BettingTree | None = None,
) -> int:
    """Write an iteration-0 checkpoint holding ``regrets``.

    Needs no source run: unlike a warm start this is computed from the tree and
    the abstraction alone, so it applies to a COLD run -- which is the point,
    since the rows it is meant to help are the ones nothing has reached.

    ``fallback`` seeds ``strategy_sum``, which is the ONLY array evaluation
    reads: `average_strategy` normalises `strategy_sum` and returns uniform when
    it sums to zero, never consulting regrets. Without it the guess reaches the
    scoreboard only by steering training, and a row training never touches plays
    uniform however good the prior there was. Size it far below the mass real
    training accumulates, so it decides untouched rows and nothing else.
    """
    tree = tree if tree is not None else build_tree(config)
    per_row = np.repeat(tree.num_actions, tree.buckets_per_node)
    starts = np.zeros(tree.num_rows, dtype=np.int64)
    np.cumsum(per_row[:-1], out=starts[1:])

    storage = StaticArrayStorage(tree)
    storage.regrets[:] = np.asarray(regrets).astype(storage.regrets.dtype)
    if fallback is not None:
        storage.strategy_sum[:] = np.asarray(fallback).astype(storage.strategy_sum.dtype)
    # `visited` gates whether a row answers AT ALL, so it must track playable
    # policy -- strategy_sum -- and not regret mass. Marking a regret-only row
    # visited makes it answer uniform while reporting itself covered, which is
    # how the fallback-mass diagnostic came to read ~0% on a table that was
    # playing uniform everywhere the prior was meant to help.
    playable = np.add.reduceat(np.asarray(storage.strategy_sum, dtype=np.float64), starts) > 0.0
    storage.visited[:] = playable
    save_checkpoint(storage, run_dir, iteration=0, abstraction_id=abstraction_hash)
    return int(playable.sum())
