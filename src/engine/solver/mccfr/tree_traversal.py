"""External-sampling MCCFR that walks node ids instead of game states.

The public betting line is CARD-INDEPENDENT: which node an action leads to, how
many board cards the dealer owes before the child acts, and what every terminal
pays are all fixed at enumeration time and recorded on :class:`Edge`. So this
walks integers -- per iteration exactly one ``GameState`` is built, the dealt
root the hole cards come from, and after that the traversal carries only
``(node_id, board, known-card mask)``. At production tree size the state
construction this avoids was 44% of an iteration and the string-keyed node
lookup another 4%.

Nothing about the RULES is re-derived here. The edge table is recorded from the
same ``GameRules.apply_action`` the state-based traversal called per visit, and
terminal payoffs are tabulated from the terminal states it produced. This module
reads that table; it does not model the game a second time.

Equivalence is a maintained property, not a claim:
``tests/engine/solver/mccfr/test_tree_traversal_equivalence.py`` runs both
traversals from one seed and requires the shared arrays to come out
bit-identical, which also pins the random stream.

Pruning is deliberately not implemented here -- it is off by default and
measured worse, and its masked paths would cost the fast path on every node.
:class:`StaticTreeSolver` keeps the generic traversal when it is enabled.
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np

from src.core.game.state import FULL_DECK
from src.engine.solver.infoset.index import preflop_hand_index
from src.engine.solver.numba_ops import (
    WEIGHTING_CODES,
    apply_regret_updates,
    compute_dcfr_strategy_weight,
    regret_matching,
)

from .traversal import identity_indices, sample_action_index

if TYPE_CHECKING:
    from collections.abc import Callable

    from src.core.game.state import Card, GameState, Street
    from src.engine.solver.betting_tree import TerminalOutcome

    from .static_solver import StaticTreeSolver


class _Walk:
    """Everything one iteration's traversal needs, resolved once.

    Built per iteration rather than per node: the tree handles, the storage
    arrays and the weighting constants never change inside an iteration, and
    the hole cards and button change only between them.
    """

    __slots__ = (
        "applied",
        "button",
        "cfr_plus",
        "compare_hands",
        "cumulative_utility",
        "dcfr_alpha",
        "dcfr_beta",
        "get_bucket",
        "hole",
        "iteration",
        "node_spec",
        "non_button",
        "preflop_bucket",
        "reach_counts",
        "regrets",
        "strategy_sum",
        "strategy_weight",
        "traverser_pays",
        "traversing_player",
        "visited",
        "weighting",
    )

    node_spec: list[tuple]
    regrets: np.ndarray
    strategy_sum: np.ndarray
    reach_counts: np.ndarray
    cumulative_utility: np.ndarray
    visited: np.ndarray
    get_bucket: Callable[[tuple[Card, Card], tuple[Card, ...], Street], int]
    compare_hands: Callable[[tuple[Card, Card], tuple[Card, Card], tuple[Card, ...]], int]

    iteration: int
    cfr_plus: bool
    weighting: int
    dcfr_alpha: float
    dcfr_beta: float
    strategy_weight: float

    hole: tuple[tuple[Card, Card], tuple[Card, Card]]
    button: int
    non_button: int
    traversing_player: int
    traverser_pays: int
    preflop_bucket: tuple[int, int]
    applied: int


def cfr_external_sampling(
    self: StaticTreeSolver,
    state: GameState,
    traversing_player: int,
) -> float:
    """One external-sampling iteration, starting from the dealt root state.

    Only the hole cards and the button are read off ``state``; the walk below
    is over node ids.
    """
    tree = self.tree
    storage = self.storage
    solver_config = self.config.solver
    weighting = solver_config.iteration_weighting

    walk = _Walk()
    walk.node_spec = tree.node_spec
    walk.regrets = storage.regrets
    walk.strategy_sum = storage.strategy_sum
    walk.reach_counts = storage.reach_counts
    walk.cumulative_utility = storage.cumulative_utility
    walk.visited = storage.visited
    walk.get_bucket = self.card_abstraction.get_bucket
    walk.compare_hands = self.rules.evaluator.compare_hands

    walk.iteration = self.iteration
    walk.cfr_plus = solver_config.cfr_plus
    walk.weighting = WEIGHTING_CODES[weighting]
    walk.dcfr_alpha = solver_config.dcfr_alpha
    walk.dcfr_beta = solver_config.dcfr_beta
    # The average-strategy weight is a function of the iteration alone, so the
    # per-opponent-node kernel call the generic traversal makes is one call here.
    if weighting == "dcfr":
        walk.strategy_weight = 1.0 * compute_dcfr_strategy_weight(
            self.iteration, solver_config.dcfr_gamma
        )
    elif weighting == "linear":
        walk.strategy_weight = 1.0 * self.iteration
    else:
        walk.strategy_weight = 1.0

    hole = state.hole_cards
    button = state.button_position
    walk.hole = hole
    walk.button = button
    walk.non_button = 1 - button
    walk.traversing_player = traversing_player
    # Payoff tables are stored button-relative; the traverser reads one column.
    walk.traverser_pays = 0 if traversing_player == button else 1
    # Pure in the hole cards, so the per-visit call the generic traversal makes
    # at every preflop node collapses to two.
    walk.preflop_bucket = (preflop_hand_index(hole[0]), preflop_hand_index(hole[1]))
    walk.applied = 0

    known = 0
    for player_cards in hole:
        for card in player_cards:
            known |= card.mask

    utility = _walk_node(walk, tree.root_id, (), known)
    self.applied_updates += walk.applied
    return utility


def _draw(known: int, count: int) -> tuple[tuple[Card, ...], int]:
    """Draw ``count`` unseen cards, and return the widened known-card mask.

    Rejection sampling against the mask, drawing in the same order and from the
    same ``random.randrange`` as ``chance.draw_cards`` — the traversal's random
    stream is part of what a run reproduces.
    """
    randrange = random.randrange
    drawn: list[Card] = []
    while len(drawn) < count:
        card = FULL_DECK[randrange(52)]
        if card.mask & known:
            continue
        known |= card.mask
        drawn.append(card)
    return tuple(drawn), known


def _terminal_value(
    walk: _Walk, terminal: TerminalOutcome, board: tuple[Card, ...], known: int
) -> float:
    """Payoff to the traverser at a hand that just ended.

    A fold is already decided by the betting line. A showdown owes whatever
    board the all-in outran, and then one hand comparison.
    """
    seat = walk.traverser_pays
    if terminal.is_fold:
        return terminal.fold[seat]

    if terminal.cards_to_deal:
        board = board + _draw(known, terminal.cards_to_deal)[0]

    hole = walk.hole
    result = walk.compare_hands(hole[0], hole[1], board)
    if result == 0:
        return terminal.tie[seat]
    winner = 0 if result == -1 else 1
    return terminal.win[seat] if winner == walk.traversing_player else terminal.lose[seat]


def _walk_node(walk: _Walk, node_id: int, board: tuple[Card, ...], known: int) -> float:
    """Recurse from one decision node, returning its utility to the traverser."""
    (
        is_preflop,
        actor_is_button,
        street,
        num_actions,
        row_base,
        slot_base,
        num_buckets,
        edges,
    ) = walk.node_spec[node_id]

    actor = walk.button if actor_is_button else walk.non_button
    if is_preflop:
        bucket = walk.preflop_bucket[actor]
    else:
        bucket = walk.get_bucket(walk.hole[actor], board, street)
        if not 0 <= bucket < num_buckets:
            # Rows are contiguous per node, so an out-of-range bucket does not
            # fall off the array — it lands on another node's infoset and the
            # two silently share storage. Only a custom or broken
            # BucketingStrategy can get here; the production one raises first.
            raise IndexError(
                f"bucket {bucket} out of range for node {node_id} "
                f"({street.name}, {num_buckets} buckets). "
                "An out-of-range bucket would alias another node's infoset."
            )

    row = row_base + bucket
    start = slot_base + bucket * num_actions
    end = start + num_actions
    walk.visited[row] = 1

    regrets = walk.regrets[start:end]
    strategy = regret_matching(regrets)

    if actor == walk.traversing_player:
        action_utilities = np.zeros(num_actions)
        for index in range(num_actions):
            edge = edges[index]
            terminal = edge.terminal
            if terminal is not None:
                action_utilities[index] = _terminal_value(walk, terminal, board, known)
                continue
            deal = edge.deal
            if deal:
                drawn, child_known = _draw(known, deal)
                action_utilities[index] = _walk_node(
                    walk, edge.child_id, board + drawn, child_known
                )
            else:
                action_utilities[index] = _walk_node(walk, edge.child_id, board, known)

        node_utility = float(np.dot(strategy, action_utilities))

        # Opponent reach is 1.0: under external sampling the opponent's actions
        # are sampled, so visit frequency already carries pi_{-i}. See
        # `traversal.cfr_external_sampling` for why passing it again is wrong.
        apply_regret_updates(
            regrets,
            identity_indices(num_actions),
            action_utilities,
            node_utility,
            1.0,
            walk.cfr_plus,
            walk.iteration,
            walk.weighting,
            walk.dcfr_alpha,
            walk.dcfr_beta,
        )
        walk.reach_counts[row] += 1
        walk.cumulative_utility[row] += node_utility
        walk.applied += 1
        return node_utility

    # Opponent node: where the average strategy accumulates, weighted by visit
    # frequency alone (again, pi_i is already in the visit frequency).
    strategy_sum = walk.strategy_sum
    weight = walk.strategy_weight
    for offset, probability in enumerate(strategy.tolist()):
        strategy_sum[start + offset] += probability * weight
    walk.applied += 1

    edge = edges[sample_action_index(strategy)]
    terminal = edge.terminal
    if terminal is not None:
        return _terminal_value(walk, terminal, board, known)
    deal = edge.deal
    if deal:
        drawn, child_known = _draw(known, deal)
        return _walk_node(walk, edge.child_id, board + drawn, child_known)
    return _walk_node(walk, edge.child_id, board, known)


__all__ = ("cfr_external_sampling",)
