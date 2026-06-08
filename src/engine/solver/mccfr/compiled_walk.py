"""The external-sampling traversal, compiled.

Assembly of four pieces that were each verified against what they replace
before anything was wired together: the edge table (``walk_arrays``), the
bucket lookup (``numba_lookup``), the hand evaluator (``numba_eval``) and the
random streams (``numba_random``). Nothing here re-derives the game; it is the
same walk ``tree_traversal`` makes, with no boundary crossings inside it.

The contract is bit-identity, not merely agreement: the same arrays and the
same random stream as ``tree_traversal``, so a run can cross this change
mid-flight without becoming a second lineage. That is what pins the details
that would otherwise be free choices —

  * the draw order. Chance is dealt where the state-based walk deals it, at the
    edge crossing, and terminals draw their runout before the showdown.
  * the arithmetic. Strategy in float64 off a float32 row, the node utility as
    a running sum in strategy order, the DCFR discount applied per visit and
    before the add — see ``numba_ops.apply_regret_updates``.
  * the generators. Advanced in place and handed back, so Python continues
    where the kernel stopped.

``Deck`` is passed in rather than rebuilt because ``FULL_DECK``'s ORDER decides
which card ``randrange(52)`` names.
"""

from __future__ import annotations

import random

import numpy as np
from numba import jit

from src.core.game.numba_eval import hand_rank
from src.core.game.state import FULL_DECK, Street
from src.engine.solver.infoset.index import preflop_hand_index
from src.engine.solver.numba_lookup import board_id, hand_id, suit_labels
from src.engine.solver.numba_ops import (
    WEIGHTING_CODES,
    compute_dcfr_strategy_weight,
    regret_matching,
)
from src.engine.solver.numba_random import (
    next_word,
    numpy_state,
    python_state,
    random_sample,
    restore_numpy_state,
    restore_python_state,
)
from src.engine.solver.vector.walk_arrays import WalkArrays

# `randrange(52)` names a POSITION in this deck, so the order is part of the
# stream, not an implementation detail.
_DECK_POSITION = {card: index for index, card in enumerate(FULL_DECK)}

_DEAL_BITS = 6
_DECK_SIZE = 52


@jit(nopython=True, cache=True)
def _draw(deck_mask, known, count, out_index, filled, state, index):
    """Deal ``count`` unseen cards, appending their deck positions to ``out_index``.

    Rejection sampling against the seen-card mask, drawing in the same order
    and by the same rule as ``chance.draw_cards`` — six top bits, reject past
    the deck, reject a card already out.
    """
    taken = 0
    while taken < count:
        while True:
            word, index = next_word(state, index)
            position = word >> (32 - _DEAL_BITS)
            if position < _DECK_SIZE:
                break
        mask = deck_mask[position]
        if mask & known:
            continue
        known |= mask
        out_index[filled + taken] = position
        taken += 1
    return known, index


@jit(nopython=True, cache=True)
def _bucket(
    node_id,
    actor,
    is_preflop,
    street_of_node,
    hole_index,
    board_index,
    board_len,
    deck_rank,
    deck_suit,
    preflop_index,
    street_board_ids,
    street_matrix,
    street_offsets,
    hand_to_col,
    sentinel,
):
    """The acting player's bucket: the preflop class, or the abstraction's cell."""
    first = hole_index[actor * 2]
    second = hole_index[actor * 2 + 1]
    if is_preflop == 1:
        return preflop_index[first, second]

    board_ranks = np.empty(board_len, dtype=np.int64)
    board_suits = np.empty(board_len, dtype=np.int64)
    for i in range(board_len):
        board_ranks[i] = deck_rank[board_index[i]]
        board_suits[i] = deck_suit[board_index[i]]

    labels = np.empty(4, dtype=np.int64)
    next_label = suit_labels(board_ranks, board_suits, board_len, labels)
    target = board_id(board_ranks, board_suits, labels)

    street = street_of_node[node_id]
    low = street_offsets[street]
    high = street_offsets[street + 1]
    row = low + np.searchsorted(street_board_ids[low:high], target)
    if row >= high or street_board_ids[row] != target:
        return -1

    hole_ranks = np.empty(2, dtype=np.int64)
    hole_suits = np.empty(2, dtype=np.int64)
    hole_ranks[0] = deck_rank[first]
    hole_suits[0] = deck_suit[first]
    hole_ranks[1] = deck_rank[second]
    hole_suits[1] = deck_suit[second]

    column = hand_to_col[hand_id(hole_ranks, hole_suits, labels, next_label)]
    if column < 0:
        return -1
    value = street_matrix[row, column]
    if value == sentinel:
        return -1
    return value


@jit(nopython=True, cache=True)
def _showdown_winner(hole_index, board_index, board_len, deck_rank, deck_suit):
    """0 if the first seat wins, 1 if the second, -1 on a tie."""
    ranks = np.empty(7, dtype=np.int64)
    suits = np.empty(7, dtype=np.int64)
    for i in range(board_len):
        ranks[i + 2] = deck_rank[board_index[i]]
        suits[i + 2] = deck_suit[board_index[i]]

    ranks[0] = deck_rank[hole_index[0]]
    suits[0] = deck_suit[hole_index[0]]
    ranks[1] = deck_rank[hole_index[1]]
    suits[1] = deck_suit[hole_index[1]]
    left = hand_rank(ranks, suits)

    ranks[0] = deck_rank[hole_index[2]]
    suits[0] = deck_suit[hole_index[2]]
    ranks[1] = deck_rank[hole_index[3]]
    suits[1] = deck_suit[hole_index[3]]
    right = hand_rank(ranks, suits)

    if left > right:
        return 0
    if left < right:
        return 1
    return -1


@jit(nopython=True, cache=True)
def _terminal_value(
    terminal,
    seat,
    traverser,
    hole_index,
    board_index,
    board_len,
    deck_rank,
    deck_suit,
    deck_mask,
    known,
    state,
    index,
    terminal_is_fold,
    terminal_cards_to_deal,
    terminal_fold,
    terminal_win,
    terminal_lose,
    terminal_tie,
):
    """What the traverser is paid, dealing the runout an all-in outran."""
    if terminal_is_fold[terminal] == 1:
        return terminal_fold[terminal, seat], index

    owed = terminal_cards_to_deal[terminal]
    if owed > 0:
        full = np.empty(5, dtype=np.int64)
        for i in range(board_len):
            full[i] = board_index[i]
        _, index = _draw(deck_mask, known, owed, full, board_len, state, index)
        winner = _showdown_winner(hole_index, full, board_len + owed, deck_rank, deck_suit)
    else:
        winner = _showdown_winner(hole_index, board_index, board_len, deck_rank, deck_suit)

    if winner < 0:
        return terminal_tie[terminal, seat], index
    if winner == traverser:
        return terminal_win[terminal, seat], index
    return terminal_lose[terminal, seat], index


@jit(nopython=True, cache=True)
def walk(
    node_id,
    board_index,
    board_len,
    known,
    traverser,
    button,
    seat,
    iteration,
    hole_index,
    regrets,
    strategy_sum,
    reach_counts,
    cumulative_utility,
    visited,
    edge_offset,
    edge_child,
    edge_deal,
    edge_terminal,
    num_actions,
    row_offset,
    slot_offset,
    buckets_per_node,
    actor_is_button,
    is_preflop,
    street_of_node,
    terminal_is_fold,
    terminal_cards_to_deal,
    terminal_fold,
    terminal_win,
    terminal_lose,
    terminal_tie,
    deck_rank,
    deck_suit,
    deck_mask,
    preflop_index,
    street_board_ids,
    street_matrix,
    street_offsets,
    hand_to_col,
    sentinel,
    cfr_plus,
    weighting,
    alpha,
    beta,
    strategy_weight,
    deal_state,
    deal_index,
    sample_state,
    sample_index,
    applied,
):
    """One decision node. Returns (utility, deal_index, sample_index, applied)."""
    count = num_actions[node_id]
    actor = button if actor_is_button[node_id] == 1 else 1 - button

    bucket = _bucket(
        node_id,
        actor,
        is_preflop[node_id],
        street_of_node,
        hole_index,
        board_index,
        board_len,
        deck_rank,
        deck_suit,
        preflop_index,
        street_board_ids,
        street_matrix,
        street_offsets,
        hand_to_col,
        sentinel,
    )
    if bucket < 0:
        # `_bucket` reports "board not in the abstraction" or "hand impossible
        # on this board" as -1, because a kernel has no exceptions to raise
        # from the lookup itself. Unchecked, that indexes one row BEFORE the
        # node's block and writes regrets into a neighbouring infoset --
        # silent corruption of the shared table, which is exactly what
        # `StaticArrayStorage.view`'s bounds check exists to stop.
        raise ValueError(
            "bucket lookup failed: the board is absent from the abstraction, "
            "or the hand is not a legal combination on it"
        )

    row = row_offset[node_id] + bucket
    start = slot_offset[node_id] + bucket * count
    visited[row] = 1

    # THE function, not a copy of it. Reproducing regret matching by hand lands
    # a different float on roughly one row in 150,000 — `np.sum` does not
    # accumulate the way a written-out loop does — and that is enough to move
    # the average strategy and break bit-identity.
    strategy = regret_matching(regrets[start : start + count])

    base = edge_offset[node_id]

    if actor == traverser:
        utilities = np.zeros(count, dtype=np.float64)
        for i in range(count):
            slot = base + i
            terminal = edge_terminal[slot]
            if terminal >= 0:
                utilities[i], deal_index = _terminal_value(
                    terminal,
                    seat,
                    traverser,
                    hole_index,
                    board_index,
                    board_len,
                    deck_rank,
                    deck_suit,
                    deck_mask,
                    known,
                    deal_state,
                    deal_index,
                    terminal_is_fold,
                    terminal_cards_to_deal,
                    terminal_fold,
                    terminal_win,
                    terminal_lose,
                    terminal_tie,
                )
                continue
            owed = edge_deal[slot]
            child_board = board_index
            child_len = board_len
            child_known = known
            if owed > 0:
                child_board = np.empty(5, dtype=np.int64)
                for j in range(board_len):
                    child_board[j] = board_index[j]
                child_known, deal_index = _draw(
                    deck_mask, known, owed, child_board, board_len, deal_state, deal_index
                )
                child_len = board_len + owed
            utilities[i], deal_index, sample_index, applied = walk(
                edge_child[slot],
                child_board,
                child_len,
                child_known,
                traverser,
                button,
                seat,
                iteration,
                hole_index,
                regrets,
                strategy_sum,
                reach_counts,
                cumulative_utility,
                visited,
                edge_offset,
                edge_child,
                edge_deal,
                edge_terminal,
                num_actions,
                row_offset,
                slot_offset,
                buckets_per_node,
                actor_is_button,
                is_preflop,
                street_of_node,
                terminal_is_fold,
                terminal_cards_to_deal,
                terminal_fold,
                terminal_win,
                terminal_lose,
                terminal_tie,
                deck_rank,
                deck_suit,
                deck_mask,
                preflop_index,
                street_board_ids,
                street_matrix,
                street_offsets,
                hand_to_col,
                sentinel,
                cfr_plus,
                weighting,
                alpha,
                beta,
                strategy_weight,
                deal_state,
                deal_index,
                sample_state,
                sample_index,
                applied,
            )

        # `np.dot`, not a running sum: the state-based traversal computes this
        # as `float(np.dot(strategy, action_utilities))`, and BLAS does not
        # accumulate in index order. A hand-rolled loop lands 1-2 ulp away,
        # which is enough to break bit-identity and, through the node utility,
        # every regret below it.
        node_utility = np.dot(strategy, utilities)

        for i in range(count):
            slot = start + i
            if weighting == 2 and iteration > 1:
                exponent = alpha if regrets[slot] > 0 else beta
                if exponent == 0.0:
                    regrets[slot] *= 0.5
                else:
                    scaled = np.float64(iteration) ** exponent
                    regrets[slot] *= scaled / (scaled + 1.0)
            weighted = utilities[i] - node_utility
            if weighting == 1:
                weighted = weighted * iteration
            updated = regrets[slot] + weighted
            if cfr_plus and updated < 0:
                updated = 0.0
            regrets[slot] = updated

        reach_counts[row] += 1
        cumulative_utility[row] += node_utility
        return node_utility, deal_index, sample_index, applied + 1

    for i in range(count):
        # Widen, add, round ONCE — numpy's `float32[i] += float64` computes in
        # float64 and rounds on the store, and narrowing the addend first
        # instead lands a different float on about one accumulation in 150,000.
        # float32, NOT float64. `tree_traversal` does
        # `strategy_sum[slot] += probability * weight` where `probability` came
        # from `.tolist()` and is a PYTHON float -- and under NEP 50 a Python
        # scalar is weak, so numpy computes that add entirely in float32.
        # Numba types the same expression as float64 and rounds on the store,
        # which is a different result about once in 150,000 accumulations.
        strategy_sum[start + i] += np.float32(strategy[i] * strategy_weight)
    applied += 1

    draw, sample_index = random_sample(sample_state, sample_index)
    acc = 0.0
    chosen = count - 1
    for i in range(count):
        acc += strategy[i]
        if draw < acc:
            chosen = i
            break

    slot = base + chosen
    terminal = edge_terminal[slot]
    if terminal >= 0:
        value, deal_index = _terminal_value(
            terminal,
            seat,
            traverser,
            hole_index,
            board_index,
            board_len,
            deck_rank,
            deck_suit,
            deck_mask,
            known,
            deal_state,
            deal_index,
            terminal_is_fold,
            terminal_cards_to_deal,
            terminal_fold,
            terminal_win,
            terminal_lose,
            terminal_tie,
        )
        return value, deal_index, sample_index, applied

    owed = edge_deal[slot]
    child_board = board_index
    child_len = board_len
    child_known = known
    if owed > 0:
        child_board = np.empty(5, dtype=np.int64)
        for j in range(board_len):
            child_board[j] = board_index[j]
        child_known, deal_index = _draw(
            deck_mask, known, owed, child_board, board_len, deal_state, deal_index
        )
        child_len = board_len + owed

    return walk(
        edge_child[slot],
        child_board,
        child_len,
        child_known,
        traverser,
        button,
        seat,
        iteration,
        hole_index,
        regrets,
        strategy_sum,
        reach_counts,
        cumulative_utility,
        visited,
        edge_offset,
        edge_child,
        edge_deal,
        edge_terminal,
        num_actions,
        row_offset,
        slot_offset,
        buckets_per_node,
        actor_is_button,
        is_preflop,
        street_of_node,
        terminal_is_fold,
        terminal_cards_to_deal,
        terminal_fold,
        terminal_win,
        terminal_lose,
        terminal_tie,
        deck_rank,
        deck_suit,
        deck_mask,
        preflop_index,
        street_board_ids,
        street_matrix,
        street_offsets,
        hand_to_col,
        sentinel,
        cfr_plus,
        weighting,
        alpha,
        beta,
        strategy_weight,
        deal_state,
        deal_index,
        sample_state,
        sample_index,
        applied,
    )


def _artifact_arrays(bucketer, street):
    """One street's board ids and bucket matrix, straight off the artifact.

    Reaching past `DenseBucketer`'s underscore is deliberate and confined here:
    a kernel needs the raw arrays, and the public `get_bucket` is exactly the
    Python call it exists to avoid. Named so the coupling is one function
    rather than four scattered attribute reads.
    """
    return bucketer._board_ids[street], np.asarray(bucketer._buckets[street])  # noqa: SLF001


def _artifact_columns(bucketer):
    """The static hand-id-to-column map and the not-a-legal-combo sentinel."""
    return (
        np.asarray(bucketer._hand_id_to_col, dtype=np.int64),  # noqa: SLF001
        int(next(iter(bucketer._sentinels.values()))),  # noqa: SLF001
    )


class CompiledContext:
    """Everything the kernel needs, resolved once per solver rather than per iteration.

    The per-street bucket matrices are stacked into one array with an offset
    per street, because a kernel cannot hold a dict of them and every street
    shares the same column space.
    """

    __slots__ = (
        "arrays",
        "deck_mask",
        "deck_rank",
        "deck_suit",
        "hand_to_col",
        "matrix",
        "preflop_index",
        "sentinel",
        "street_ids",
        "street_of_node",
        "street_offsets",
    )

    def __init__(self, tree, bucketer, deck):
        self.arrays = WalkArrays(tree)
        self.deck_rank = np.array([c.rank_eval7() for c in deck], dtype=np.int64)
        self.deck_suit = np.array([c.suit_eval7() for c in deck], dtype=np.int64)
        self.deck_mask = np.array([c.mask for c in deck], dtype=np.int64)

        # `bucket_of` bypasses the abstraction preflop and uses the 169 classes,
        # so the table is built from that same function, not from the bucketer.
        table = np.zeros((52, 52), dtype=np.int64)
        for i, first in enumerate(deck):
            for j, second in enumerate(deck):
                if i != j:
                    table[i, j] = preflop_hand_index((first, second))
        self.preflop_index = table

        order = (Street.FLOP, Street.TURN, Street.RIVER)
        ids, matrices, offsets = [], [], [0, 0]
        for street in order:
            street_ids, matrix = _artifact_arrays(bucketer, street)
            ids.append(street_ids)
            matrices.append(matrix)
            offsets.append(offsets[-1] + street_ids.size)
        self.street_ids = np.concatenate(ids)
        self.matrix = np.vstack(matrices)
        self.street_offsets = np.array(offsets, dtype=np.int64)
        self.hand_to_col, self.sentinel = _artifact_columns(bucketer)

        index_of = {Street.PREFLOP: 0, Street.FLOP: 1, Street.TURN: 2, Street.RIVER: 3}
        self.street_of_node = np.array(
            [index_of[node.street] for node in tree.nodes], dtype=np.int64
        )


def run_iteration(solver, context, iteration: int) -> float:
    """One ``train_iteration``, with the traversal inside the kernel.

    Hole cards are still dealt in Python, by ``random.sample(FULL_DECK, 4)``,
    because that is where ``chance.deal_initial_state`` draws them and the
    stream has to match card for card. The kernel then takes both generators,
    advances them, and hands the state back so Python continues where it
    stopped — see ``numba_random``.
    """
    cards = random.sample(FULL_DECK, 4)
    hole_index = np.array([_DECK_POSITION[card] for card in cards], dtype=np.int64)

    traverser = iteration % 2
    button = (iteration // 2) % 2
    known = 0
    for card in cards:
        known |= card.mask

    solver_config = solver.config.solver
    weighting = WEIGHTING_CODES[solver_config.iteration_weighting]
    if solver_config.iteration_weighting == "dcfr":
        strategy_weight = 1.0 * compute_dcfr_strategy_weight(iteration, solver_config.dcfr_gamma)
    elif solver_config.iteration_weighting == "linear":
        strategy_weight = 1.0 * iteration
    else:
        strategy_weight = 1.0

    deal_state, deal_index = python_state()
    sample_state, sample_index = numpy_state()

    arrays = context.arrays
    storage = solver.storage
    utility, deal_index, sample_index, applied = walk(
        0,
        np.zeros(5, dtype=np.int64),
        0,
        known,
        traverser,
        button,
        0 if traverser == button else 1,
        iteration,
        hole_index,
        storage.regrets,
        storage.strategy_sum,
        storage.reach_counts,
        storage.cumulative_utility,
        storage.visited,
        arrays.edge_offset,
        arrays.edge_child,
        arrays.edge_deal,
        arrays.edge_terminal,
        arrays.num_actions,
        arrays.row_offset,
        arrays.slot_offset,
        arrays.buckets_per_node,
        arrays.node_actor_is_button,
        arrays.is_preflop,
        context.street_of_node,
        arrays.terminal_is_fold,
        arrays.terminal_cards_to_deal,
        arrays.terminal_fold,
        arrays.terminal_win,
        arrays.terminal_lose,
        arrays.terminal_tie,
        context.deck_rank,
        context.deck_suit,
        context.deck_mask,
        context.preflop_index,
        context.street_ids,
        context.matrix,
        context.street_offsets,
        context.hand_to_col,
        context.sentinel,
        solver_config.cfr_plus,
        weighting,
        solver_config.dcfr_alpha,
        solver_config.dcfr_beta,
        strategy_weight,
        deal_state,
        deal_index,
        sample_state,
        sample_index,
        0,
    )

    restore_python_state(deal_state, deal_index)
    restore_numpy_state(sample_state, sample_index)
    solver.applied_updates += applied
    return -utility if traverser == 1 else utility
