"""The flop-subtree walk of the public-tree best response, compiled.

`PublicTreeBestResponse._walk` is the SPECIFICATION. This reproduces it
operation for operation -- visit order, tally order, every masked write, even
the zero vectors the Python walk adds -- so the two agree to the last bit,
pinned by `tests/pipeline/evaluation/estimators/test_public_tree_walk.py`. It
covers the plain measurement; the resolver, the decomposition and the
in-abstraction responder keep the Python walk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numba import njit

from src.engine.search.range_inference import NUM_COMBOS, blocked_combos
from src.engine.search.subgame_cfr import (
    _CARD_A,
    _CARD_B,
    _masses,
    _nonblocking_mass,
    _pairwise_sum,
)
from src.engine.solver.mccfr.chance import begin_street

if TYPE_CHECKING:
    from src.core.game.state import GameState
    from src.pipeline.evaluation.estimators.public_tree_br import PublicTreeBestResponse

DECISION, FOLD, SHOWDOWN, CHANCE = 0, 1, 2, 3
_STREET_OF_BOARD = {3: "FLOP", 4: "TURN", 5: "RIVER"}


@dataclass(frozen=True)
class Subtree:
    """The betting structure below one flop deal. Cards never enter it: the
    tree below a deal is the same whatever was dealt, so one serves every flop
    under the same preflop line and button."""

    kind: np.ndarray
    actor: np.ndarray
    child_start: np.ndarray
    child_count: np.ndarray
    children: np.ndarray
    table: np.ndarray  # decision node -> index of its policy table
    fold_payoff: np.ndarray  # (nodes, 2): each seat's chips at a fold
    pot: np.ndarray
    invested: np.ndarray  # (nodes, 2)
    rows: np.ndarray  # every table's (buckets + 1) x actions rows, flattened
    missing: np.ndarray
    row_offset: np.ndarray  # table -> its first row, indexing `missing`
    value_offset: np.ndarray  # table -> its first value, indexing `rows`
    row_count: np.ndarray  # buckets + 1: the last row answers a blocked combo
    table_actions: np.ndarray
    depth: int  # nodes on the longest root-to-leaf path


@dataclass(frozen=True)
class Boards:
    """Every board one flop job can reach: the flop, its sampled turns and their
    sampled rivers, with the deals between them and what each board needs."""

    length: np.ndarray
    buckets: np.ndarray  # (boards, combos) bucket per combo, -1 where blocked
    deal_start: np.ndarray
    deal_count: np.ndarray
    deal_board: np.ndarray  # option -> the board it deals to
    deal_weight: np.ndarray
    deal_block: np.ndarray  # (options, combos) combos the dealt card blocks
    sd_start: np.ndarray  # showdown arrays per 5-card board, by offset
    sd_count: np.ndarray
    sd_groups_start: np.ndarray
    sd_groups_count: np.ndarray
    sd_flat_start: np.ndarray
    sd_flat_count: np.ndarray
    sorted_combo: np.ndarray
    group_starts: np.ndarray
    card_pos_flat: np.ndarray
    seg_end: np.ndarray
    ptr_worse: np.ndarray
    ptr_group: np.ndarray
    group: np.ndarray
    a_idx: np.ndarray
    b_idx: np.ndarray


def build_subtree(engine: PublicTreeBestResponse, state: GameState) -> Subtree:
    """Walk the betting tree below ``state`` once, with the engine's own rules."""
    rules, model = engine._rules, engine._action_model  # noqa: SLF001 -- same estimator
    kind: list[int] = []
    actor: list[int] = []
    children_of: list[list[int]] = []
    table: list[int] = []
    fold_payoff: list[tuple[float, float]] = []
    pot: list[float] = []
    invested: list[tuple[float, float]] = []
    tables: list[tuple[np.ndarray, np.ndarray]] = []
    table_of_context: dict[tuple, int] = {}
    spare = [card for card in _deck() if card not in state.board]

    def visit(node_state: GameState) -> int:
        node = len(kind)
        kind.append(DECISION)
        actor.append(-1)
        children_of.append([])
        table.append(-1)
        fold_payoff.append((0.0, 0.0))
        pot.append(0.0)
        invested.append((0.0, 0.0))
        if node_state.is_terminal:
            if node_state.ended_by_fold:
                kind[node] = FOLD
                fold_payoff[node] = (
                    float(rules.get_payoff(node_state, 0)),
                    float(rules.get_payoff(node_state, 1)),
                )
            else:
                kind[node] = SHOWDOWN
                chips = rules.invested_chips(node_state)
                pot[node] = float(node_state.pot)
                invested[node] = (float(chips[0]), float(chips[1]))
            return node
        if len(node_state.board) < node_state.street.board_card_count:
            kind[node] = CHANCE
            dealt = begin_street(node_state, (spare[len(node_state.board)],))
            children_of[node] = [visit(dealt)]
            return node
        legal = rules.get_legal_actions(node_state, model)
        actor[node] = node_state.current_player
        key = (
            node_state.current_player,
            node_state.street,
            node_state.normalized_betting_sequence(),
        )
        index = table_of_context.get(key)
        if index is None:
            index = table_of_context[key] = len(tables)
            tables.append(engine._policy_table(node_state, legal, key))  # noqa: SLF001
        table[node] = index
        children_of[node] = [visit(rules.apply_action(node_state, action)) for action in legal]
        return node

    visit(state)
    counts = [len(c) for c in children_of]
    depth = [1] * len(kind)
    for node in range(len(kind) - 1, -1, -1):  # children are numbered after parents
        if children_of[node]:
            depth[node] = 1 + max(depth[c] for c in children_of[node])
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    row_counts = [rows.shape[0] for rows, _ in tables]
    row_offsets = np.concatenate([[0], np.cumsum(row_counts)[:-1]])
    value_offsets = np.concatenate([[0], np.cumsum([rows.size for rows, _ in tables])[:-1]])
    return Subtree(
        kind=np.array(kind, dtype=np.int64),
        actor=np.array(actor, dtype=np.int64),
        child_start=starts,
        child_count=np.array(counts, dtype=np.int64),
        children=np.array([c for group in children_of for c in group], dtype=np.int64),
        table=np.array(table, dtype=np.int64),
        fold_payoff=np.array(fold_payoff, dtype=np.float64).reshape(-1, 2),
        pot=np.array(pot, dtype=np.float64),
        invested=np.array(invested, dtype=np.float64).reshape(-1, 2),
        rows=np.concatenate([rows.ravel() for rows, _ in tables]).astype(np.float64),
        missing=np.concatenate([missing for _, missing in tables]).astype(np.bool_),
        row_offset=row_offsets.astype(np.int64),
        value_offset=value_offsets.astype(np.int64),
        row_count=np.array(row_counts, dtype=np.int64),
        table_actions=np.array([rows.shape[1] for rows, _ in tables], dtype=np.int64),
        depth=depth[0],
    )


def build_boards(engine: PublicTreeBestResponse, flop: tuple) -> Boards:
    """The flop's sampled turns and rivers, through the engine's own board plan."""
    from src.core.game.state import Street  # noqa: PLC0415 -- enum lookup by name only

    plan = engine._plan  # noqa: SLF001 -- same estimator
    boards: list[tuple] = [tuple(flop)]
    deal_start: list[int] = []
    deal_count: list[int] = []
    deal_board: list[int] = []
    deal_weight: list[float] = []
    deal_block: list[np.ndarray] = []
    b = 0
    while b < len(boards):
        board = boards[b]
        options = plan.deal_options(board) if len(board) < 5 else []
        deal_start.append(len(deal_board))
        deal_count.append(len(options))
        for cards, weight in options:
            deal_board.append(len(boards))
            boards.append((*board, *cards))
            deal_weight.append(float(weight))
            deal_block.append(blocked_combos(cards))
        b += 1

    buckets = np.stack(
        [engine._bucket_vector(board, Street[_STREET_OF_BOARD[len(board)]]) for board in boards]  # noqa: SLF001
    ).astype(np.int64)
    rivers = [i for i, board in enumerate(boards) if len(board) == 5]
    evaluators = {i: engine._showdown_evaluator(boards[i]) for i in rivers}  # noqa: SLF001

    def stack(attr: str, width: int | None = None) -> np.ndarray:
        parts = [getattr(evaluators[i], attr) for i in rivers]
        if not parts:
            return np.zeros((0, width) if width else 0, dtype=np.int64)
        return np.concatenate(parts).astype(np.int64)

    def spans(attr: str) -> tuple[np.ndarray, np.ndarray]:
        start = np.zeros(len(boards), dtype=np.int64)
        count = np.zeros(len(boards), dtype=np.int64)
        cursor = 0
        for i in rivers:
            n = len(getattr(evaluators[i], attr))
            start[i], count[i] = cursor, n
            cursor += n
        return start, count

    sd_start, sd_count = spans("sorted_combo")
    groups_start, groups_count = spans("group_starts")
    flat_start, flat_count = spans("card_pos_flat")
    return Boards(
        length=np.array([len(board) for board in boards], dtype=np.int64),
        buckets=buckets,
        deal_start=np.array(deal_start, dtype=np.int64),
        deal_count=np.array(deal_count, dtype=np.int64),
        deal_board=np.array(deal_board, dtype=np.int64),
        deal_weight=np.array(deal_weight, dtype=np.float64),
        deal_block=(
            np.stack(deal_block) if deal_block else np.zeros((0, NUM_COMBOS), dtype=np.bool_)
        ),
        sd_start=sd_start,
        sd_count=sd_count,
        sd_groups_start=groups_start,
        sd_groups_count=groups_count,
        sd_flat_start=flat_start,
        sd_flat_count=flat_count,
        sorted_combo=stack("sorted_combo"),
        group_starts=stack("group_starts"),
        card_pos_flat=stack("card_pos_flat"),
        seg_end=stack("_seg_end", 2),
        ptr_worse=stack("_ptr_worse", 2),
        ptr_group=stack("_ptr_group", 2),
        group=stack("_group"),
        a_idx=stack("_a_idx"),
        b_idx=stack("_b_idx"),
    )


def _deck() -> list:
    from src.core.game.state import FULL_DECK  # noqa: PLC0415 -- one list, built once per subtree

    return list(FULL_DECK)


def walk_flop_part(
    tree: Subtree,
    boards: Boards,
    br_seat: int,
    reach: np.ndarray,
    own_reach: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float, float]:
    """(best, self-play, nodes, decision mass, missing mass) of one flop subtree."""
    depth = tree.depth + 1
    width = int(tree.table_actions.max()) if tree.table_actions.size else 1
    workspace = (
        np.empty((depth, NUM_COMBOS)),  # a node's best-response values, by depth
        np.empty((depth, NUM_COMBOS)),  # and its self-play values
        np.empty((depth, NUM_COMBOS)),  # the opponent reach a node was entered with
        np.empty((depth, NUM_COMBOS)),  # the responder's own reach, when a node changed it
        np.empty((depth, NUM_COMBOS, width)),  # a responder's per-combo strategy
        np.empty((depth, NUM_COMBOS, width)),  # its children's best-response values
        np.empty((depth, NUM_COMBOS, width)),  # its children's self-play values
        np.empty(NUM_COMBOS),  # the opponent reach on fallback rows, compacted
    )
    counts = np.zeros(1, dtype=np.int64)
    masses = np.zeros(2, dtype=np.float64)
    shape = (
        tree.kind, tree.actor, tree.child_start, tree.child_count, tree.children,
        tree.table, tree.fold_payoff, tree.pot, tree.invested, tree.rows, tree.missing,
        tree.row_offset, tree.value_offset, tree.row_count, tree.table_actions,
    )  # fmt: skip
    deals = (
        boards.length, boards.buckets, boards.deal_start, boards.deal_count,
        boards.deal_board, boards.deal_weight, boards.deal_block, boards.sd_start,
        boards.sd_count, boards.sd_groups_start, boards.sd_groups_count,
        boards.sd_flat_start, boards.sd_flat_count, boards.sorted_combo,
        boards.group_starts, boards.card_pos_flat, boards.seg_end, boards.ptr_worse,
        boards.ptr_group, boards.group, boards.a_idx, boards.b_idx,
    )  # fmt: skip
    _walk(
        0,
        0,
        0,
        np.ascontiguousarray(reach, dtype=np.float64),
        np.ascontiguousarray(own_reach, dtype=np.float64),
        br_seat,
        shape,
        deals,
        workspace,
        counts,
        masses,
    )
    return (
        workspace[0][0].copy(),
        workspace[1][0].copy(),
        int(counts[0]),
        float(masses[0]),
        float(masses[1]),
    )


@njit(cache=False)
def _any(values):
    """``ndarray.any()``: True on any nonzero, NaN included."""
    for i in range(values.shape[0]):  # noqa: SIM110 -- numba compiles no generator
        if values[i] != 0.0:
            return True
    return False


@njit(cache=False)
def _showdown(b, pot, invested, reach, deals):
    """``win * pot + tie * (pot / 2.0) - invested * alive`` on a complete board."""
    lo, n = deals[7][b], deals[8][b]
    glo, gn = deals[9][b], deals[10][b]
    flo, fn = deals[11][b], deals[12][b]
    combos = reach.shape[0]
    win = np.zeros(combos, dtype=np.float64)
    tie = np.zeros(combos, dtype=np.float64)
    alive = np.zeros(combos, dtype=np.float64)
    _masses(
        reach,
        deals[13][lo : lo + n],
        deals[14][glo : glo + gn],
        deals[15][flo : flo + fn],
        deals[16][lo : lo + n],
        deals[17][lo : lo + n],
        deals[18][lo : lo + n],
        deals[19][lo : lo + n],
        deals[20][lo : lo + n],
        deals[21][lo : lo + n],
        win,
        tie,
        alive,
    )
    half = pot / 2.0
    out = np.empty(combos, dtype=np.float64)
    for h in range(combos):
        out[h] = (win[h] * pot + tie[h] * half) - invested * alive[h]
    return out


@njit(cache=False)
def _runout(b, pot, invested, reach, deals):
    """`_runout_values`: an all-in board completed over the sampled deals."""
    length, deal_start, deal_count = deals[0], deals[2], deals[3]
    deal_board, deal_weight, deal_block = deals[4], deals[5], deals[6]
    if length[b] == 5:
        return _showdown(b, pot, invested, reach, deals)
    combos = reach.shape[0]
    values = np.zeros(combos, dtype=np.float64)
    for o in range(deal_start[b], deal_start[b] + deal_count[b]):
        weight = deal_weight[o]
        child_reach = np.empty(combos, dtype=np.float64)
        for h in range(combos):
            child_reach[h] = (0.0 if deal_block[o, h] else reach[h]) * weight
        if not _any(child_reach):
            continue
        child = _runout(deal_board[o], pot, invested, child_reach, deals)
        for h in range(combos):
            if deal_block[o, h]:
                child[h] = 0.0
            values[h] += child[h]
    return values


@njit(cache=False)
def _walk(n, b, d, opp, own, br_seat, shape, deals, ws, counts, masses):
    """One node of `PublicTreeBestResponse._walk`; its values land in ``ws[0][d]``
    and ``ws[1][d]``. A child is entered at depth ``d + 1``, so a node's buffers
    are never overwritten while it still needs them."""
    kind, actor, child_start, child_count, children = (
        shape[0], shape[1], shape[2], shape[3], shape[4],
    )  # fmt: skip
    best, self_play = ws[0][d], ws[1][d]
    combos = opp.shape[0]
    counts[0] += 1
    if not _any(opp):
        best[:] = 0.0
        self_play[:] = 0.0
        return
    k = kind[n]
    if k == FOLD:
        mass = _nonblocking_mass(opp, _CARD_A, _CARD_B)
        payoff = shape[6][n, br_seat]
        for h in range(combos):
            best[h] = payoff * mass[h]
            self_play[h] = best[h]
        return
    if k == SHOWDOWN:
        values = _runout(b, shape[7][n], shape[8][n, br_seat], opp, deals)
        for h in range(combos):
            best[h] = values[h]
            self_play[h] = values[h]
        return

    child_best, child_self = ws[0][d + 1], ws[1][d + 1]
    best[:] = 0.0
    self_play[:] = 0.0
    if k == CHANCE:
        deal_start, deal_count, deal_board = deals[2], deals[3], deals[4]
        deal_weight, deal_block = deals[5], deals[6]
        child_reach = ws[2][d + 1]
        child_node = children[child_start[n]]
        for o in range(deal_start[b], deal_start[b] + deal_count[b]):
            weight = deal_weight[o]
            for h in range(combos):
                child_reach[h] = (0.0 if deal_block[o, h] else opp[h]) * weight
            if _any(child_reach):
                _walk(child_node, deal_board[o], d + 1, child_reach, own, br_seat, shape,
                      deals, ws, counts, masses)  # fmt: skip
                for h in range(combos):
                    if deal_block[o, h]:
                        child_best[h] = 0.0
                        child_self[h] = 0.0
                for h in range(combos):
                    best[h] += child_best[h]
                for h in range(combos):
                    self_play[h] += child_self[h]
        return

    rows, missing = shape[9], shape[10]
    t = shape[5][n]
    first_row, first_value = shape[11][t], shape[12][t]
    table_rows, width = shape[13][t], shape[14][t]
    bucket = deals[1][b]
    start, count = child_start[n], child_count[n]

    if actor[n] == br_seat:
        sigma, best_children, self_children = ws[4][d], ws[5][d], ws[6][d]
        for h in range(combos):
            r = bucket[h]
            if r < 0:
                r += table_rows
            for a in range(width):
                sigma[h, a] = rows[first_value + r * width + a]
        child_own = ws[3][d + 1]
        for a in range(count):
            for h in range(combos):
                child_own[h] = own[h] * sigma[h, a]
            _walk(children[start + a], b, d + 1, opp, child_own, br_seat, shape, deals, ws,
                  counts, masses)  # fmt: skip
            for h in range(combos):
                best_children[h, a] = child_best[h]
                self_children[h, a] = child_self[h]
        for h in range(combos):
            choice = 0
            top = best_children[h, 0]
            if top == top:
                for a in range(1, width):
                    v = best_children[h, a]
                    if v != v:
                        choice = a
                        break
                    if v > top:
                        top = v
                        choice = a
            best[h] = best_children[h, choice]
            acc = 0.0
            for a in range(width):
                acc += sigma[h, a] * self_children[h, a]
            self_play[h] = acc
        return

    # Opponent: the tallies first, in the Python walk's order, then each action.
    masses[0] += _pairwise_sum(opp, 0, combos)
    kept = ws[7]
    m = 0
    for h in range(combos):
        r = bucket[h]
        if r < 0:
            r += table_rows
        if missing[first_row + r]:
            kept[m] = opp[h]
            m += 1
    masses[1] += _pairwise_sum(kept, 0, m)
    child_reach = ws[2][d + 1]
    for a in range(count):
        for h in range(combos):
            r = bucket[h]
            if r < 0:
                r += table_rows
            child_reach[h] = opp[h] * rows[first_value + r * width + a]
        if not _any(child_reach):
            continue
        _walk(children[start + a], b, d + 1, child_reach, own, br_seat, shape, deals, ws,
              counts, masses)  # fmt: skip
        for h in range(combos):
            best[h] += child_best[h]
        for h in range(combos):
            self_play[h] += child_self[h]


__all__ = ("Boards", "Subtree", "build_boards", "build_subtree", "walk_flop_part")
