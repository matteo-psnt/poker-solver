"""The betting tree as the flat arrays a compiled traversal indexes.

External sampling needs two things ``CompiledTree`` deliberately does not
carry, both because it serves a vector pass that samples the public board once
per iteration and therefore treats a street boundary as an ordinary edge:

    edge_deal        board cards owed before the child acts — 0 within a
                     street, 3 onto the flop, 1 onto the turn and river. A
                     per-hand traversal deals at each crossing; a public-chance
                     pass has already dealt.
    terminal payoffs per SEAT and per outcome, not one button-relative number.
                     The vector pass resolves a showdown by comparing ranges
                     and scaling half the pot; a per-hand traversal needs what
                     each seat is paid when it wins, loses or ties.

These are DERIVED FROM ``BettingTree.edges``, not re-walked from the rules.
That matters: ``_Compiler`` re-applies every action against ``GameRules`` to
rediscover structure the enumeration already recorded, so the repository
currently walks the rules twice and can drift once. Deriving here means the
compiled walk and the object-form traversal read one source, and
``test_walk_arrays`` pins the derived edges against ``_Compiler``'s independent
walk — so the remaining duplication is guarded rather than merely tolerated,
and can be collapsed into this once the walk lands.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from src.core.game.state import Street

if TYPE_CHECKING:
    from src.engine.solver.betting_tree import BettingTree

# Terminal kinds, matching `compiled_tree.TerminalKind` so the two tables mean
# the same thing where they overlap.
FOLD = 0
SHOWDOWN = 1


class WalkArrays:
    """Flat arrays for one betting tree. Everything a kernel needs, nothing else.

    Layout mirrors ``CompiledTree``: node ``n``'s edges occupy
    ``[edge_offset[n], edge_offset[n + 1])`` in ``legal_actions`` order, so edge
    slot ``i`` and action slot ``i`` are the same action.

    Terminal rows are indexed by ``edge_terminal``; ``-1`` there means the edge
    leads to a decision node instead, named by ``edge_child``.

    Payoff arrays are ``(num_terminals, 2)``, indexed ``[terminal, seat]`` with
    seat 0 the BUTTON — the enumeration is button-relative, so a traversal
    picks its column from whether the traverser holds the button.
    """

    __slots__ = (
        "buckets_per_node",
        "edge_child",
        "edge_deal",
        "edge_offset",
        "edge_terminal",
        "is_preflop",
        "node_actor_is_button",
        "num_actions",
        "row_base",
        "row_stride",
        "slot_base",
        "slot_stride",
        "terminal_cards_to_deal",
        "terminal_fold",
        "terminal_is_fold",
        "terminal_lose",
        "terminal_tie",
        "terminal_win",
    )

    def __init__(self, tree: BettingTree):
        count = len(tree.nodes)
        self.edge_offset = np.zeros(count + 1, dtype=np.int64)
        np.cumsum(tree.num_actions, out=self.edge_offset[1:])
        total = int(self.edge_offset[-1])

        self.edge_child = np.full(total, -1, dtype=np.int64)
        self.edge_deal = np.zeros(total, dtype=np.int8)
        self.edge_terminal = np.full(total, -1, dtype=np.int64)

        # Terminal outcomes are already interned by the enumeration, so identity
        # is the dedup key and the table has one row per distinct payoff table.
        rows: dict[int, int] = {}
        is_fold: list[int] = []
        cards: list[int] = []
        fold: list[tuple[float, float]] = []
        win: list[tuple[float, float]] = []
        lose: list[tuple[float, float]] = []
        tie: list[tuple[float, float]] = []

        for node_id, edges in enumerate(tree.edges):
            base = int(self.edge_offset[node_id])
            for slot, edge in enumerate(edges):
                position = base + slot
                if edge.terminal is None:
                    self.edge_child[position] = edge.child_id
                    self.edge_deal[position] = edge.deal
                    continue
                terminal = edge.terminal
                index = rows.get(id(terminal))
                if index is None:
                    index = len(is_fold)
                    rows[id(terminal)] = index
                    is_fold.append(1 if terminal.is_fold else 0)
                    cards.append(terminal.cards_to_deal)
                    fold.append(terminal.fold)
                    win.append(terminal.win)
                    lose.append(terminal.lose)
                    tie.append(terminal.tie)
                self.edge_terminal[position] = index

        self.terminal_is_fold = np.array(is_fold, dtype=np.uint8)
        self.terminal_cards_to_deal = np.array(cards, dtype=np.int8)
        self.terminal_fold = np.array(fold, dtype=np.float64).reshape(-1, 2)
        self.terminal_win = np.array(win, dtype=np.float64).reshape(-1, 2)
        self.terminal_lose = np.array(lose, dtype=np.float64).reshape(-1, 2)
        self.terminal_tie = np.array(tie, dtype=np.float64).reshape(-1, 2)

        self.num_actions = np.asarray(tree.num_actions, dtype=np.int64)
        self.row_base = np.asarray(tree.row_base[:count], dtype=np.int64)
        self.row_stride = np.asarray(tree.row_stride[:count], dtype=np.int64)
        self.slot_base = np.asarray(tree.slot_base[:count], dtype=np.int64)
        self.slot_stride = np.asarray(tree.slot_stride[:count], dtype=np.int64)
        self.buckets_per_node = np.asarray(tree.buckets_per_node, dtype=np.int64)
        self.node_actor_is_button = np.array(
            [node.actor_is_button for node in tree.nodes], dtype=np.uint8
        )
        self.is_preflop = np.array(
            [node.street == Street.PREFLOP for node in tree.nodes], dtype=np.uint8
        )

    @property
    def num_edges(self) -> int:
        return int(self.edge_offset[-1])

    @property
    def num_terminals(self) -> int:
        return int(self.terminal_is_fold.shape[0])

    def __str__(self) -> str:
        folds = int(self.terminal_is_fold.sum())
        return (
            f"WalkArrays(nodes={self.edge_offset.shape[0] - 1:,}, "
            f"edges={self.num_edges:,}, terminals={self.num_terminals:,} "
            f"[fold={folds:,}, showdown={self.num_terminals - folds:,}])"
        )
