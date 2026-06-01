"""Edges, terminals and a level order for the public betting tree.

``BettingTree`` enumerates *decision nodes only*. Its walk returns at terminals
without recording them, collapses street transitions by filling in placeholder
board cards, and stores no child pointers — everything a recursive traversal can
rediscover by re-applying actions, and everything a vector traversal cannot.

This module compiles that missing structure once, into flat arrays:

    edges        one slot per (node, action), pointing at a decision node or a
                 terminal, in the node's own ``legal_actions`` order
    terminals    kind (fold / showdown) and the button's payoff magnitude
    levels       nodes grouped by depth, so a pass can go level by level rather
                 than recursing

Street transitions stay collapsed, and that is deliberate rather than inherited:
the design this serves samples the public board once per iteration (public chance
sampling) and vectorises over private hands, so a street boundary is an ordinary
edge and the board is a per-iteration input, not a branch in the tree.

Payoffs are button-relative, matching ``BettingTree``'s button-symmetric node
identity. ``terminal_value`` is what the *button* wins:

    fold        signed: positive when the non-button folded, negative otherwise.
                Fully determined by the public sequence, so the vector pass needs
                no card information at a fold terminal at all.
    showdown    unsigned magnitude (half the pot); the sign comes from comparing
                hands, which is the only place cards enter the terminal layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import numpy as np

from src.core.game.rules import GameRules
from src.engine.solver.betting_tree import (
    _PLACEHOLDER_BOARD,
    _PLACEHOLDER_HOLE,
    BettingTree,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.game.state import GameState, Street

# A terminal's ``child_id`` entries index the terminal table; decision-node
# entries index the node table. One array holds both, so the kind is carried
# alongside rather than encoded in the sign (which would make node 0 ambiguous).
EDGE_TO_NODE = 0
EDGE_TO_TERMINAL = 1


class TerminalKind(IntEnum):
    """Why the hand ended, which decides how its value is computed."""

    FOLD = 0
    SHOWDOWN = 1


@dataclass(frozen=True, slots=True)
class CompiledTree:
    """The betting tree as flat arrays a vector pass can walk.

    Attributes:
        tree: The enumeration this was compiled from; owns node identity, the
            per-node action counts and the infoset row layout.
        edge_offset: Exclusive prefix sum of action counts. Node ``n``'s edges
            are ``[edge_offset[n], edge_offset[n + 1])``, in ``legal_actions``
            order, so edge slot ``i`` and action slot ``i`` mean the same action.
        edge_kind: ``EDGE_TO_NODE`` or ``EDGE_TO_TERMINAL`` per edge.
        edge_target: Node id or terminal id per edge, per ``edge_kind``.
        terminal_kind: ``TerminalKind`` per terminal.
        terminal_value: Button payoff (fold, signed) or half-pot (showdown).
        terminal_street: Street the hand ended on. Showdowns before the river
            still need a full board, which is why this is recorded.
        depth: Longest path from the root to each node, in decision nodes.
        level_nodes: Node ids sorted by depth, then by id.
        level_offset: Level ``d`` occupies ``level_nodes[level_offset[d] :
            level_offset[d + 1]]``.
        parent_count: How many edges point at each node. All-ones would mean a
            pure tree; any entry above one makes this a DAG, and a forward pass
            must then accumulate reach rather than assign it.
    """

    tree: BettingTree
    edge_offset: np.ndarray
    edge_kind: np.ndarray
    edge_target: np.ndarray
    terminal_kind: np.ndarray
    terminal_value: np.ndarray
    terminal_street: np.ndarray
    depth: np.ndarray
    level_nodes: np.ndarray
    level_offset: np.ndarray
    parent_count: np.ndarray

    @property
    def num_nodes(self) -> int:
        return len(self.tree)

    @property
    def num_edges(self) -> int:
        return int(self.edge_offset[-1])

    @property
    def num_terminals(self) -> int:
        return int(self.terminal_kind.shape[0])

    @property
    def num_levels(self) -> int:
        return int(self.level_offset.shape[0]) - 1

    @property
    def is_dag(self) -> bool:
        """True when some node is reachable by more than one edge."""
        return bool((self.parent_count > 1).any())

    def edges_of(self, node_id: int) -> tuple[int, int]:
        return int(self.edge_offset[node_id]), int(self.edge_offset[node_id + 1])

    def nodes_at_level(self, level: int) -> np.ndarray:
        return self.level_nodes[self.level_offset[level] : self.level_offset[level + 1]]

    def __str__(self) -> str:
        folds = int((self.terminal_kind == TerminalKind.FOLD).sum())
        showdowns = self.num_terminals - folds
        return (
            f"CompiledTree(nodes={self.num_nodes:,}, edges={self.num_edges:,}, "
            f"terminals={self.num_terminals:,} [fold={folds:,}, showdown={showdowns:,}], "
            f"levels={self.num_levels}, dag={self.is_dag})"
        )


class _Compiler:
    """Re-walks the enumeration, recording what the enumeration discarded.

    Terminals are deduplicated on the same key decision nodes use — ``(street,
    normalized_betting_sequence)`` — so a terminal reached by two paths is one
    row, exactly as a decision node reached twice is one node. Without that the
    terminal table would count paths rather than states and would disagree with
    the node table about what the tree is.
    """

    def __init__(self, tree: BettingTree, rules: GameRules):
        self.tree = tree
        self.rules = rules

        num_nodes = len(tree)
        self.edge_offset = np.zeros(num_nodes + 1, dtype=np.int64)
        np.cumsum(tree.num_actions, out=self.edge_offset[1:])
        num_edges = int(self.edge_offset[-1])

        self.edge_kind = np.full(num_edges, -1, dtype=np.int8)
        self.edge_target = np.full(num_edges, -1, dtype=np.int64)

        self.terminal_index: dict[tuple[Street, str], int] = {}
        self.terminal_kind: list[int] = []
        self.terminal_value: list[float] = []
        self.terminal_street: list[int] = []

        self.parent_count = np.zeros(num_nodes, dtype=np.int64)
        self.depth = np.full(num_nodes, -1, dtype=np.int64)
        self.visited: set[int] = set()

    def run(self) -> CompiledTree:
        root = self.rules.create_initial_state(
            starting_stack=self.tree.starting_stack,
            hole_cards=_PLACEHOLDER_HOLE,
            button=0,
        )
        self._walk(self._settle_board(root), depth=0)

        level_nodes, level_offset = _build_levels(self.depth)
        return CompiledTree(
            tree=self.tree,
            edge_offset=self.edge_offset,
            edge_kind=self.edge_kind,
            edge_target=self.edge_target,
            terminal_kind=np.array(self.terminal_kind, dtype=np.int8),
            terminal_value=np.array(self.terminal_value, dtype=np.float64),
            terminal_street=np.array(self.terminal_street, dtype=np.int8),
            depth=self.depth,
            level_nodes=level_nodes,
            level_offset=level_offset,
            parent_count=self.parent_count,
        )

    def _settle_board(self, state: GameState) -> GameState:
        """Fill placeholder board cards so the walk sees a fully-dealt street.

        Mirrors ``BettingTree._walk``: the betting tree does not depend on card
        identities, so one representative board stands in for every runout.
        """
        needed = state.street.board_card_count
        if len(state.board) >= needed:
            return state
        return state.replace(board=_PLACEHOLDER_BOARD[:needed], validate=False)

    def _walk(self, state: GameState, depth: int) -> None:
        node_id = self.tree.node_id(state)

        # Depth is the LONGEST path to a node, so a node re-reached deeper has to
        # be re-walked to push that depth through its descendants. Without this
        # a level pass could read a node whose parent sits in the same level.
        if node_id in self.visited and depth <= self.depth[node_id]:
            return
        self.visited.add(node_id)
        self.depth[node_id] = max(self.depth[node_id], depth)

        start = int(self.edge_offset[node_id])
        for slot, action in enumerate(self.tree.legal_actions(node_id)):
            child = self._settle_board(self.rules.apply_action(state, action))
            edge = start + slot

            if child.is_terminal:
                self.edge_kind[edge] = EDGE_TO_TERMINAL
                self.edge_target[edge] = self._terminal_id(child)
                continue

            child_id = self.tree.node_id(child)
            if self.edge_kind[edge] == -1:
                # Count each edge once: a re-walk revisits the same slots.
                self.parent_count[child_id] += 1
            self.edge_kind[edge] = EDGE_TO_NODE
            self.edge_target[edge] = child_id
            self._walk(child, depth + 1)

    def _terminal_id(self, state: GameState) -> int:
        key = (state.street, state.normalized_betting_sequence())
        existing = self.terminal_index.get(key)
        if existing is not None:
            return existing

        terminal_id = len(self.terminal_kind)
        self.terminal_index[key] = terminal_id

        if state.ended_by_fold:
            self.terminal_kind.append(TerminalKind.FOLD)
            self.terminal_value.append(_fold_value_for_button(state))
        else:
            self.terminal_kind.append(TerminalKind.SHOWDOWN)
            self.terminal_value.append(state.pot / 2.0)
        self.terminal_street.append(state.street.value)
        return terminal_id


def _fold_value_for_button(state: GameState) -> float:
    """What the button wins when this fold ends the hand.

    ``current_player`` is the folder at the time of the fold (``get_payoff``
    relies on the same fact), and the folder forfeits exactly what it invested.
    """
    invested = GameRules.invested_chips(state)
    folder = state.current_player
    if folder == state.button_position:
        return -invested[state.button_position]
    return invested[folder]


def _build_levels(depth: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group node ids by depth into a flat array plus offsets."""
    if (depth < 0).any():
        raise ValueError("Compilation left nodes unreachable; the walk did not cover the tree.")

    order = np.argsort(depth, kind="stable")
    counts = np.bincount(depth)
    offset = np.zeros(counts.shape[0] + 1, dtype=np.int64)
    np.cumsum(counts, out=offset[1:])
    return order.astype(np.int64), offset


def compile_tree(tree: BettingTree, rules: GameRules) -> CompiledTree:
    """Compile edges, terminals and a level order for an enumerated tree."""
    compiled = _Compiler(tree, rules).run()
    _verify(compiled)
    return compiled


def _verify(compiled: CompiledTree) -> None:
    """Assert the invariants the vector pass relies on.

    Cheap relative to compilation and load-bearing: a level pass that reads a
    child before its parent produces plausible numbers, not a crash.
    """
    if (compiled.edge_kind == -1).any():
        raise ValueError("Compilation left edges unassigned.")

    node_edges = compiled.edge_kind == EDGE_TO_NODE
    parents = compiled.depth[np.repeat(np.arange(compiled.num_nodes), compiled.tree.num_actions)]
    children = compiled.edge_target[node_edges]
    if not (parents[node_edges] < compiled.depth[children]).all():
        raise ValueError("Level order is not topological: an edge does not increase depth.")


__all__: Sequence[str] = (
    "EDGE_TO_NODE",
    "EDGE_TO_TERMINAL",
    "CompiledTree",
    "TerminalKind",
    "compile_tree",
)
