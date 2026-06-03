"""Static enumeration of the public betting tree.

The public betting tree of HUNL under a fixed action abstraction and stack depth
is *small* and *finite*: 57,604 decision nodes under
`config/training/production.yaml`, enumerable in seconds. Every information set
the solver will ever touch is a pair

    (public betting node, card bucket)

and both components are known before training starts, which makes the whole
infoset space a preallocated flat array indexed by integer arithmetic. Nothing
is discovered at runtime, so nothing has to be reconciled across workers.

Node identity:
    A node is keyed by ``(street, betting_sequence)``. Verified unique -- across
    the full production enumeration every such key maps to exactly one
    ``(actor, pot, stacks, to_call)``, so the flat cross-street token string does
    not merge structurally distinct nodes. See ``test_betting_tree.py``.

Button relativity:
    Nodes are enumerated once, button-relative, and the tree is *exactly*
    button-symmetric. The absolute seat carries no strategic information, so
    keying infosets on it duplicates the whole space and splits an already
    starved update budget across two identical halves. This module keys on the
    acting seat *relative to the button*.

Stack depth:
    The enumeration is valid for ONE starting stack. A state that does not map
    to an enumerated node raises rather than silently landing on a wrong row.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from src.core.game.state import Card, GameState, Street

if TYPE_CHECKING:
    from collections.abc import Sequence

    from src.core.actions.action_model import ActionModel
    from src.core.game.actions import Action
    from src.core.game.rules import GameRules
    from src.engine.solver.protocols import BucketingStrategy

# Enumeration needs a concrete state to walk, but the betting tree does not
# depend on card identities — only on street, pot, stacks and to_call. These are
# placeholders that get overwritten as the walk advances streets.
_PLACEHOLDER_HOLE = (
    (Card.new("As"), Card.new("Kd")),
    (Card.new("Qh"), Card.new("Jc")),
)
_PLACEHOLDER_BOARD = (
    Card.new("2c"),
    Card.new("7d"),
    Card.new("9h"),
    Card.new("4s"),
    Card.new("Ts"),
)

# Preflop uses the 169 canonical starting hands directly rather than equity
# buckets, so its "bucket count" is fixed by the game, not by the abstraction.
NUM_PREFLOP_HANDS = 169


@dataclass(frozen=True, slots=True)
class BettingNode:
    """One public decision point.

    node_id:
        Dense index into the tree's arrays; assigned in deterministic DFS order.
    actor_is_button:
        Which seat acts, relative to the button. The absolute seat is not part of
        node identity (the tree is button-symmetric).
    legal_actions:
        The exact action list the action model produces here, in order. Action
        slot i of every infoset at this node means this action.
    """

    node_id: int
    street: Street
    betting_sequence: str
    actor_is_button: bool
    legal_actions: tuple[Action, ...]

    @property
    def num_actions(self) -> int:
        return len(self.legal_actions)


@dataclass(frozen=True, slots=True)
class TerminalOutcome:
    """What an action that ends the hand pays, for every way it can end.

    Payoffs are constants of the betting line. ``get_payoff`` reads only
    ``pot``, ``stacks`` and the winner, and the first two are fixed by the
    sequence of bets — no card ever moves them. So the whole payoff table is
    computable at enumeration time and the traversal is left with one job at a
    terminal: decide *who won*, which for a fold is also already decided.

    Each field is a ``(button, non-button)`` pair, since the tree is stored
    button-relative. ``win``/``lose`` are indexed by the player being paid, not
    by the winner: ``win[0]`` is what the button collects when the button wins.

    cards_to_deal:
        Board cards the runout still owes at this terminal — zero for a fold,
        and up to five for an all-in that closed the action before the river.
    """

    is_fold: bool
    cards_to_deal: int
    fold: tuple[float, float]
    win: tuple[float, float]
    lose: tuple[float, float]
    tie: tuple[float, float]


@dataclass(frozen=True, slots=True)
class Edge:
    """Where one action at one node leads.

    Exactly one of ``child_id``/``terminal`` is set: an action either hands the
    turn to another decision node or ends the hand.

    deal:
        Board cards the chance dealer owes before the child acts — 0 within a
        street, 3 onto the flop, 1 onto the turn and river.
    """

    child_id: int
    deal: int
    terminal: TerminalOutcome | None


class BettingTree:
    """Every public betting node, enumerated once.

    Construction walks the full tree depth-first, which is a few seconds at
    production settings. Build it once per process and share it.

    Layout:
        Infosets are laid out node-major. Node ``n`` owns ``buckets(street(n))``
        consecutive rows starting at ``row_offset[n]``; row ``r`` of node ``n``
        owns ``num_actions(n)`` consecutive slots starting at ``slot_offset[n] +
        local_row * num_actions(n)``. The layout is ragged — nodes with fewer
        actions occupy fewer slots — so there is no padding waste, unlike a
        dense ``(num_infosets, max_actions)`` rectangle.
    """

    def __init__(
        self,
        rules: GameRules,
        action_model: ActionModel,
        *,
        starting_stack: int,
        buckets_per_street: dict[Street, int],
    ):
        self.rules = rules
        self.action_model = action_model
        self.starting_stack = starting_stack
        self.buckets_per_street = dict(buckets_per_street)

        self.nodes: list[BettingNode] = []
        self._index: dict[tuple[Street, str], int] = {}
        self.edges: list[tuple[Edge, ...]] = []
        # Terminals are interned: the production tree ends 101,904 ways but a
        # payoff table is fixed by (pot, stacks, how it ended), and the action
        # abstraction produces far fewer distinct ones than that. Every worker
        # process holds its own tree, so the saving is per worker.
        self._terminals: dict[tuple, TerminalOutcome] = {}

        self._enumerate()
        self._build_layout()

    # ---- enumeration -----------------------------------------------------

    def _enumerate(self) -> None:
        root = self.rules.create_initial_state(
            starting_stack=self.starting_stack,
            hole_cards=_PLACEHOLDER_HOLE,
            button=0,
        )
        self.root_id = self._register(root)

    def _register(self, state: GameState) -> int:
        """Id of the decision node ``state`` sits at, enumerating it if new.

        Deals the placeholder board first when the state arrives mid-chance, so
        every registered node has a board matching its street. Recursion happens
        only for a node seen for the first time — the walk is a tree, so a
        repeat key would re-derive an identical subtree.
        """
        needed = state.street.board_card_count
        if len(state.board) < needed:
            # The betting tree is card-independent, so one representative board
            # stands in for every runout.
            state = dataclasses.replace(state, board=_PLACEHOLDER_BOARD[:needed])

        key = (state.street, state.normalized_betting_sequence())
        existing = self._index.get(key)
        if existing is not None:
            return existing

        legal_actions = self.rules.get_legal_actions(state, action_model=self.action_model)
        if not legal_actions:
            raise ValueError(f"Non-terminal state with no legal actions: {state}")

        node = BettingNode(
            node_id=len(self.nodes),
            street=state.street,
            betting_sequence=key[1],
            actor_is_button=state.current_player == state.button_position,
            legal_actions=tuple(legal_actions),
        )
        self._index[key] = node.node_id
        self.nodes.append(node)
        # Reserve this node's slot before recursing: children append after it,
        # so the list stays index-aligned with `nodes` under a depth-first walk.
        self.edges.append(())
        self.edges[node.node_id] = tuple(self._edge(state, action) for action in node.legal_actions)
        return node.node_id

    def _edge(self, state: GameState, action: Action) -> Edge:
        """Record where ``action`` leads, reading the answer off the rules engine.

        Nothing here re-derives the game: the child comes from the same
        ``apply_action`` the traversal used to call per visit, and the terminal
        payoffs come off that child's own pot and stacks. The enumeration is
        simply the last time anyone has to ask.
        """
        child = self.rules.apply_action(state, action)
        if child.is_terminal:
            key = (
                child.pot,
                child.stacks,
                child.ended_by_fold,
                child.current_player,
                len(child.board),
            )
            outcome = self._terminals.get(key)
            if outcome is None:
                outcome = self._terminals[key] = _terminal_outcome(child)
            return Edge(-1, 0, outcome)

        deal = child.street.board_card_count - len(child.board)
        return Edge(self._register(child), deal, None)

    # ---- layout ----------------------------------------------------------

    def _build_layout(self) -> None:
        n = len(self.nodes)
        self.num_actions = np.zeros(n, dtype=np.int64)
        rows_per_node = np.zeros(n, dtype=np.int64)

        for node in self.nodes:
            self.num_actions[node.node_id] = node.num_actions
            rows_per_node[node.node_id] = self.num_buckets(node.street)

        # Kept as an array, not recomputed from street: the storage bounds check
        # runs once per node visit, and an array index beats a Street comparison
        # plus dict lookup on that path.
        self.buckets_per_node = rows_per_node

        # Exclusive prefix sums: node n's rows/slots begin at offset[n].
        self.row_offset = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(rows_per_node, out=self.row_offset[1:])

        slots_per_node = rows_per_node * self.num_actions
        self.slot_offset = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(slots_per_node, out=self.slot_offset[1:])

        self.num_rows = int(self.row_offset[-1])
        self.num_slots = int(self.slot_offset[-1])

        # Plain-Python mirrors of the three arrays the traversal indexes once
        # per node visit. Pulling a scalar out of an int64 array boxes a numpy
        # object and costs several times a list index, and this is the hottest
        # read in the solver.
        self.row_offset_list: list[int] = self.row_offset.tolist()
        self.slot_offset_list: list[int] = self.slot_offset.tolist()
        self.num_actions_list: list[int] = self.num_actions.tolist()

        # Everything the traversal reads at a node, denormalized into one tuple
        # so a visit costs one list index and one unpack instead of a walk
        # through three arrays and a dataclass.
        self.node_spec: list[tuple] = [
            (
                node.street == Street.PREFLOP,
                node.actor_is_button,
                node.street,
                node.num_actions,
                self.row_offset_list[node.node_id],
                self.slot_offset_list[node.node_id],
                int(rows_per_node[node.node_id]),
                self.edges[node.node_id],
            )
            for node in self.nodes
        ]

    def num_buckets(self, street: Street) -> int:
        """Rows this street's nodes own — 169 canonical hands preflop, else buckets."""
        if street == Street.PREFLOP:
            return NUM_PREFLOP_HANDS
        return self.buckets_per_street[street]

    # ---- lookup ----------------------------------------------------------

    def node_id(self, state: GameState) -> int:
        """Dense id of the node ``state`` sits at.

        Raises when the state is off-tree, which in practice means a stack depth
        or action abstraction the tree was not built for. Failing loudly here is
        the point: the previous string-keyed design would silently allocate a
        fresh infoset instead.
        """
        key = (state.street, state.normalized_betting_sequence())
        node_id = self._index.get(key)
        if node_id is None:
            raise KeyError(
                f"State is off-tree: street={state.street}, "
                f"betting_sequence={key[1]!r}. The tree was enumerated for "
                f"starting_stack={self.starting_stack} under this action model; "
                "a different stack depth or action abstraction needs its own tree."
            )
        return node_id

    def row(self, node_id: int, bucket: int) -> int:
        """Flat infoset row for ``bucket`` at ``node_id``."""
        return int(self.row_offset[node_id]) + bucket

    def slots(self, node_id: int, bucket: int) -> tuple[int, int]:
        """Half-open slot range ``[start, end)`` backing one infoset's action vector."""
        width = int(self.num_actions[node_id])
        start = int(self.slot_offset[node_id]) + bucket * width
        return start, start + width

    def legal_actions(self, node_id: int) -> tuple[Action, ...]:
        return self.nodes[node_id].legal_actions

    def fingerprint(self) -> str:
        """Stable hash of everything that fixes what a stored row MEANS.

        A checkpoint is a bare array of numbers; the tree is the only thing that
        says which infoset each row belongs to. Loading one against a different
        tree would not fail — it would silently reinterpret every row as a
        different infoset, and training would continue on scrambled regrets. So
        this covers node identity and order, per-node action counts, and the
        per-street bucket counts: change any of them and the layout or the
        meaning changes.

        Node ORDER matters as much as membership, since ids are assigned by DFS
        order and the offsets follow from it.
        """
        digest = hashlib.sha256()
        digest.update(b"betting-tree-v1")
        digest.update(str(self.starting_stack).encode())
        for street in sorted(self.buckets_per_street, key=lambda s: s.name):
            digest.update(f"{street.name}={self.num_buckets(street)};".encode())
        for node in self.nodes:
            digest.update(
                f"{node.street.name}|{node.betting_sequence}|{node.num_actions};".encode()
            )
        return digest.hexdigest()[:16]

    def __len__(self) -> int:
        return len(self.nodes)

    def __str__(self) -> str:
        by_street: dict[str, int] = {}
        for node in self.nodes:
            by_street[str(node.street)] = by_street.get(str(node.street), 0) + 1
        breakdown = ", ".join(f"{s}={c}" for s, c in by_street.items())
        return (
            f"BettingTree(nodes={len(self.nodes)}, {breakdown}, "
            f"rows={self.num_rows:,}, slots={self.num_slots:,})"
        )


def _terminal_outcome(state: GameState) -> TerminalOutcome:
    """Tabulate every payoff a terminal state can produce.

    Mirrors ``GameRules.get_payoff`` expression for expression, including the
    order of the additions, so the tabulated value is the same float the live
    formula would return rather than merely the same number in exact
    arithmetic. Enumeration runs button-relative with ``button=0``, so seat 0
    here IS the button.
    """
    pot = state.pot
    stacks = state.stacks
    starting = (pot + stacks[0] + stacks[1]) / 2
    share = pot / 2

    win = ((stacks[0] + pot) - starting, (stacks[1] + pot) - starting)
    lose = (stacks[0] - starting, stacks[1] - starting)
    tie = ((stacks[0] + share) - starting, (stacks[1] + share) - starting)

    if state.ended_by_fold:
        # `current_player` on a terminal state is whoever folded into it.
        winner = 1 - state.current_player
        fold = (win[0], lose[1]) if winner == 0 else (lose[0], win[1])
        return TerminalOutcome(True, 0, fold, win, lose, tie)

    return TerminalOutcome(False, 5 - len(state.board), (0.0, 0.0), win, lose, tie)


def buckets_from_abstraction(card_abstraction: BucketingStrategy) -> dict[Street, int]:
    """Read per-street bucket counts off a bucketing strategy."""
    return {
        street: card_abstraction.num_buckets(street)
        for street in (Street.FLOP, Street.TURN, Street.RIVER)
    }


def build_betting_tree(
    rules: GameRules,
    action_model: ActionModel,
    card_abstraction: BucketingStrategy,
    *,
    starting_stack: int,
) -> BettingTree:
    """Build the tree for a solver's game/abstraction pairing."""
    return BettingTree(
        rules,
        action_model,
        starting_stack=starting_stack,
        buckets_per_street=buckets_from_abstraction(card_abstraction),
    )


__all__: Sequence[str] = (
    "NUM_PREFLOP_HANDS",
    "BettingNode",
    "BettingTree",
    "Edge",
    "TerminalOutcome",
    "buckets_from_abstraction",
    "build_betting_tree",
)
