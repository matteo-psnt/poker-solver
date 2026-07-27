"""Static enumeration of the public betting tree.

The public betting tree of HUNL under a fixed action abstraction and stack depth
is *small* and *finite*: 57,604 decision nodes under `config/training/production.yaml`,
enumerable in seconds. Every information set the solver will ever touch is a pair

    (public betting node, card bucket)

and both components are known before training starts. That makes the entire
infoset space a preallocated flat array indexed by integer arithmetic, which is
why this module exists.

What it replaces: discovering infosets at runtime by hashing a string key,
allocating an id, and reconciling those ids across worker processes. That
machinery cost ~3,000 lines and produced the cross-worker id-frontier stall, the
fork-OOM, the capacity/resize path, and a measured 39-74% rate of dropped
updates. None of those failure modes have an analogue here: there is nothing to
discover, so there is nothing to reconcile.

Node identity:
    A node is keyed by ``(street, betting_sequence)``. Verified unique — across
    the full production enumeration, every such key maps to exactly one
    ``(actor, pot, stacks, to_call)``, so the flat cross-street token string does
    not merge structurally distinct nodes. See ``test_betting_tree.py``.

Button relativity:
    Nodes are enumerated once, button-relative. The tree is *exactly*
    button-symmetric: enumerating from button=0 and button=1 yields identical key
    sets with identical ``(relative actor, pot, button-relative stacks, to_call,
    action count)``. The absolute seat therefore carries no strategic
    information, and keying infosets on it — as ``InfoSetKey.player_position``
    does — duplicates the whole space, splitting an already starved update budget
    across two identical halves. This module keys on the acting seat *relative to
    the button* instead.

Stack depth:
    The enumeration is valid for one starting stack. A state that does not map to
    an enumerated node raises rather than silently landing on a wrong row — the
    stack-depth assumption the SPR bucket used to paper over is now enforced.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.core.actions.action_model import ActionModel
from src.core.game.actions import Action
from src.core.game.rules import GameRules
from src.core.game.state import Card, GameState, Street

# Enumeration needs a concrete state to walk, but the betting tree does not
# depend on card identities — only on street, pot, stacks and to_call. These are
# placeholders that get overwritten as the walk advances streets.
_PLACEHOLDER_HOLE = ((Card("As"), Card("Kd")), (Card("Qh"), Card("Jc")))
_PLACEHOLDER_BOARD = (Card("2c"), Card("7d"), Card("9h"), Card("4s"), Card("Ts"))

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

        self._enumerate()
        self._build_layout()

    # ---- enumeration -----------------------------------------------------

    def _enumerate(self) -> None:
        root = self.rules.create_initial_state(
            starting_stack=self.starting_stack,
            hole_cards=_PLACEHOLDER_HOLE,
            button=0,
        )
        self._walk(root)

    def _walk(self, state: GameState) -> None:
        if state.is_terminal:
            return

        needed = state.street.board_card_count
        if len(state.board) < needed:
            # Chance node: the betting tree is card-independent, so one
            # representative board stands in for every runout.
            self._walk(dataclasses.replace(state, board=_PLACEHOLDER_BOARD[:needed]))
            return

        legal_actions = self.rules.get_legal_actions(state, action_model=self.action_model)
        if not legal_actions:
            return

        key = (state.street, state.normalized_betting_sequence())
        if key not in self._index:
            node = BettingNode(
                node_id=len(self.nodes),
                street=state.street,
                betting_sequence=key[1],
                actor_is_button=state.current_player == state.button_position,
                legal_actions=tuple(legal_actions),
            )
            self._index[key] = node.node_id
            self.nodes.append(node)

        for action in legal_actions:
            self._walk(self.rules.apply_action(state, action))

    # ---- layout ----------------------------------------------------------

    def _build_layout(self) -> None:
        n = len(self.nodes)
        self.num_actions = np.zeros(n, dtype=np.int64)
        rows_per_node = np.zeros(n, dtype=np.int64)

        for node in self.nodes:
            self.num_actions[node.node_id] = node.num_actions
            rows_per_node[node.node_id] = self.num_buckets(node.street)

        # Exclusive prefix sums: node n's rows/slots begin at offset[n].
        self.row_offset = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(rows_per_node, out=self.row_offset[1:])

        slots_per_node = rows_per_node * self.num_actions
        self.slot_offset = np.zeros(n + 1, dtype=np.int64)
        np.cumsum(slots_per_node, out=self.slot_offset[1:])

        self.num_rows = int(self.row_offset[-1])
        self.num_slots = int(self.slot_offset[-1])

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


def buckets_from_abstraction(card_abstraction) -> dict[Street, int]:
    """Read per-street bucket counts off a bucketing strategy."""
    return {
        street: card_abstraction.num_buckets(street)
        for street in (Street.FLOP, Street.TURN, Street.RIVER)
    }


def build_betting_tree(
    rules: GameRules,
    action_model: ActionModel,
    card_abstraction,
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
    "buckets_from_abstraction",
    "build_betting_tree",
)
