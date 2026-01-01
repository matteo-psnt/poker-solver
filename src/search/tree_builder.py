"""Local lookahead tree builder for HU resolver."""

from __future__ import annotations

from dataclasses import dataclass, field

from src.actions.action_model import ActionModel
from src.game.actions import Action
from src.game.rules import GameRules
from src.game.state import GameState, Street


@dataclass
class LocalTreeNode:
    """Single node in a bounded local lookahead tree."""

    state: GameState
    depth: int
    action_from_parent: Action | None = None
    actions: list[Action] = field(default_factory=list)
    children: list["LocalTreeNode"] = field(default_factory=list)
    is_leaf: bool = False


@dataclass
class LocalTree:
    """Container for resolver lookahead tree."""

    root: LocalTreeNode

    @property
    def leaves(self) -> list[LocalTreeNode]:
        nodes: list[LocalTreeNode] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            if node.is_leaf or not node.children:
                nodes.append(node)
            else:
                stack.extend(node.children)
        return nodes


def build_local_tree(
    state: GameState,
    *,
    action_model: ActionModel,
    rules: GameRules,
    max_depth: int,
) -> LocalTree:
    """
    Build a depth-limited local tree from the current public state.

    This builder expands only player actions (not chance fanout), which keeps
    runtime stable under tight resolver budgets.
    """
    root = LocalTreeNode(state=state, depth=0)
    _expand(root, action_model=action_model, rules=rules, max_depth=max_depth)
    return LocalTree(root=root)


def _expand(
    node: LocalTreeNode,
    *,
    action_model: ActionModel,
    rules: GameRules,
    max_depth: int,
) -> None:
    if node.depth >= max_depth or node.state.is_terminal:
        node.is_leaf = True
        return
    if _is_chance_node(node.state):
        node.is_leaf = True
        return

    # Skip explicit chance expansion to keep branching bounded.
    actions = action_model.get_legal_actions(node.state)
    if not actions:
        node.is_leaf = True
        return

    node.actions = actions
    for action in actions:
        next_state = node.state.apply_action(action, rules)
        child = LocalTreeNode(
            state=next_state,
            depth=node.depth + 1,
            action_from_parent=action,
        )
        node.children.append(child)
        _expand(child, action_model=action_model, rules=rules, max_depth=max_depth)


def _is_chance_node(state: GameState) -> bool:
    expected_board_size = {
        Street.PREFLOP: 0,
        Street.FLOP: 3,
        Street.TURN: 4,
        Street.RIVER: 5,
    }
    return len(state.board) < expected_board_size[state.street]
