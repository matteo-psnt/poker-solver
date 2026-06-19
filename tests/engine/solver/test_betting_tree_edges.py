"""Does the edge table say what the rules engine says — at every edge?

``tree_traversal`` stopped asking ``GameRules`` where an action leads and what a
terminal pays; it reads :class:`Edge` and :class:`TerminalOutcome` instead. The
bit-identity test next door checks that substitution by *sampling*: two
traversals, one seed, a couple of hundred iterations. Sampling reaches the
edges MCCFR happens to walk, and the ones it misses are the ones nobody would
notice were wrong — a rare all-in line paying the wrong constant costs a little
accuracy on a path taken a little of the time, forever, silently.

So this checks the table exhaustively, against the only authority there is. It
re-walks the whole tree carrying a live ``GameState`` beside the node id, and at
every node, for every action, asserts that:

    the child the table names        == the node ``rules.apply_action`` lands on
    the board cards the table owes   == the cards that child is short
    what the table says a hand pays  == ``rules.get_payoff`` on that terminal

The payoff check needs a winner, and a fold's is fixed while a showdown's comes
from cards. So the walk runs once per outcome, with hole cards chosen to make
the board play out as a button win, a non-button win, and a tie — which is how
all three columns of the table get read.
"""

from __future__ import annotations

import dataclasses

import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Card, GameState, Street
from src.engine.solver.betting_tree import build_betting_tree
from src.shared.config.loader import load_training_config
from tests.test_helpers import make_test_config

# A royal flush on board: any two hole cards play it, so "both players tie" is
# a property of the board and needs no coincidence between the holdings.
TIE_BOARD = (Card.new("As"), Card.new("Ks"), Card.new("Qs"), Card.new("Js"), Card.new("Ts"))
# A board nobody improves on, so the hole cards alone decide it.
SPLIT_BOARD = (Card.new("2c"), Card.new("7d"), Card.new("9h"), Card.new("4s"), Card.new("Td"))

# (label, hole cards, board, winning seat) — seat 0 is the button, because the
# enumeration walks button=0 and the payoff table is stored button-relative.
OUTCOMES = (
    (
        "button wins",
        ((Card.new("Ac"), Card.new("Ad")), (Card.new("3c"), Card.new("5d"))),
        SPLIT_BOARD,
        0,
    ),
    (
        "non-button wins",
        ((Card.new("3c"), Card.new("5d")), (Card.new("Ac"), Card.new("Ad"))),
        SPLIT_BOARD,
        1,
    ),
    ("tie", ((Card.new("2h"), Card.new("3d")), (Card.new("4c"), Card.new("5h"))), TIE_BOARD, -1),
)


class Buckets:
    """Bucket counts only decide how many ROWS a node owns, never its edges."""

    def __init__(self, counts):
        self.counts = counts

    def get_bucket(self, hole_cards, board, street):
        return 0

    def num_buckets(self, street):
        return self.counts.get(street, 169)


def _tree_for(config, counts):
    action_model = ActionModel(config)
    rules = GameRules(config.game.small_blind, config.game.big_blind)
    tree = build_betting_tree(
        rules, action_model, Buckets(counts), starting_stack=config.game.starting_stack
    )
    return rules, tree


def _with_board(state: GameState, board: tuple[Card, ...]) -> GameState:
    """Board filled to whatever the state's street expects."""
    needed = state.street.board_card_count
    if len(state.board) == needed:
        return state
    return state.replace(board=board[:needed], validate=False)


def _expected_payoff(terminal, seat: int, winner: int) -> float:
    if terminal.is_fold:
        return terminal.fold[seat]
    if winner == -1:
        return terminal.tie[seat]
    return terminal.win[seat] if winner == seat else terminal.lose[seat]


def _check_every_edge(rules, tree, hole_cards, board, winner) -> tuple[int, dict[str, int]]:
    """Re-derive the whole table from the rules engine. Returns (nodes, tallies)."""
    root = rules.create_initial_state(
        starting_stack=tree.starting_stack, hole_cards=hole_cards, button=0
    )
    seen: set[int] = set()
    tally = {"fold": 0, "showdown_complete": 0, "showdown_owed": 0, "deals": 0}

    def walk(node_id: int, state: GameState) -> None:
        assert node_id not in seen, (
            f"node {node_id} reached twice — the enumeration is not a tree, and "
            "every id would then have two parents' edges written over it"
        )
        seen.add(node_id)

        node = tree.nodes[node_id]
        assert tree.node_id(state) == node_id
        assert node.actor_is_button == (state.current_player == state.button_position)
        legal = rules.get_legal_actions(state, action_model=tree.action_model)
        assert tuple(legal) == node.legal_actions

        edges = tree.edges[node_id]
        assert len(edges) == len(node.legal_actions)

        for edge, action in zip(edges, node.legal_actions, strict=True):
            child = rules.apply_action(state, action)

            if not child.is_terminal:
                assert edge.terminal is None, f"node {node_id}/{action} is not terminal"
                owed = child.street.board_card_count - len(child.board)
                assert edge.deal == owed
                tally["deals"] += owed > 0
                walk(edge.child_id, _with_board(child, board))
                continue

            terminal = edge.terminal
            assert terminal is not None, f"node {node_id}/{action} ends the hand"
            assert terminal.is_fold == child.ended_by_fold
            assert terminal.cards_to_deal == (0 if child.ended_by_fold else 5 - len(child.board))

            # A showdown is settled on the full five cards, exactly as
            # `chance.deal_remaining_cards` would leave it — and `cards_to_deal`
            # is the count it would have had to draw. A fold's board is ignored.
            settled = (
                child
                if child.ended_by_fold
                else child.replace(street=Street.RIVER, board=board[:5], validate=False)
            )
            for seat in (0, 1):
                assert rules.get_payoff(settled, seat) == _expected_payoff(
                    terminal, seat, winner
                ), f"payoff mismatch at node {node_id}, action {action}, seat {seat}"

            if terminal.is_fold:
                tally["fold"] += 1
            elif terminal.cards_to_deal:
                tally["showdown_owed"] += 1
            else:
                tally["showdown_complete"] += 1

    walk(tree.root_id, root)
    return len(seen), tally


@pytest.mark.parametrize(("label", "hole_cards", "board", "winner"), OUTCOMES)
def test_every_edge_agrees_with_the_rules_engine(label, hole_cards, board, winner):
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=20)
    rules, tree = _tree_for(config, {Street.FLOP: 2, Street.TURN: 2, Street.RIVER: 2})

    reached, tally = _check_every_edge(rules, tree, hole_cards, board, winner)

    assert reached == len(tree.nodes), "the edge table does not reach every enumerated node"
    # The tallies are the proof the walk was not trivial: all three terminal
    # shapes and at least one street transition were actually checked.
    assert tally["fold"] > 0
    assert tally["showdown_complete"] > 0
    assert tally["showdown_owed"] > 0
    assert tally["deals"] > 0


@pytest.mark.timeout(60)
def test_every_edge_agrees_at_production_scale():
    """The small tree cannot produce production's edge shapes.

    ``production.yaml`` adds the preflop templates, `min_raise`/`pot_raise`/
    `jam` and five raises a street — 225,055 edges, including the sizings that
    resolve to the whole stack, the all-in-for-less that the default model
    never reaches, and the big blind's option behind a limp. One outcome column is enough here; the three-column check
    above is what the arithmetic needs, and this is what the SHAPES need.
    """
    config = load_training_config("production")
    rules, tree = _tree_for(config, {Street.FLOP: 2, Street.TURN: 2, Street.RIVER: 2})

    _, hole_cards, board, winner = OUTCOMES[0]
    reached, tally = _check_every_edge(rules, tree, hole_cards, board, winner)

    assert reached == len(tree.nodes) == 81_518
    assert sum(len(edges) for edges in tree.edges) == 225_055
    assert tally["fold"] > 0
    assert tally["showdown_complete"] > 0
    assert tally["showdown_owed"] > 0


@pytest.mark.parametrize(
    ("label", "corrupt"),
    [
        (
            "payoff",
            lambda e: dataclasses.replace(
                e,
                terminal=dataclasses.replace(e.terminal, win=e.terminal.lose, lose=e.terminal.win),
            ),
        ),
        (
            "cards owed",
            lambda e: dataclasses.replace(
                e,
                terminal=dataclasses.replace(
                    e.terminal, cards_to_deal=e.terminal.cards_to_deal + 1
                ),
            ),
        ),
        (
            "fold flag",
            lambda e: dataclasses.replace(
                e, terminal=dataclasses.replace(e.terminal, is_fold=not e.terminal.is_fold)
            ),
        ),
    ],
)
def test_the_exhaustive_walk_would_notice_a_corrupt_terminal(label, corrupt):
    """The walk above passes. Show that it can fail.

    A comparison against an oracle proves nothing until you have watched it
    reject something — and the failure modes worth pinning are the three ways
    a terminal row can be wrong: the wrong payoff, the wrong runout length, and
    the wrong idea of how the hand ended.
    """
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=20)
    rules, tree = _tree_for(config, {Street.FLOP: 2, Street.TURN: 2, Street.RIVER: 2})

    corrupted = 0
    for node_id, edges in enumerate(tree.edges):
        rebuilt = []
        for edge in edges:
            if edge.terminal is None:
                rebuilt.append(edge)
                continue
            rebuilt.append(corrupt(edge))
            corrupted += 1
        tree.edges[node_id] = tuple(rebuilt)
    assert corrupted > 0

    with pytest.raises(AssertionError):
        _check_every_edge(rules, tree, *OUTCOMES[0][1:])


def test_the_table_is_a_tree_with_one_parent_per_node():
    """Every node but the root is named by exactly one edge.

    If two nodes shared a child the walk above would still pass on the first
    visit and the traversal would still run — but the DFS ids the storage
    layout is built from would no longer describe a tree.
    """
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=20)
    _, tree = _tree_for(config, {Street.FLOP: 2, Street.TURN: 2, Street.RIVER: 2})

    parents: dict[int, int] = {}
    for node_id, edges in enumerate(tree.edges):
        for edge in edges:
            if edge.terminal is None:
                assert edge.child_id not in parents, (
                    f"node {edge.child_id} is a child of both {parents.get(edge.child_id)} "
                    f"and {node_id}"
                )
                parents[edge.child_id] = node_id

    assert tree.root_id == 0
    assert set(parents) == set(range(1, len(tree.nodes)))


def test_node_spec_and_the_layout_accessors_cannot_disagree():
    """``tree_traversal`` inlines ``tree.row``/``tree.slots``. Pin them together.

    The traversal computes ``row_base + bucket`` and ``slot_base + bucket *
    num_actions`` from ``node_spec`` rather than calling the accessors, because
    a call per node visit is the thing this change exists to remove. That is a
    second copy of the layout arithmetic, and a drift between the two would
    not crash — it would silently read another infoset's regrets.
    """
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=20)
    _, tree = _tree_for(config, {Street.FLOP: 3, Street.TURN: 4, Street.RIVER: 5})

    for node in tree.nodes:
        spec = tree.node_spec[node.node_id]
        is_preflop, actor_is_button, street, num_actions, row_base, slot_base, buckets, edges = spec

        assert is_preflop == (node.street == Street.PREFLOP)
        assert actor_is_button == node.actor_is_button
        assert street == node.street
        assert num_actions == node.num_actions
        assert buckets == tree.num_buckets(node.street)
        assert edges is tree.edges[node.node_id]

        for bucket in range(buckets):
            assert row_base + bucket == tree.row(node.node_id, bucket)
            assert (
                slot_base + bucket * num_actions,
                slot_base + bucket * num_actions + num_actions,
            ) == tree.slots(node.node_id, bucket)
