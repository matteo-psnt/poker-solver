"""The betting tree is the foundation of integer infoset indexing.

Everything downstream — the flat storage layout, the deletion of the id
allocation protocol, the removal of ``spr_bucket`` and ``player_position`` —
rests on three claims proved here: the enumeration is *complete* (a live
traversal never reaches a node it does not contain), node identity is *unique*
(one key never merges two structurally distinct nodes), and the tree is
*button-symmetric* (the absolute seat carries no information).
"""

from __future__ import annotations

import dataclasses
import random

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Card, Street
from src.engine.solver.betting_tree import NUM_PREFLOP_HANDS, BettingTree
from src.engine.solver.infoset.index import (
    infoset_row,
    preflop_hand_index,
)
from tests.test_helpers import make_test_config

BOARD = (Card("2c"), Card("7d"), Card("9h"), Card("4s"), Card("Ts"))
HOLE = ((Card("As"), Card("Kd")), (Card("Qh"), Card("Jc")))

# A deliberately small action abstraction: the structural claims are
# abstraction-independent, and a 2-raise cap keeps enumeration fast enough for
# the default 5s timeout.
BUCKETS = {Street.FLOP: 4, Street.TURN: 5, Street.RIVER: 6}


@pytest.fixture(scope="module")
def config():
    return make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=40)


@pytest.fixture(scope="module")
def rules(config):
    return GameRules(small_blind=config.game.small_blind, big_blind=config.game.big_blind)


@pytest.fixture(scope="module")
def tree(rules, config):
    return BettingTree(
        rules,
        ActionModel(config),
        starting_stack=config.game.starting_stack,
        buckets_per_street=BUCKETS,
    )


def _walk_states(rules, action_model, state, visit, depth=0):
    """Depth-first walk yielding every decision state, dealing boards as needed."""
    if state.is_terminal or depth > 60:
        return
    needed = state.street.board_card_count
    if len(state.board) < needed:
        _walk_states(
            rules, action_model, dataclasses.replace(state, board=BOARD[:needed]), visit, depth + 1
        )
        return
    actions = rules.get_legal_actions(state, action_model=action_model)
    if not actions:
        return
    visit(state)
    for action in actions:
        _walk_states(rules, action_model, rules.apply_action(state, action), visit, depth + 1)


class TestEnumeration:
    def test_tree_is_non_trivial(self, tree):
        assert len(tree) > 100
        assert tree.num_rows > 0
        assert tree.num_slots > 0

    def test_enumeration_is_deterministic(self, rules, config):
        second = BettingTree(
            rules,
            ActionModel(config),
            starting_stack=config.game.starting_stack,
            buckets_per_street=BUCKETS,
        )
        first = BettingTree(
            rules,
            ActionModel(config),
            starting_stack=config.game.starting_stack,
            buckets_per_street=BUCKETS,
        )
        assert [n.betting_sequence for n in first.nodes] == [
            n.betting_sequence for n in second.nodes
        ]

    def test_node_identity_never_merges_distinct_states(self, rules, config, tree):
        """One (street, betting_sequence) key => exactly one structural state.

        If the flat cross-street token string could merge two different nodes,
        the whole indexing scheme would silently alias two infosets onto one row.
        """
        signatures: dict[int, set] = {}

        def visit(state):
            node_id = tree.node_id(state)
            actor_is_button = state.current_player == state.button_position
            button_relative_stacks = (
                state.stacks[state.button_position],
                state.stacks[1 - state.button_position],
            )
            signatures.setdefault(node_id, set()).add(
                (actor_is_button, state.pot, button_relative_stacks, state.to_call)
            )

        for button in (0, 1):
            root = rules.create_initial_state(config.game.starting_stack, HOLE, button=button)
            _walk_states(rules, ActionModel(config), root, visit)

        ambiguous = {k: v for k, v in signatures.items() if len(v) > 1}
        assert not ambiguous, f"{len(ambiguous)} node keys merge distinct states"

    def test_live_traversal_never_goes_off_tree(self, rules, config, tree):
        """Completeness: every reachable decision state maps to an enumerated node."""
        seen: set[int] = set()

        def visit(state):
            seen.add(tree.node_id(state))  # raises KeyError if unenumerated

        for button in (0, 1):
            root = rules.create_initial_state(config.game.starting_stack, HOLE, button=button)
            _walk_states(rules, ActionModel(config), root, visit)

        assert seen == set(range(len(tree))), "enumeration and live traversal disagree"

    def test_off_tree_state_raises_rather_than_aliasing(self, rules, config, tree):
        """A different stack depth must fail loudly, not land on a wrong row."""
        action_model = ActionModel(config)
        deep = rules.create_initial_state(config.game.starting_stack * 3, HOLE, button=0)
        # Walk to a state whose betting sequence cannot exist in the shallow tree.
        for _ in range(3):
            actions = rules.get_legal_actions(deep, action_model=action_model)
            raises = [a for a in actions if a.amount and a.amount > 0]
            if not raises:
                break
            deep = rules.apply_action(deep, raises[-1])

        with pytest.raises(KeyError, match="off-tree"):
            tree.node_id(deep)

    def test_actions_recorded_match_the_action_model(self, rules, config, tree):
        action_model = ActionModel(config)

        def visit(state):
            node = tree.nodes[tree.node_id(state)]
            live = rules.get_legal_actions(state, action_model=action_model)
            assert tuple(live) == node.legal_actions

        root = rules.create_initial_state(config.game.starting_stack, HOLE, button=0)
        _walk_states(rules, action_model, root, visit)


class TestButtonSymmetry:
    def test_both_buttons_yield_the_same_nodes(self, rules, config, tree):
        """The claim that justifies dropping player_position from the key.

        If enumerating from either button produces the same key set with the same
        button-relative structure, then the absolute seat is not part of the
        strategic situation, and keying on it doubles the infoset space for
        nothing.
        """
        action_model = ActionModel(config)
        per_button = []
        for button in (0, 1):
            collected: dict[tuple, tuple] = {}

            def visit(state, out=collected):
                key = (state.street, state.normalized_betting_sequence())
                out[key] = (
                    state.current_player == state.button_position,
                    state.pot,
                    (
                        state.stacks[state.button_position],
                        state.stacks[1 - state.button_position],
                    ),
                    state.to_call,
                    len(rules.get_legal_actions(state, action_model=action_model)),
                )

            root = rules.create_initial_state(config.game.starting_stack, HOLE, button=button)
            _walk_states(rules, action_model, root, visit)
            per_button.append(collected)

        assert per_button[0].keys() == per_button[1].keys()
        assert per_button[0] == per_button[1]

    def test_same_node_for_either_button(self, rules, config, tree):
        """The same situation resolves to one node id regardless of seat."""
        b0 = rules.create_initial_state(config.game.starting_stack, HOLE, button=0)
        b1 = rules.create_initial_state(config.game.starting_stack, HOLE, button=1)
        assert b0.current_player != b1.current_player
        assert tree.node_id(b0) == tree.node_id(b1)


class TestLayout:
    def test_rows_and_slots_are_contiguous_and_exact(self, tree):
        """No gaps, no overlap, no padding waste across the ragged layout."""
        expected_rows = sum(tree.num_buckets(n.street) for n in tree.nodes)
        expected_slots = sum(tree.num_buckets(n.street) * n.num_actions for n in tree.nodes)
        assert tree.num_rows == expected_rows
        assert tree.num_slots == expected_slots

    def test_every_infoset_maps_to_a_unique_row(self, tree):
        rows = set()
        for node in tree.nodes:
            for bucket in range(tree.num_buckets(node.street)):
                rows.add(tree.row(node.node_id, bucket))
        assert len(rows) == tree.num_rows
        assert rows == set(range(tree.num_rows))

    def test_slot_ranges_tile_the_array_without_overlap(self, tree):
        # Layout order is street -> bucket -> node; the ranges must tile the
        # array exactly in that order, no gap and no overlap.
        cursor = 0
        for street in Street:
            nodes = [node for node in tree.nodes if node.street == street]
            if not nodes:
                continue
            for bucket in range(tree.num_buckets(street)):
                for node in nodes:
                    start, end = tree.slots(node.node_id, bucket)
                    assert start == cursor, "slot ranges must tile in layout order"
                    assert end - start == node.num_actions
                    cursor = end
        assert cursor == tree.num_slots


class TestProductionScale:
    """The small fixtures above cannot overflow anything; production can.

    At production bucket counts the tree spans ~89M action slots. If the offset
    arrays were int32 the last node's slot range would wrap silently, and every
    infoset past the wrap point would alias one before it — training would run,
    converge to nonsense, and nothing would raise. These assertions cost one tree
    build and close that off.
    """

    @pytest.fixture(scope="class")
    def production_tree(self):
        """Built the way a node builds it: bucket counts come from the CONFIG.

        Hardcoding the counts here would put a hole in the fingerprint guard
        below, in the one dimension it exists for — ``fingerprint()`` covers
        per-street bucket counts, so a river count moved 600 -> 800 orphans
        every checkpoint on the share, and a fixture that never consults
        ``config/abstraction/production.yaml`` would stay green through it.
        """
        from src.pipeline.abstraction.config import PrecomputeConfig
        from src.shared.config.loader import load_config

        config = load_config("config/training/production.yaml")
        rules = GameRules(small_blind=config.game.small_blind, big_blind=config.game.big_blind)
        return BettingTree(
            rules,
            ActionModel(config),
            starting_stack=config.game.starting_stack,
            buckets_per_street=PrecomputeConfig.from_yaml(
                config.card_abstraction.config
            ).num_buckets,
        )

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_production_layout_is_the_one_the_share_was_written_against(self, production_tree):
        """A golden fingerprint, and it is a LINEAGE GUARD — never re-pin it.

        A checkpoint on the share is a bare array of numbers; this tree is the
        only thing that says which infoset each row is. Change node identity,
        node ORDER, a node's action count or a per-street bucket count and the
        fingerprint moves — at which point every checkpoint ever written either
        refuses to load or, if the guard were relaxed, silently reinterprets
        every row as a different infoset and keeps training.

        So a failure here is never a stale constant. It means the layout moved,
        and the question is what happens to the runs on the share — not what to
        edit on this line. See ``static_checkpoint``, which refuses the load.
        """
        assert production_tree.buckets_per_street == {
            Street.FLOP: 100,
            Street.TURN: 300,
            Street.RIVER: 600,
        }, "the abstraction config moved; the fingerprint below moved with it"
        # Re-pinned 2026-08-23: the game gained the big blind's option behind a
        # limp (`c-x`, `c-b*` lines), a deliberate lineage break -- every
        # checkpoint written against ca50e2d3291fa227 / 57,604 nodes is
        # unloadable at HEAD by design.
        # Re-pinned 2026-08-24: bucket-major layout (betting-tree-v2), a
        # deliberate lineage break -- same game, same totals, but every row
        # moved address, so checkpoints against b0367ae018a58a2f refuse to
        # load rather than being read as a permutation.
        assert production_tree.fingerprint() == "37c7818a12cf353c"
        assert len(production_tree.nodes) == 81_518
        assert production_tree.num_rows == 45_538_298
        assert production_tree.num_slots == 125_463_187

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_offsets_are_int64(self, production_tree):
        assert production_tree.row_base.dtype == np.int64
        assert production_tree.row_stride.dtype == np.int64
        assert production_tree.slot_base.dtype == np.int64
        assert production_tree.slot_stride.dtype == np.int64
        assert production_tree.num_actions.dtype == np.int64

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_last_slot_range_closes_the_array_exactly(self, production_tree):
        """The final infoset must end precisely at num_slots — no wrap, no gap."""
        tree = production_tree
        assert tree.num_slots > np.iinfo(np.int32).max // 32, (
            "production tree unexpectedly small; this test would prove nothing"
        )
        # Layout-last is the LAST STREET's last node at its top bucket, not
        # tree.nodes[-1] -- ids are DFS order, the layout is street-major.
        last_street = [s for s in Street if any(n.street == s for n in tree.nodes)][-1]
        last = [n for n in tree.nodes if n.street == last_street][-1]
        _, end = tree.slots(last.node_id, tree.num_buckets(last_street) - 1)
        assert end == tree.num_slots
        assert tree.row(last.node_id, tree.num_buckets(last_street) - 1) == tree.num_rows - 1

    @pytest.mark.slow
    @pytest.mark.timeout(120)
    def test_all_slot_offsets_are_monotonic(self, production_tree):
        """A wrap would show up as a negative base or stride somewhere."""
        tree = production_tree
        assert np.all(tree.row_base >= 0)
        assert np.all(tree.slot_base >= 0)
        assert np.all(tree.row_stride > 0)
        assert np.all(tree.slot_stride > 0)


class TestPreflopIndex:
    def test_covers_exactly_169_hands(self):
        seen = set()
        ranks = "AKQJT98765432"
        suits = "shdc"
        for i, r1 in enumerate(ranks):
            for r2 in ranks[i:]:
                if r1 == r2:
                    seen.add(preflop_hand_index((Card(r1 + "s"), Card(r2 + "h"))))
                else:
                    seen.add(preflop_hand_index((Card(r1 + "s"), Card(r2 + "s"))))
                    seen.add(preflop_hand_index((Card(r1 + "s"), Card(r2 + "h"))))
        assert seen == set(range(NUM_PREFLOP_HANDS))
        assert len(suits) == 4  # guard against typo in the loop above

    def test_order_independent_and_suit_symmetric(self):
        assert preflop_hand_index((Card("As"), Card("Kd"))) == preflop_hand_index(
            (Card("Kd"), Card("As"))
        )
        assert preflop_hand_index((Card("As"), Card("Ks"))) == preflop_hand_index(
            (Card("Ah"), Card("Kh"))
        )

    def test_suited_and_offsuit_are_distinct(self):
        assert preflop_hand_index((Card("As"), Card("Ks"))) != preflop_hand_index(
            (Card("As"), Card("Kd"))
        )


class TestInfosetRow:
    def test_rows_stay_in_bounds_over_random_play(self, rules, config, tree):
        """Fuzz: any reachable infoset indexes inside the preallocated array."""

        class FixedBuckets:
            def get_bucket(self, hole_cards, board, street):
                # sum(ord(...)), not hash(): hash is randomised per process, so
                # bucket assignment differed between runs of identical code.
                # This fuzz passes card STRINGS, so rank_eval7 is unavailable.
                seed = sum(map(ord, repr(hole_cards[0]))) + street.value
                return (seed % BUCKETS[street] + BUCKETS[street]) % BUCKETS[street]

            def num_buckets(self, street):
                return BUCKETS[street]

        abstraction = FixedBuckets()
        action_model = ActionModel(config)
        rng = random.Random(7)

        for trial in range(200):
            state = rules.create_initial_state(config.game.starting_stack, HOLE, button=trial % 2)
            guard = 0
            while not state.is_terminal and guard < 60:
                guard += 1
                needed = state.street.board_card_count
                if len(state.board) < needed:
                    state = dataclasses.replace(state, board=BOARD[:needed])
                    continue
                actions = rules.get_legal_actions(state, action_model=action_model)
                if not actions:
                    break
                row = infoset_row(state, state.current_player, tree, abstraction)
                assert 0 <= row < tree.num_rows
                state = rules.apply_action(state, rng.choice(actions))
