"""CFR-BR trains the blueprint against a best response in the REAL game.

Four things decide whether this is CFR-BR or an expensive way to run CFR. The
opponent's substituted strategy has to actually beat the regret-matched one it
replaced, or the substitution is not reaching the passes. It has to write the
CFR player's rows and nothing else, since its own table is scaffolding. It must
not be allowed to choose on cards that are still face down -- the failure that
would look like a triumph, because a clairvoyant opponent is a harder one. And
the point of the whole exercise is the last test here: real-game exploitability
below what plain CFR reaches on the SAME abstraction, bought by giving up
exploitability inside it.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Street
from src.engine.solver.betting_tree import BettingTree
from src.engine.solver.vector.cfr_br import BR_REGIONS, CFRBestResponse, TrunkLayout
from src.engine.solver.vector.compiled_tree import compile_tree
from src.engine.solver.vector.mixture import BoardMixtureCFR
from tests.engine.solver.vector.contexts import (
    MIN_SHOWDOWN_SIGNAL,
    prefix_consistent_contexts,
    showdown_signal,
)
from tests.test_helpers import make_test_config

# A 24-card deck and an 8-chip stack: 171 holdings over a 440-node tree, so a
# few hundred exact CFR-BR iterations fit a test. The bucket ORDERING is what
# carries the meaning (see ``contexts``), not the deck size.
DECK = 24
BUCKETS = {Street.FLOP: 3, Street.TURN: 3, Street.RIVER: 4}
ALL_COUNTS = {Street.PREFLOP: 169, **BUCKETS}
STACK = 8

# One flop, two turns, two rivers each. The chance layer of the game these tests
# solve IS these four boards, so a maximisation joint over the boards a street
# cannot yet tell apart is an exact best response at every street.
BOARDS = [[0, 1, 2, turn, river] for turn in (3, 4) for river in (5, 6)]


@pytest.fixture(scope="module")
def parts():
    config = make_test_config(seed=42, small_blind=1, big_blind=2, starting_stack=STACK)
    rules = GameRules(small_blind=1, big_blind=2)
    tree = BettingTree(rules, ActionModel(config), starting_stack=STACK, buckets_per_street=BUCKETS)
    compiled = compile_tree(tree, rules)
    contexts = prefix_consistent_contexts(BOARDS, ALL_COUNTS, num_cards=DECK)
    assert showdown_signal(contexts, ALL_COUNTS) > MIN_SHOWDOWN_SIGNAL
    pairs = float(np.mean([(~context.blocks).sum() for context in contexts]))
    return compiled, contexts, pairs


def _driver(compiled, region: str, num_boards: int, weighting: str = "none"):
    tree = compiled.tree
    layout = TrunkLayout(tree, BR_REGIONS[region])
    regrets = np.zeros(tree.num_slots, dtype=np.float32)
    strategy_sum = np.zeros(tree.num_slots, dtype=np.float32)
    trunk = np.zeros(max(1, layout.num_slots), dtype=np.float32)
    driver = CFRBestResponse(
        compiled,
        regrets,
        strategy_sum,
        trunk,
        br_streets=BR_REGIONS[region],
        weighting=weighting,
        cfr_plus=True,
        showdown="matmul",
        num_boards=num_boards,
    )
    return driver, regrets, strategy_sum, trunk


def _root_values(driver, contexts, boards):
    """(hybrid root value, regret-matched root value) per seat, same reaches.

    At iteration 0 every table is zero, so the hybrid's trunk and the blueprint
    regret-match to the SAME uniform strategy and the only difference left is
    the maximisation on the best-response streets. Later in a run they are two
    different tables and the comparison would not be max-against-mix any more.
    """
    driver.prepare(contexts, 0, boards=boards)
    initial = np.ones(contexts[0].num_hands, dtype=np.float32)
    for kernel in driver.kernels:
        kernel.opponent = None
        kernel.update_players = ()
        kernel.forward(initial)
        kernel.evaluate_terminals()
        kernel.value[:] = 0.0
    driver.hybrid_pass()
    hybrid = [sum(float(k.value[p, 0].sum()) for k in driver.kernels) for p in (0, 1)]

    for kernel in driver.kernels:
        kernel.forward(initial)
        kernel.evaluate_terminals()
        kernel.backward()
    matched = [sum(float(k.value[p, 0].sum()) for k in driver.kernels) for p in (0, 1)]
    return hybrid, matched


class TestTheOpponentIsReallyBestResponding:
    @pytest.mark.parametrize("region", ["river", "turn_river", "postflop", "all"])
    def test_the_hybrid_beats_the_regret_matched_strategy_it_replaced(self, parts, region):
        """A best response is optimal by construction; below that, it is not wired in.

        Both root values are counterfactual and both fold in the SAME opponent
        reach, so the only difference is what the responder itself played.
        """
        compiled, contexts, _ = parts
        driver, _, _, _ = _driver(compiled, region, len(contexts))
        hybrid, matched = _root_values(driver, contexts, BOARDS)
        assert hybrid[0] > matched[0]
        assert hybrid[1] > matched[1]

    @pytest.mark.timeout(60)  # four regions x four boards of real tree passes
    def test_a_wider_best_response_region_is_worth_at_least_as_much(self, parts):
        """Streets nest, so from one starting point the responder's value must too."""
        compiled, contexts, _ = parts
        values = []
        for region in ("river", "turn_river", "postflop", "all"):
            driver, _, _, _ = _driver(compiled, region, len(contexts))
            hybrid, _ = _root_values(driver, contexts, BOARDS)
            values.append(hybrid[0])
        assert values == sorted(values), values


class TestWhatEachTableHolds:
    def test_the_trunk_covers_every_street_the_responder_does_not(self, parts):
        compiled, _, _ = parts
        tree = compiled.tree
        for region in ("river", "turn_river", "postflop", "all"):
            layout = TrunkLayout(tree, BR_REGIONS[region])
            excluded = frozenset(BR_REGIONS[region])
            expected = sum(
                tree.num_buckets(node.street) * node.num_actions
                for node in tree.nodes
                if node.street not in excluded
            )
            assert layout.num_slots == expected
            assert all(
                layout.base[node.node_id] < 0 for node in tree.nodes if node.street in excluded
            )
        assert TrunkLayout(tree, BR_REGIONS["all"]).num_slots == 0

    def test_the_opponent_table_never_touches_the_blueprint(self, parts):
        """The hybrid's trunk regrets are a different object from either seat's answer.

        They minimise regret against the CFR player's sequence, not against a
        hybrid, so sharing rows with the blueprint would be a different
        algorithm -- and a silent one, since both are regret tables.
        """
        compiled, contexts, _ = parts
        driver, regrets, strategy_sum, trunk = _driver(compiled, "river", 1)
        for iteration in range(4):
            driver.iterate(contexts[:1], iteration)

        assert np.count_nonzero(trunk) > 0
        assert np.count_nonzero(regrets) > 0
        assert np.count_nonzero(strategy_sum) > 0
        # Both seats' rows are written: CFR-BR here runs both games at once.
        tree = compiled.tree
        for actor in (True, False):
            nodes = [node.node_id for node in tree.nodes if node.actor_is_button is actor]
            slots = np.concatenate(
                [np.arange(*tree.slots(node, 0), dtype=np.int64) for node in nodes[:200]]
            )
            assert np.count_nonzero(strategy_sum[slots]) > 0


class TestClairvoyanceIsNotPricedIn:
    def test_one_action_is_chosen_across_runouts_a_street_cannot_tell_apart(self, parts):
        """The failure that looks like a triumph: an opponent reading face-down cards.

        Four boards share a flop; two share each turn. A best response standing
        on the flop sees one observation, so its action for a given holding must
        be the same on all four -- and on the turn, the same on the two that
        share it. Maximising per board instead makes the opponent clairvoyant,
        which is a HARDER opponent and so improves nothing this measures.
        """
        compiled, contexts, _ = parts
        driver, _, _, _ = _driver(compiled, "postflop", len(contexts))
        driver.iterate(contexts, 0, boards=BOARDS)

        shared = {Street.FLOP: [0, 1, 2, 3], Street.TURN: [0, 1]}
        checked = 0
        for position, (group, _) in enumerate(driver.plan):
            members = shared.get(group.street)
            if members is None:
                continue
            picks = [driver.agents[index][group.actor].picks[position] for index in members]
            if picks[0] is None:
                continue
            reference = dict(zip(driver._global_hand_id[members[0]], picks[0][0], strict=True))
            for index, chosen in zip(members[1:], picks[1:], strict=True):
                for hand, action in zip(driver._global_hand_id[index], chosen[0], strict=True):
                    if hand in reference:
                        assert reference[hand] == action
            checked += 1
        assert checked > 0


class TestSequentialBinding:
    # Two iterations over four boards, twice: real tree passes, not a fixture.
    @pytest.mark.timeout(120)
    def test_one_rebound_kernel_writes_exactly_what_four_kernels_do(self, parts):
        """Rebinding is what makes ``runouts_per_flop`` affordable; it must be free.

        A kernel's hand-space scratch is ~3 GB at production size, so holding one
        per runout is what a 32 GiB node cannot do. The two modes differ only in
        how many are alive at once, and any drift between them would be a
        different trainer wearing the same knob.
        """
        compiled, contexts, _ = parts
        # Runouts of ONE flop, which is what the sampler draws: they share the
        # flop's prefix, so only a river region is legal here.
        river = [Street.RIVER]
        tables = []
        for sequential in (False, True):
            driver, regrets, strategy_sum, trunk = _driver(compiled, "river", len(contexts))
            driver.sequential = sequential and len(contexts) > 1
            driver.br_streets = frozenset(river)
            for iteration in range(2):
                driver.iterate(contexts, iteration, boards=BOARDS)
            tables.append((regrets.copy(), strategy_sum.copy(), trunk.copy()))

        for parallel, rebound in zip(*tables, strict=True):
            assert np.array_equal(parallel, rebound)

    def test_sequential_refuses_a_region_it_cannot_maximise_across(self, parts):
        """Four runouts share a flop, so a flop best response needs them at once."""
        compiled, contexts, _ = parts
        driver, _, _, _ = _driver(compiled, "postflop", len(contexts))
        driver.sequential = True
        with pytest.raises(ValueError, match="Sequential binding cannot best-respond"):
            driver.iterate(contexts, 0, boards=BOARDS)


@pytest.mark.slow
@pytest.mark.timeout(900)
def test_cfr_br_is_less_exploitable_in_the_real_game_than_cfr(parts):
    """The claim the algorithm exists for, on a game small enough to check exactly.

    Feeding the WHOLE enumerated chance layer as one iteration's boards makes
    the joint maximisation a real best response at every street, so this is
    CFR-BR with no trunk and no sampling approximation. Both arms solve the same
    abstraction for the same number of iterations; ``unconstrained`` is the
    analogue of the programme's ``exact_br`` -- the responder is told its exact
    two cards but not the future.
    """
    compiled, contexts, pairs = parts
    initial = np.ones(contexts[0].num_hands, dtype=np.float32)
    iterations = 400

    plain = BoardMixtureCFR(compiled, contexts, boards=BOARDS, cfr_plus=True)
    for _ in range(iterations):
        plain.iterate(initial)

    driver, _, strategy_sum, _ = _driver(compiled, "all", len(contexts))
    for iteration in range(iterations):
        driver.iterate(contexts, iteration, boards=BOARDS)

    scorer = BoardMixtureCFR(compiled, contexts, boards=BOARDS)
    scorer.strategy_sum[:] = plain.strategy_sum
    cfr_real = scorer.exploitability(initial, pairs, unconstrained=True)
    cfr_abstract = scorer.exploitability(initial, pairs, unconstrained=False)
    scorer.strategy_sum[:] = strategy_sum
    br_real = scorer.exploitability(initial, pairs, unconstrained=True)
    br_abstract = scorer.exploitability(initial, pairs, unconstrained=False)

    assert br_real < cfr_real, f"CFR-BR {br_real} did not beat CFR {cfr_real} in the real game"
    # The trade that identifies the algorithm: CFR-BR is NOT solving the
    # abstract game and must be worse there, or it is just CFR.
    assert br_abstract > cfr_abstract
