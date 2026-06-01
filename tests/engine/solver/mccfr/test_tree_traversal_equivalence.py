"""Does walking node ids compute exactly what walking game states computed?

``tree_traversal`` drops the ``GameState`` from the hot path and reads the
betting line off a table recorded at enumeration time instead. That is a
rewrite of the traversal, and the failure mode it risks is not a crash: a
mis-tabulated terminal payoff or a chance deal moved by one call site produces
a solver that still converges, to a slightly different thing, on a trajectory
nobody can reproduce.

So the bar here is bit-identity, not agreement to a tolerance. Both traversals
run from the same seed into their own storage, and every shared array must come
out byte-for-byte equal — regrets and strategy sums in float32, where a
reordered addition would show. Both random streams are then read one more time,
because identical arrays would not by themselves prove that the two made the
same draws in the same order: a run's reproducibility depends on that, and it
is what makes the fast path a speedup rather than a new lineage.
"""

from __future__ import annotations

import random

import numpy as np
import pytest

from src.core.actions.action_model import ActionModel
from src.core.game.rules import GameRules
from src.core.game.state import Card, GameState, Street
from src.engine.solver.betting_tree import build_betting_tree
from src.engine.solver.mccfr import chance, tree_traversal
from src.engine.solver.mccfr.static_solver import StaticTreeSolver
from src.engine.solver.storage.static_array import StaticArrayStorage
from src.shared.config.loader import load_training_config
from tests.test_helpers import make_test_config

ARRAYS = ("regrets", "strategy_sum", "reach_counts", "cumulative_utility", "visited")


class BucketsByStreet:
    """Deterministic, cheap stand-in for the equity abstraction."""

    def __init__(self, counts):
        self.counts = counts

    def get_bucket(self, hole_cards, board, street):
        total = hole_cards[0].rank_eval7() + hole_cards[1].rank_eval7() + board[0].rank_eval7()
        return total % self.counts[street]

    def num_buckets(self, street):
        return self.counts[street]


def build(config, counts=None):
    """Tree and abstraction for ``config``; both arms share them.

    Bucket counts change only how many ROWS a node owns, never the edge
    structure, so a production action model can be exercised against a handful
    of buckets and still walk exactly the edges production walks.
    """
    abstraction = BucketsByStreet(counts or {Street.FLOP: 4, Street.TURN: 3, Street.RIVER: 2})
    action_model = ActionModel(config)
    tree = build_betting_tree(
        GameRules(config.game.small_blind, config.game.big_blind),
        action_model,
        abstraction,
        starting_stack=config.game.starting_stack,
    )
    return action_model, abstraction, tree


def run(config, built, *, walk_tree: bool, iterations: int, seed: int):
    """Train one solver and return its arrays plus the next draw from each stream."""
    action_model, abstraction, tree = built
    storage = StaticArrayStorage(tree)
    solver = StaticTreeSolver(action_model, abstraction, storage, config, tree=tree)
    solver._walk_tree = walk_tree

    random.seed(seed)
    np.random.seed(seed)
    utilities = []
    for iteration in range(iterations):
        solver.iteration = iteration
        utilities.append(solver.train_iteration())

    arrays = {name: getattr(storage, name).copy() for name in ARRAYS}
    return arrays, utilities, solver.applied_updates, (random.random(), np.random.random())


@pytest.mark.parametrize(
    ("weighting", "cfr_plus"),
    [("none", False), ("linear", False), ("dcfr", False), ("dcfr", True)],
)
def test_tree_walk_is_bit_identical_to_state_walk(weighting, cfr_plus):
    config = make_test_config(
        seed=42,
        starting_stack=200,
        iteration_weighting=weighting,
        cfr_plus=cfr_plus,
    )

    built = build(config)
    fast, fast_utilities, fast_applied, fast_streams = run(
        config, built, walk_tree=True, iterations=60, seed=1234
    )
    reference, ref_utilities, ref_applied, ref_streams = run(
        config, built, walk_tree=False, iterations=60, seed=1234
    )

    for name in ARRAYS:
        assert np.array_equal(fast[name], reference[name]), f"{name} diverged"
    assert fast_utilities == ref_utilities
    assert fast_applied == ref_applied
    # Same number of draws, in the same order, from both streams.
    assert fast_streams == ref_streams


def test_the_fast_path_actually_trained_something():
    """Guards the comparison above from passing on two empty tables."""
    config = make_test_config(seed=42, starting_stack=200, iteration_weighting="dcfr")
    fast, _, applied, _ = run(config, build(config), walk_tree=True, iterations=60, seed=1234)

    assert applied > 0
    assert int(fast["visited"].sum()) > 0
    assert np.count_nonzero(fast["regrets"]) > 0
    assert np.count_nonzero(fast["strategy_sum"]) > 0


def test_the_compared_run_reaches_every_kind_of_ending(monkeypatch):
    """Bit-identity is only as strong as the paths the comparison walks.

    The three terminals are tabulated differently and fail differently: a fold
    pays a constant, a river showdown owes no cards, and an all-in before the
    river owes a runout — which is also the only terminal that draws from the
    random stream. A comparison that never hit the third would say nothing
    about the half of the table most likely to be wrong. Streets are checked
    too, since reaching the river means both the 3-card and the 1-card chance
    deals fired at the right edges.
    """
    seen: set[tuple[bool, int]] = set()
    real = tree_traversal._terminal_value

    def spy(walk, terminal, board, known):
        seen.add((terminal.is_fold, terminal.cards_to_deal))
        return real(walk, terminal, board, known)

    monkeypatch.setattr(tree_traversal, "_terminal_value", spy)

    config = make_test_config(seed=42, starting_stack=200, iteration_weighting="dcfr")
    run(config, build(config), walk_tree=True, iterations=60, seed=1234)

    assert (True, 0) in seen, "no fold terminal"
    assert (False, 0) in seen, "no showdown with a complete board"
    assert any(not is_fold and owed > 0 for is_fold, owed in seen), (
        "no all-in showdown that owed a runout — the terminal path that draws cards"
    )


def test_the_compared_run_reaches_every_street():
    config = make_test_config(seed=42, starting_stack=200, iteration_weighting="dcfr")
    action_model, abstraction, tree = build(config)
    storage = StaticArrayStorage(tree)
    solver = StaticTreeSolver(action_model, abstraction, storage, config, tree=tree)

    random.seed(1234)
    np.random.seed(1234)
    for iteration in range(60):
        solver.iteration = iteration
        solver.train_iteration()

    reached = {
        node.street
        for node in tree.nodes
        if storage.visited[
            tree.row(node.node_id, 0) : tree.row(node.node_id, 0) + tree.num_buckets(node.street)
        ].any()
    }
    assert reached == {Street.PREFLOP, Street.FLOP, Street.TURN, Street.RIVER}


@pytest.mark.timeout(30)
def test_bit_identical_under_the_production_action_model(monkeypatch):
    """The comparison above walks a toy action model. Production's is not one.

    ``config/training/production.yaml`` adds the preflop templates, `min_raise`
    / `pot_raise` / `jam`, `after_two_raises: [jam]` and five raises a street.
    Those produce edge shapes the default model never reaches — a sizing that
    resolves to the whole stack, and an all-in for less than the call — and
    those are exactly where a mis-tabulated ``TerminalOutcome`` would pay the
    wrong constant while the run still converged to something plausible.

    Bucket counts are cut to two a street because they do not change a single
    edge: they set how many rows a node owns, not where an action leads. So
    this walks the production betting tree against a table small enough to
    compare in a unit test.
    """
    config = load_training_config("production")
    counts = {Street.FLOP: 2, Street.TURN: 2, Street.RIVER: 2}
    built = build(config, counts)

    endings: set[tuple[bool, int]] = set()
    real = tree_traversal._terminal_value

    def spy(walk, terminal, board, known):
        endings.add((terminal.is_fold, terminal.cards_to_deal))
        return real(walk, terminal, board, known)

    with monkeypatch.context() as patched:
        patched.setattr(tree_traversal, "_terminal_value", spy)
        fast, fast_utilities, fast_applied, fast_streams = run(
            config, built, walk_tree=True, iterations=200, seed=99
        )

    reference, ref_utilities, ref_applied, ref_streams = run(
        config, built, walk_tree=False, iterations=200, seed=99
    )

    for name in ARRAYS:
        assert np.array_equal(fast[name], reference[name]), f"{name} diverged"
    assert fast_utilities == ref_utilities
    assert fast_applied == ref_applied
    assert fast_streams == ref_streams

    assert (True, 0) in endings, "no fold terminal"
    assert (False, 0) in endings, "no showdown with a complete board"
    assert any(not is_fold and owed > 0 for is_fold, owed in endings), (
        "no all-in showdown that owed a runout"
    )


@pytest.mark.timeout(30)
def test_bit_identical_at_a_production_iteration_count():
    """Same comparison, but at the *t* a real run spends its life at.

    Every check above starts at t=0, where the DCFR discount ``t^a/(t^a+1)`` is
    near a half and the strategy weight ``t^gamma`` is near one. Production
    runs at t in the tens of millions, where those are ~1.0 and ~1e14 — a
    different floating-point regime entirely, and the one where a hoisted
    weight (this path computes ``t^gamma`` once an iteration instead of once a
    node) would show up as drift if it were not exactly the same value.
    """
    config = make_test_config(seed=42, starting_stack=200, iteration_weighting="dcfr")
    built = build(config)
    start = 30_000_000

    def at(walk_tree: bool):
        action_model, abstraction, tree = built
        storage = StaticArrayStorage(tree)
        solver = StaticTreeSolver(action_model, abstraction, storage, config, tree=tree)
        solver._walk_tree = walk_tree
        random.seed(7)
        np.random.seed(7)
        for iteration in range(start, start + 60):
            solver.iteration = iteration
            solver.train_iteration()
        return {name: getattr(storage, name).copy() for name in ARRAYS}

    fast, reference = at(True), at(False)
    for name in ARRAYS:
        assert np.array_equal(fast[name], reference[name]), f"{name} diverged at t={start:,}"
    assert np.count_nonzero(fast["strategy_sum"]) > 0, "nothing accumulated; the check is vacuous"


@pytest.mark.timeout(30)
def test_a_checkpoint_written_by_the_old_traversal_resumes_onto_the_new_one(tmp_path):
    """The deployment question, not the greenfield one.

    Runs on the share are trained in chunks: a task loads the last checkpoint,
    advances the absolute iteration target, and publishes. When this change
    lands, live runs will be mid-flight — every one of them resumes a table
    that the state-based traversal wrote onto a traversal that walks node ids.

    Nothing about the checkpoint changed (same layout, same fingerprint, so it
    loads at all), but "loads" is the weak claim. The one worth pinning is that
    the run CONTINUES ON THE SAME TRAJECTORY: an arm that resumes onto the new
    traversal must end byte-identical to an arm that resumes onto the old one.
    If it did not, every published run would be split into two lineages at
    whichever chunk boundary happened to straddle the deploy — and the ledger
    would go on comparing them as one.

    Both arms take the same checkpoint round trip, so any RNG the IO path
    touches perturbs them equally.
    """
    config = make_test_config(seed=42, starting_stack=200, iteration_weighting="dcfr")
    built = build(config)
    action_model, abstraction, tree = built

    def solver_for(walk_tree: bool, storage):
        made = StaticTreeSolver(
            action_model,
            abstraction,
            storage,
            config,
            tree=tree,
            checkpoint_dir=tmp_path / "run",
        )
        made._walk_tree = walk_tree
        return made

    # Phase 1: 80 iterations under the traversal that wrote every table now on
    # the share, then a checkpoint — the artifact a live run would be holding.
    written = StaticArrayStorage(tree)
    writer = solver_for(False, written)
    random.seed(555)
    np.random.seed(555)
    for iteration in range(80):
        writer.iteration = iteration
        writer.train_iteration()
    writer.checkpoint()

    # Phase 2: two resumes off that one checkpoint, differing only in traversal.
    def resume(walk_tree: bool):
        storage = StaticArrayStorage(tree)
        resumed = solver_for(walk_tree, storage)
        restored = resumed.restore()
        assert restored == 80, "the checkpoint did not carry the absolute iteration"
        random.seed(777)
        np.random.seed(777)
        for iteration in range(restored, restored + 80):
            resumed.iteration = iteration
            resumed.train_iteration()
        return {name: getattr(storage, name).copy() for name in ARRAYS}

    onto_new, onto_old = resume(True), resume(False)

    for name in ARRAYS:
        assert np.array_equal(onto_new[name], onto_old[name]), (
            f"{name} diverged after resume — a run straddling this deploy would "
            "become two lineages at the chunk boundary"
        )
    # The resume must also have MOVED the table, or the comparison is empty.
    assert not np.array_equal(onto_new["regrets"], written.regrets)


@pytest.mark.parametrize("count", [1, 3, 5])
def test_the_card_draw_matches_the_one_the_state_traversal_uses(count):
    """The runout is drawn by a second implementation now. Pin it to the first.

    ``chance.draw_cards`` rebuilds the seen-card mask from the state on every
    call; ``tree_traversal._draw`` carries it down the recursion instead. Same
    mask, so same rejections and the same cards — but only if the draws also
    come off ``random.randrange`` in the same order, which is what makes a run
    reproducible across this change.
    """
    hole = ((Card.new("As"), Card.new("Kd")), (Card.new("Qh"), Card.new("Jc")))
    board = (Card.new("2c"), Card.new("7d"), Card.new("9h"))
    state = GameState(
        street=Street.FLOP,
        pot=10,
        stacks=(95, 95),
        board=board,
        hole_cards=hole,
        betting_history=(),
        button_position=0,
        current_player=1,
        to_call=0,
        is_terminal=False,
    )

    known = 0
    for player_cards in hole:
        for card in player_cards:
            known |= card.mask
    for card in board:
        known |= card.mask

    random.seed(2024)
    expected = chance.draw_cards(state, count)
    random.seed(2024)
    drawn, widened = tree_traversal._draw(known, count)

    assert list(drawn) == expected
    assert widened == known | sum(card.mask for card in drawn)
    # Drawing again must continue the stream, not repeat it.
    assert tree_traversal._draw(widened, 1)[0][0] not in drawn


def test_an_out_of_range_bucket_raises_instead_of_aliasing():
    """The fast path indexes rows directly, so it owns this check now.

    Rows are contiguous per node: an out-of-range bucket does not fall off the
    array, it lands on a different node's infoset and the two silently share
    storage. ``StaticArrayStorage.view`` used to be the only place that could
    catch it, and the tree walk no longer goes through ``view``.
    """
    config = make_test_config(seed=42, starting_stack=200, iteration_weighting="dcfr")
    action_model, abstraction, tree = build(config)
    solver = StaticTreeSolver(
        action_model, abstraction, StaticArrayStorage(tree), config, tree=tree
    )

    class OutOfRange:
        def get_bucket(self, hole_cards, board, street):
            return 10_000

        def num_buckets(self, street):
            return abstraction.num_buckets(street)

    solver.card_abstraction = OutOfRange()
    random.seed(1)
    np.random.seed(1)

    def train_until_it_lands_on_a_postflop_node():
        for iteration in range(200):
            solver.iteration = iteration
            solver.train_iteration()

    with pytest.raises(IndexError, match="out of range"):
        train_until_it_lands_on_a_postflop_node()


@pytest.mark.parametrize(("traversal", "walks_tree"), [("tree", True), ("state", False)])
def test_the_traversal_config_selects_the_path(traversal, walks_tree):
    """The A/B switch has to actually switch, or a benchmark measures one arm twice.

    This is the seam the cloud comparison drives through ``--set
    solver__traversal=state``; a default that quietly won over the override
    would produce two identical arms and a 1.0x speedup nobody could explain.
    """
    config = make_test_config(seed=42, starting_stack=200, **{"solver.traversal": traversal})
    action_model, abstraction, tree = build(config)
    solver = StaticTreeSolver(
        action_model, abstraction, StaticArrayStorage(tree), config, tree=tree
    )

    assert config.solver.traversal == traversal
    assert solver._walk_tree is walks_tree
