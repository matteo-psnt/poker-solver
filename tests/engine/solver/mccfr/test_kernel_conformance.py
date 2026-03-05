"""Does the production CFR kernel converge in games whose answer we know?

Everything else that scores this solver measures HUNL, where the equilibrium is
unknown and every number is an estimate with a confidence interval. Here the
answer is known exactly, so a deviation is a deviation and not a noise band.

The solver under test is the shipped ``MCCFRSolver`` with its HUNL state machine
swapped out (see ``extensive_game_solver``); the regret math, the update
placement, the averaging and the sampling are the production ones. The scoring is
done by ``best_response.exploitability``, itself already anchored to hand-computed
Kuhn values in ``tests/pipeline/evaluation/test_best_response.py``. So a failure
here implicates the kernel and little else.

**Finding, 2026-07-25.** It does not converge. Under external sampling the regret
update is scaled by ``reach_probs[opponent]`` in ``traversal.cfr_external_sampling``
-- the product of the opponent's *sampled action* probabilities along the path
(instrumented: chance never multiplies into it). But the opponent's actions were
sampled, so the traversal already visits an infoset with probability pi_{-i},
carrying that same action reach along with the chance reach. Applying it a second
time squares the action component, and the regrets then minimise a reweighted
objective whose fixed point is not the equilibrium. It is the same double-counting
``docs/AVERAGE_STRATEGY_WEIGHTING.md`` diagnosed for the average-strategy
accumulator, still live on the regret accumulator.

Evidence that this is a floor rather than slow convergence -- Leduc, seed 42,
identical apart from that one multiplier:

===========  ========  ========  ========  ========
iterations      100k      250k      500k        1M
===========  ========  ========  ========  ========
current       0.16402   0.11945   0.13381   0.11580
multiplier    0.10823   0.06976   0.04701   0.03348
 dropped
===========  ========  ========  ========  ========

Dropping it decays 3.23x over a 10x iteration budget, against the 3.16x that
O(1/sqrt(T)) predicts. Keeping it decays 1.42x, non-monotonically, and the gap
between the two *widens* (1.52x -> 3.46x) -- a uniformly slower method would hold
a constant ratio. Kuhn agrees but is a weak witness on its own: it is shallow
enough that ~97% of regret updates carry a multiplier of exactly 1.0.

``test_dropping_the_opponent_reach_multiplier_restores_convergence`` pins the
mechanism, and the convergence assertions are ``xfail(strict=True)`` so they
announce themselves the moment the kernel is fixed.
"""

from __future__ import annotations

import pytest

from src.core.game.actions import bet, call, check, fold, raises
from src.engine.solver.mccfr import traversal
from src.pipeline.evaluation.best_response import exploitability, on_policy_value
from tests.engine.solver.mccfr.extensive_game_solver import (
    ExtensiveGameSolver,
    average_policy,
    current_policy,
)
from tests.pipeline.evaluation.kuhn_poker import KuhnPoker
from tests.pipeline.evaluation.leduc_poker import LeducPoker
from tests.test_helpers import build_test_storage, make_test_config

KUHN_GAME_VALUE_P0 = -1.0 / 18.0

# Core Action objects standing in for each game's own action labels. Their type
# and amount are arbitrary tags -- nothing in the kernel reads them.
KUHN_ACTIONS = {"p": check(), "b": bet(1)}
LEDUC_ACTIONS = {"f": fold(), "c": call(), "r": raises(1)}

KUHN_ITERATIONS = 200_000
LEDUC_ITERATIONS = 100_000

# The plateau the kernel currently settles on, with headroom. Not a target -- a
# tripwire, so a change that makes convergence *worse* cannot pass unnoticed
# while the tight assertions below sit in xfail.
KUHN_CURRENT_PLATEAU = 0.03


def _train(game, actions, iterations: int, session: str, **config_overrides):
    config = make_test_config(seed=42, **config_overrides)
    storage = build_test_storage(session, initial_capacity=20_000)
    solver = ExtensiveGameSolver(game, actions, storage, config)
    for _ in range(iterations):
        solver.train_iteration()
    return solver


@pytest.fixture(scope="module")
def kuhn_game() -> KuhnPoker:
    return KuhnPoker()


@pytest.fixture(scope="module")
def kuhn_solver(kuhn_game) -> ExtensiveGameSolver:
    return _train(kuhn_game, KUHN_ACTIONS, KUHN_ITERATIONS, "conformance-kuhn")


@pytest.mark.slow
@pytest.mark.timeout(90)
class TestKuhnGroundTruth:
    def test_average_strategy_attains_the_known_game_value(self, kuhn_game, kuhn_solver):
        """Kuhn's value to the first player is exactly -1/18 at equilibrium.

        Passing this while exploitability stalls is consistent, not contradictory:
        Kuhn's equilibria are a one-parameter family all worth -1/18, and a nearby
        strategy still prices out close to it. The value confirms the game is
        wired up correctly; only exploitability sees the deviation.
        """
        value = on_policy_value(kuhn_game, 0, average_policy(kuhn_solver))
        assert value == pytest.approx(KUHN_GAME_VALUE_P0, abs=0.01)

    def test_the_solver_discovers_exactly_kuhn_s_information_sets(self, kuhn_solver):
        """12 = 3 cards x (2 own decision points) x 2 players."""
        assert kuhn_solver.num_infosets() == 12

    def test_exploitability_stays_within_the_known_plateau(self, kuhn_game, kuhn_solver):
        assert exploitability(kuhn_game, average_policy(kuhn_solver)) < KUHN_CURRENT_PLATEAU

    @pytest.mark.xfail(
        strict=True,
        reason=(
            "Production kernel does not converge: external sampling scales the regret "
            "update by the opponent's reach, which the sampled visit frequency already "
            "supplies. See the module docstring and the mechanism test below."
        ),
    )
    def test_average_strategy_converges_to_equilibrium(self, kuhn_game, kuhn_solver):
        assert exploitability(kuhn_game, average_policy(kuhn_solver)) < 5e-3


@pytest.mark.slow
@pytest.mark.timeout(90)
class TestRegretWeightingMechanism:
    def test_dropping_the_opponent_reach_multiplier_restores_convergence(
        self, kuhn_game, monkeypatch
    ):
        """Neutralising just that one factor recovers O(1/sqrt(T)).

        This is what separates "the kernel deviates" from "the adapter is wrong":
        an adapter bug would not be repaired by changing a regret weight, and
        every other part of the setup is held fixed, seed included.
        """
        original = traversal.apply_regret_updates

        def without_opponent_reach(
            regrets, target_indices, utilities, node_utility, opponent_reach, *args
        ):
            return original(regrets, target_indices, utilities, node_utility, 1.0, *args)

        monkeypatch.setattr(traversal, "apply_regret_updates", without_opponent_reach)
        solver = _train(kuhn_game, KUHN_ACTIONS, KUHN_ITERATIONS, "conformance-kuhn-unweighted")

        assert exploitability(kuhn_game, average_policy(solver)) < 5e-3


@pytest.mark.slow
@pytest.mark.timeout(90)
class TestStrategyBridge:
    def test_the_bridge_reads_the_average_not_the_current_iterate(self, kuhn_game, kuhn_solver):
        """The guard that matters most for this whole test module.

        CFR's guarantee is about the average strategy; the current iterate has
        none, and regret matching drives it to a near-pure strategy. A bridge
        that silently returned the current iterate could still pass a loose
        convergence bound by luck, so the two are pinned apart structurally.

        The separation shows up where Kuhn says it should: holding the jack after
        a check, the average bluffs at a genuinely mixed frequency while the
        current iterate is pure.
        """
        average, current = average_policy(kuhn_solver), current_policy(kuhn_solver)
        divergence = max(
            abs(a - c)
            for card in range(3)
            for history in ("", "p", "b", "pb")
            for a, c in zip(
                average((card, history), ("p", "b")), current((card, history), ("p", "b"))
            )
        )

        assert divergence > 0.2
        assert exploitability(kuhn_game, average) < exploitability(kuhn_game, current)

    def test_an_untrained_solver_plays_uniformly_and_is_exploitable(self, kuhn_game):
        untrained = _train(kuhn_game, KUHN_ACTIONS, 0, "conformance-kuhn-untrained")
        policy = average_policy(untrained)

        assert list(policy((0, ""), ("p", "b"))) == [0.5, 0.5]
        assert exploitability(kuhn_game, policy) > 0.05


@pytest.mark.slow
@pytest.mark.timeout(180)
class TestLeduc:
    """A second, larger game with a mid-tree chance node and two betting rounds.

    Kuhn alone cannot distinguish a kernel that mishandles a deeper tree, and
    Leduc's public card exercises reach propagation through a non-root chance
    node -- the place an adapter bug would most plausibly hide. Kuhn passing its
    ground-truth checks first is what makes a Leduc reading interpretable.
    """

    def test_exploitability_falls_as_training_continues(self):
        game = LeducPoker()
        solver = _train(game, LEDUC_ACTIONS, 20_000, "conformance-leduc")
        early = exploitability(game, average_policy(solver))

        for _ in range(LEDUC_ITERATIONS - 20_000):
            solver.train_iteration()
        later = exploitability(game, average_policy(solver))

        assert later < early

    def test_the_opponent_reach_multiplier_costs_accuracy_here_too(self, monkeypatch):
        """The same comparison as the Kuhn mechanism test, in a deeper tree.

        Leduc is the load-bearing witness for the finding: it has a mid-tree
        chance node and two betting rounds, so most regret updates carry a
        genuinely sub-1.0 multiplier, where Kuhn's shallow tree leaves ~97% of
        them at exactly 1.0. Asserted at a modest budget; the module docstring
        records the full 1M curve that distinguishes a floor from slowness.
        """
        game = LeducPoker()
        current = exploitability(
            game, average_policy(_train(game, LEDUC_ACTIONS, LEDUC_ITERATIONS, "leduc-current"))
        )

        original = traversal.apply_regret_updates
        monkeypatch.setattr(
            traversal,
            "apply_regret_updates",
            lambda regrets, indices, utilities, node_utility, opponent_reach, *args: original(
                regrets, indices, utilities, node_utility, 1.0, *args
            ),
        )
        dropped = exploitability(
            game, average_policy(_train(game, LEDUC_ACTIONS, LEDUC_ITERATIONS, "leduc-dropped"))
        )

        assert dropped < 0.8 * current


@pytest.mark.slow
@pytest.mark.timeout(120)
class TestOutcomeSampling:
    """The non-production sampler, recorded rather than trusted.

    Outcome sampling was independently found non-convergent on Kuhn in July; that
    this harness reproduces the same verdict from the *production* code path is
    the best available independent check that the harness itself is sound. Its
    importance weight (``reach_probs[opponent] / strategy_prob``) is the
    legitimate 1/q correction and is a separate question from the external-sampling
    finding above.
    """

    def test_outcome_sampling_does_not_converge_either(self, kuhn_game):
        solver = _train(
            kuhn_game, KUHN_ACTIONS, 50_000, "conformance-kuhn-os", sampling_method="outcome"
        )
        assert exploitability(kuhn_game, average_policy(solver)) > 0.1
