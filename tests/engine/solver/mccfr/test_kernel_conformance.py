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

**These tests found a real defect on the day they were written, 2026-07-25**, and
it is now fixed. Under external sampling the regret update was scaled by
``reach_probs[opponent]`` -- the product of the opponent's *sampled action*
probabilities (instrumented: chance never multiplied into it). Because those
actions were sampled, the traversal already visits an infoset with probability
pi_{-i}, carrying that same action reach; applying it again squared the action
component and the regrets minimised a reweighted objective whose fixed point is
not the equilibrium. Same double-counting ``docs/AVERAGE_STRATEGY_WEIGHTING.md``
diagnosed for the average-strategy accumulator, still live on the regret one.

It was a floor, not slowness -- Leduc, seed 42, identical but for that multiplier:

===========  ========  ========  ========  ========
iterations      100k      250k      500k        1M
===========  ========  ========  ========  ========
before        0.16402   0.11945   0.13381   0.11580
after         0.10823   0.06976   0.04701   0.03348
===========  ========  ========  ========  ========

After decays 3.23x over a 10x iteration budget against the 3.16x that O(1/sqrt(T))
predicts; before decayed 1.42x, non-monotonically, and the gap between the two
*widened* (1.52x -> 3.46x), which a uniformly slower method would not do. On Kuhn
the fix moved exploitability at 200k from 1.3e-2 to 2.9e-3.

The convergence assertions below are the permanent guard: reintroducing the
multiplier puts Kuhn back at ~1.3e-2 and fails
``test_average_strategy_converges_to_equilibrium`` outright.
"""

from __future__ import annotations

import pytest

from src.core.game.actions import bet, call, check, fold, raises
from src.pipeline.evaluation.best_response import exploitability, on_policy_value
from tests.engine.solver.mccfr.dict_storage import DictStorage
from tests.engine.solver.mccfr.extensive_game_solver import (
    ExtensiveGameSolver,
    average_policy,
    current_policy,
)
from tests.pipeline.evaluation.kuhn_poker import KuhnPoker
from tests.pipeline.evaluation.leduc_poker import LeducPoker
from tests.test_helpers import make_test_config

KUHN_GAME_VALUE_P0 = -1.0 / 18.0

# Core Action objects standing in for each game's own action labels. Their type
# and amount are arbitrary tags -- nothing in the kernel reads them.
KUHN_ACTIONS = {"p": check(), "b": bet(1)}
LEDUC_ACTIONS = {"f": fold(), "c": call(), "r": raises(1)}

KUHN_ITERATIONS = 200_000
LEDUC_ITERATIONS = 100_000

# Measured 0.108 at LEDUC_ITERATIONS post-fix, against 0.164 before it.
LEDUC_EXPLOITABILITY_BOUND = 0.15


def _train(game, actions, iterations: int, session: str, **config_overrides):
    config = make_test_config(seed=42, **config_overrides)
    storage = DictStorage()
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

    def test_average_strategy_converges_to_equilibrium(self, kuhn_game, kuhn_solver):
        """The binding assertion, and the regression guard for the 07-25 fix.

        Measures 2.9e-3 at this budget. Reintroducing the opponent-reach
        multiplier the fix removed puts it back at ~1.3e-2 and fails here.
        """
        assert exploitability(kuhn_game, average_policy(kuhn_solver)) < 5e-3


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
                average((card, history), ("p", "b")),
                current((card, history), ("p", "b")),
                strict=True,
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
        # Level, not just direction: the pre-fix kernel sat at 0.164 here, so this
        # bound is the deeper-tree half of the regression guard.
        assert later < LEDUC_EXPLOITABILITY_BOUND


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
