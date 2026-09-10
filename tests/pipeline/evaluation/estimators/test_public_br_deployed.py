"""`exact_br` against the DEPLOYED system (blueprint + resolver).

The estimator's value is that it is a best response to a FIXED strategy and
that two runs agree bit for bit. Both are easy to lose here, so both are pinned.

Every walk here runs a resolver, which makes these the slowest tests in the fast
gate: MEASURED serially on an idle machine, 39.4s, 31.1s and 15s twice. Against
the 60s default that leaves the slowest 1.5x of headroom, and `-n auto` then
runs it beside eleven competing workers -- so the timeout fired on whichever of
them lost the CPU race, a different one each time. The explicit timeouts below
are that measurement plus room for a loaded machine, NOT an assertion about
speed.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

from src.pipeline.evaluation.estimators.public_tree_br import (
    PublicBRConfig,
    PublicTreeBestResponse,
)
from tests.test_helpers import build_trained_test_solver


@pytest.fixture(scope="module")
def solver():
    return build_trained_test_solver(200)


def _tier(*, deployed: bool) -> PublicBRConfig:
    """The smallest tier that still walks a whole hand, so the resolver runs."""
    return PublicBRConfig(
        num_flops=1, num_turns=1, num_rivers=1, resolver_iterations=8, deployed=deployed
    )


def _score(solver, *, deployed: bool) -> float:
    walker = PublicTreeBestResponse(solver, _tier(deployed=deployed), starting_stack=400)
    return walker.evaluate().exploitability_mbb


@pytest.mark.timeout(300)
class TestTheResolverIsActuallyMeasured:
    def test_deployed_differs_from_the_blueprint(self, solver):
        """A resolver that never ran would return the blueprint's own number."""
        assert _score(solver, deployed=True) != _score(solver, deployed=False)


@pytest.mark.timeout(300)
class TestTheNumberIsReproducible:
    """Zero evaluation variance is what makes two checkpoints exactly paired.

    The resolver samples leaf rollouts, so it needs a seed derived from the
    public state. Deriving it from `hash()` passes WITHIN one process and fails
    across processes, because Python randomizes string hashing per process --
    measured as the same tier scoring 579.600561 and 574.904912 in two
    interpreters. Hence the subprocess.
    """

    def test_identical_in_a_second_interpreter(self, solver):
        here = _score(solver, deployed=True)
        script = textwrap.dedent("""
            from src.pipeline.evaluation.estimators.public_tree_br import (
                PublicBRConfig, PublicTreeBestResponse)
            from tests.test_helpers import build_trained_test_solver
            config = PublicBRConfig(
                num_flops=1, num_turns=1, num_rivers=1,
                resolver_iterations=8, deployed=True)
            walker = PublicTreeBestResponse(
                build_trained_test_solver(200), config, starting_stack=400)
            print(repr(walker.evaluate().exploitability_mbb))
        """)
        out = subprocess.run(
            [sys.executable, "-c", script], capture_output=True, text=True, check=True
        )
        assert float(out.stdout.strip().splitlines()[-1]) == here


class TestTheLedgerCanTellTheArmsApart:
    """blueprint+resolver is a different STRATEGY, not a knob on the same one.

    Measured 08-31 before this entered the tier: a deployed row scoring 3076.4
    and a blueprint row scoring 2409.8 recorded byte-identical knobs, so the
    ledger paired them and the 667 mbb/hand difference read as noise on one arm.
    """

    def test_deployed_does_not_share_a_tier_with_the_blueprint(self):
        from src.pipeline.evaluation.ledger.tiers import build_exact_br_knobs_from_params

        blueprint = build_exact_br_knobs_from_params(
            num_flops=1, num_turns=1, num_rivers=1, board_seed=7
        )
        deployed = build_exact_br_knobs_from_params(
            num_flops=1,
            num_turns=1,
            num_rivers=1,
            board_seed=7,
            deployed=True,
            resolver_iterations=200,
            resolver_prior_weight=50.0,
        )
        assert blueprint != deployed
        assert deployed["deployed"] is True
        # Two deployed arms that solved differently are also different tiers.
        other = build_exact_br_knobs_from_params(
            num_flops=1,
            num_turns=1,
            num_rivers=1,
            board_seed=7,
            deployed=True,
            resolver_iterations=40,
            resolver_prior_weight=50.0,
        )
        assert other != deployed


@pytest.mark.timeout(300)
class TestTheResolverCacheIsBounded:
    """Keyed by (context, BOARD), the resolver cache grows with the walk instead
    of saturating like the blueprint's per-context table -- and the fork-join's
    RAM guard cannot see it, so it hands out workers sized for the blueprint's
    caches alone. Unbounded, deployed arms were killed after 4-6 h on a node
    while short ones passed, which reads as a flaky pool rather than a leak.

    Eviction is safe because the row is a deterministic function of its key, so
    the bound costs recomputes and never the answer -- asserted here, since a
    cache that changed the number would be a correctness bug, not a tuning one.
    """

    def test_the_cache_never_exceeds_its_bound(self, solver, monkeypatch):
        import src.pipeline.evaluation.estimators.public_tree_br as module

        monkeypatch.setattr(module, "_RESOLVER_CACHE_ENTRIES", 4)
        walker = module.PublicTreeBestResponse(solver, _tier(deployed=True), starting_stack=400)
        walker.evaluate()
        assert len(walker._resolver_cache) <= 5

    def test_evicting_does_not_move_the_number(self, solver, monkeypatch):
        import src.pipeline.evaluation.estimators.public_tree_br as module

        roomy = _score(solver, deployed=True)
        monkeypatch.setattr(module, "_RESOLVER_CACHE_ENTRIES", 1)
        cramped = _score(solver, deployed=True)
        assert roomy == cramped
