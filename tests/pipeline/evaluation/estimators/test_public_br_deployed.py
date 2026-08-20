"""`exact_br` against the DEPLOYED system (blueprint + resolver).

The estimator's value is that it is a best response to a FIXED strategy and
that two runs agree bit for bit. Both are easy to lose here, so both are pinned.
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


class TestTheResolverIsActuallyMeasured:
    def test_deployed_differs_from_the_blueprint(self, solver):
        """A resolver that never ran would return the blueprint's own number."""
        assert _score(solver, deployed=True) != _score(solver, deployed=False)


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
