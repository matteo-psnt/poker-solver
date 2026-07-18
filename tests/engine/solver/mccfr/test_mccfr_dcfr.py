"""Does the ``iteration_weighting`` knob reach the regret math?

The exact discount factors are pinned in ``test_model.py`` and
``test_numba_ops.py``; what these add is that the SOLVER honours the config it
was handed. Three tests here once asserted that Pydantic stores the string it
was given and that at least one infoset had been touched -- both pass under
every weighting, so nothing observed DCFR at all.
"""

import numpy as np
import pytest

from tests.test_helpers import DummyCardAbstraction, build_test_solver, make_test_config


def _train(weighting: str, iterations: int = 60) -> np.ndarray:
    """Regrets after training one fixed tree under ``weighting``, from one seed."""
    config = make_test_config(
        seed=42,
        iteration_weighting=weighting,
        dcfr_alpha=1.5,
        dcfr_beta=0.0,
        dcfr_gamma=2.0,
    )
    solver, storage = build_test_solver(config, DummyCardAbstraction())
    np.random.seed(7)
    for _ in range(iterations):
        solver.train_iteration()
    return np.concatenate([infoset.regrets.copy() for infoset in storage.iter_infosets()])


class TestTheWeightingReachesTheMath:
    @pytest.mark.timeout(60)
    def test_dcfr_and_linear_do_not_produce_the_same_table(self):
        """The knob is read at the update, not just stored on the config."""
        assert not np.allclose(_train("dcfr"), _train("linear"))

    @pytest.mark.timeout(60)
    def test_dcfr_carries_less_negative_regret_than_linear(self):
        """beta=0 HALVES negative regret on every visit; linear discounts none
        of it. That sign convention is the load-bearing half of DCFR here --
        paired at 30M, beta=0.5 and linear both scored ~1000 WORSE."""
        dcfr, linear = _train("dcfr"), _train("linear")
        assert np.abs(np.minimum(dcfr, 0.0)).sum() < np.abs(np.minimum(linear, 0.0)).sum()

    def test_training_under_dcfr_still_fills_the_table(self):
        """The floor the deleted `test_dcfr_convergence` actually checked, kept
        as one cheap assertion rather than a test named for convergence."""
        assert (_train("dcfr") != 0.0).any()
