"""A continuation must not quietly become a different trainer.

`--set` overrides do not carry into a `submit --run ... --to N`: the node
rebuilds the config from `--config` plus whatever flags that task passed. The
run's own metadata is the only record of what it was trained with, so the
resume path has to read it.
"""

from __future__ import annotations

import pytest

from src.pipeline.services.pcs_training import verify_trainer_knobs
from tests.test_helpers import make_test_config


def _with(config, block: str, **fields):
    return config.model_copy(update={block: getattr(config, block).model_copy(update=fields)})


def test_matching_knobs_pass():
    config = make_test_config(seed=1)
    verify_trainer_knobs(config, config)


@pytest.mark.parametrize(
    ("block", "fields"),
    [
        ("pcs", {"cfr_br": "river"}),
        ("pcs", {"runouts_per_flop": 4}),
        ("solver", {"cfr_plus": True}),
        ("solver", {"iteration_weighting": "dcfr"}),
    ],
)
def test_a_changed_trainer_knob_refuses_the_resume(block, fields):
    """The failure this exists for: a CFR-BR ladder continued as plain PCS.

    Nothing else on the resume path looks at `solver` or `pcs` — the action and
    card-abstraction checks both pass — so without this the rungs simply come
    from two different algorithms and the curve is meaningless.
    """
    stored = make_test_config(seed=1)
    current = _with(stored, block, **fields)
    assert current != stored
    with pytest.raises(ValueError, match="does not carry into a continuation"):
        verify_trainer_knobs(stored, current)
