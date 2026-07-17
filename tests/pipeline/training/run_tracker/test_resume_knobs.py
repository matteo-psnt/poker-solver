"""A continuation must not quietly become a different trainer.

`--set` overrides do not carry into `submit --run ... --to N`: the node rebuilds
the config from `--config` plus whatever flags that task passed. The run's own
record is the only thing that knows what it was started with, and the other two
resume checks -- the action hash and the card-abstraction hash -- both pass
while the algorithm changes underneath them.
"""

from __future__ import annotations

import pytest

from src.core.actions.action_model import ActionModel
from src.pipeline.services.pcs_training import TRAINER_BLOCKS as PCS_BLOCKS
from src.pipeline.services.static_training import TRAINER_BLOCKS as SCALAR_BLOCKS
from src.pipeline.training.run_tracker import RunTracker
from src.shared.config import Config


def _tracker(tmp_path, config: Config) -> RunTracker:
    return RunTracker(
        run_dir=tmp_path / "run-test",
        config_name="test",
        config=config,
        action_config_hash=ActionModel(config).get_config_hash(),
    )


def _with(config: Config, block: str, **fields) -> Config:
    return config.model_copy(update={block: getattr(config, block).model_copy(update=fields)})


def test_matching_knobs_pass(tmp_path):
    config = Config.default()
    _tracker(tmp_path, config).verify_trainer_knobs(config, PCS_BLOCKS)


@pytest.mark.parametrize(
    ("block", "fields"),
    [
        ("pcs", {"cfr_br": "river"}),
        ("pcs", {"runouts_per_flop": 4}),
        ("solver", {"cfr_plus": True}),
        ("solver", {"iteration_weighting": "dcfr"}),
        ("solver", {"dcfr_alpha": 2.5}),
    ],
)
def test_a_changed_trainer_knob_refuses_the_resume(tmp_path, block, fields):
    """The failure this exists for: a CFR-BR ladder continued as plain PCS."""
    stored = Config.default()
    current = _with(stored, block, **fields)
    assert current != stored
    with pytest.raises(ValueError, match="does not carry into a continuation"):
        _tracker(tmp_path, stored).verify_trainer_knobs(current, PCS_BLOCKS)


def test_the_scalar_trainer_does_not_police_a_section_it_never_reads(tmp_path):
    """`pcs` knobs are not part of what the scalar trainer IS.

    Refusing on them would block legitimate scalar resumes for a difference that
    changes nothing about the run.
    """
    stored = Config.default()
    current = _with(stored, "pcs", cfr_br="river")
    tracker = _tracker(tmp_path, stored)
    tracker.verify_trainer_knobs(current, SCALAR_BLOCKS)  # allowed
    with pytest.raises(ValueError, match="does not carry into a continuation"):
        tracker.verify_trainer_knobs(current, PCS_BLOCKS)


def test_the_scalar_trainer_still_refuses_a_changed_solver(tmp_path):
    stored = Config.default()
    current = _with(stored, "solver", cfr_plus=True)
    with pytest.raises(ValueError, match="does not carry into a continuation"):
        _tracker(tmp_path, stored).verify_trainer_knobs(current, SCALAR_BLOCKS)
