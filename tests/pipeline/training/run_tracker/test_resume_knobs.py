"""A continuation trains what it was.

The config a task carries -- `--config` plus its own `--set` flags -- once
rebuilt the trainer on every continuation, and a CFR-BR ladder was one
forgotten flag from continuing as plain PCS with the action hash, the
abstraction hash and the kernel name all still matching. Now the record's
config is the only one a continuation reads. It is adopted, not enforced by
refusal, because a Batch retry of a FRESH run re-runs that run's own argv --
`--config` and `--set` included -- against the record its first attempt wrote.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from src.pipeline.services import pcs_training, static_training
from src.pipeline.training.run_tracker import RunTracker, continued_config
from src.pipeline.training.run_tracker import tracker as tracker_module
from src.shared.config import Config


def _tracker(tmp_path) -> RunTracker:
    stored = Config.default().merge({"solver": {"cfr_plus": True}})
    return RunTracker(
        run_dir=tmp_path / "run-a",
        config_name="test",
        config=stored,
        action_config_hash="abc123",
    )


def test_the_record_wins_over_whatever_the_task_carried(tmp_path, monkeypatch):
    """Through a mock, not caplog: `configure_logging` elsewhere in the suite
    detaches the root handler caplog listens on."""
    log = MagicMock()
    monkeypatch.setattr(tracker_module, "logger", log)
    tracker = _tracker(tmp_path)
    config = continued_config(tracker, "production", {"solver__cfr_plus": "false"}, seed=7)
    assert config == tracker.metadata.config
    assert config.solver.cfr_plus is True
    assert log.info.call_args.args[-1] == "--config, --set, --seed"


def test_a_bare_continuation_says_nothing(tmp_path, monkeypatch):
    log = MagicMock()
    monkeypatch.setattr(tracker_module, "logger", log)
    continued_config(_tracker(tmp_path), None, {}, None)
    log.info.assert_not_called()


SERVICES = [
    lambda name, **kw: pcs_training.train_pcs(name, iterations=10, **kw),
    lambda name, **kw: static_training.train_static(name, num_iterations=10, **kw),
]


@pytest.mark.parametrize("train", SERVICES, ids=["pcs", "scalar"])
def test_a_fresh_run_still_needs_a_config(tmp_path, train):
    with pytest.raises(ValueError, match="fresh run needs a config"):
        train(None, runs_dir=tmp_path)
