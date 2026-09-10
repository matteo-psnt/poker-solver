"""A continuation trains what it was.

The config a task carries -- `--config` plus its own `--set` flags -- once
rebuilt the trainer on every continuation, and a CFR-BR ladder was one
forgotten flag from continuing as plain PCS with the action hash, the
abstraction hash and the kernel name all still matching. Now the record's
config is the only one a continuation reads, and naming another is refused.
"""

from __future__ import annotations

import pytest

from src.pipeline.services import pcs_training, static_training
from src.pipeline.training.run_tracker import refuse_config_on_continue


@pytest.mark.parametrize(
    "named",
    [
        {"config_name": "production"},
        {"overrides": {"solver__cfr_plus": "true"}},
        {"seed": 7},
        {"config_name": "production", "overrides": {"pcs__cfr_br": "river"}, "seed": 1},
    ],
)
def test_a_continuation_refuses_anything_that_would_rebuild_its_config(named):
    with pytest.raises(ValueError, match="already has a config on its record"):
        refuse_config_on_continue(
            "run-a",
            named.get("config_name"),
            named.get("overrides"),
            named.get("seed"),
        )


def test_a_bare_continuation_passes():
    refuse_config_on_continue("run-a", None, {}, None)


SERVICES = [
    lambda name, **kw: pcs_training.train_pcs(name, iterations=10, **kw),
    lambda name, **kw: static_training.train_static(name, num_iterations=10, **kw),
]


@pytest.mark.parametrize("train", SERVICES, ids=["pcs", "scalar"])
def test_the_service_refuses_before_touching_the_run(tmp_path, train):
    """Refused on the record alone: no tracker load, no abstraction, no tree."""
    (tmp_path / "run-a").mkdir()
    (tmp_path / "run-a" / ".run.json").write_text("{}")
    with pytest.raises(ValueError, match="already has a config on its record"):
        train("production", run_id="run-a", runs_dir=tmp_path)


@pytest.mark.parametrize("train", SERVICES, ids=["pcs", "scalar"])
def test_a_fresh_run_still_needs_a_config(tmp_path, train):
    with pytest.raises(ValueError, match="fresh run needs a config"):
        train(None, runs_dir=tmp_path)
