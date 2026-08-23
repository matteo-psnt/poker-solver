"""The public-chance-sampling task: the scalar contract plus a retention knob."""

from __future__ import annotations

import dataclasses

import pytest

from src.interfaces.cloud.tasks import spec
from src.shared.cloudtask import kinds
from src.shared.cloudtask.kinds import BadTaskError, TaskName


def _leg(**overrides) -> spec.TaskSpec:
    base = spec.TaskSpec(
        code_snapshot="snap-1", op=TaskName.TRAIN_PCS, config="production", to=9600
    )
    return dataclasses.replace(base, **overrides)


def test_a_pcs_leg_is_accepted():
    _leg().validate()


def test_the_target_is_absolute_here_too():
    with pytest.raises(BadTaskError, match="ABSOLUTE"):
        _leg(to=0).validate()


def test_a_pcs_leg_needs_a_config_even_when_continuing():
    with pytest.raises(BadTaskError, match="config"):
        _leg(config="", run_id="run-a").validate()


def test_retention_reaches_the_node_and_a_scalar_leg_leaves_it_empty():
    assert _leg(retain_every=800).environment()["RUN_RETAIN_EVERY"] == "800"
    scalar = spec.TaskSpec(code_snapshot="s", op=TaskName.TRAIN, config="production", to=1000)
    assert scalar.environment()["RUN_RETAIN_EVERY"] == ""


def test_the_label_names_the_kernel():
    """A run id on the pool is `run-<label>`, so the kernel shows in every record."""
    label = kinds.kind(TaskName.TRAIN_PCS).label(_leg(arm="pcs-w8"))
    assert label == "pcs-production-to9.6k-pcs-w8"
