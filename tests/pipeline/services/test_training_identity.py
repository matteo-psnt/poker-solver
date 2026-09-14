"""A continuation must preserve the meaning of every checkpoint row."""

from __future__ import annotations

from functools import partial
from unittest.mock import Mock

import pytest

from src.core.actions.action_model import ActionModel
from src.pipeline.services import pcs_training, static_training
from src.pipeline.training.run_tracker import RunTracker
from tests.memory_record import MemoryRecord
from tests.pipeline.training.test_static_parallel import Buckets, _config


class ReachedTrainingError(Exception):
    pass


@pytest.fixture(params=["pcs", "scalar"])
def continuation(request, monkeypatch, tmp_path):
    kernel = request.param
    service = pcs_training if kernel == "pcs" else static_training
    config = _config()
    record = MemoryRecord()
    tracker = RunTracker(
        run_dir=tmp_path / "existing",
        config_name="test",
        config=config,
        action_config_hash=ActionModel(config).get_config_hash(),
        card_abstraction_hash="original-buckets",
        kernel=kernel,
        sink=record,
        source=record,
    )
    tracker.mark_completed()
    monkeypatch.setattr(service, "load_training_config", lambda *_a, **_kw: config)
    monkeypatch.setattr(service.blueprint, "build_card_abstraction", lambda *_a, **_kw: Buckets())
    resolve = Mock(return_value="original-buckets")
    monkeypatch.setattr(service.blueprint, "resolve_card_abstraction_hash", resolve)
    train = Mock(side_effect=ReachedTrainingError)
    monkeypatch.setattr(service, "train_static_parallel", train)

    if kernel == "pcs":
        invoke = partial(
            pcs_training.train_pcs,
            "test",
            iterations=2,
            runs_dir=tmp_path,
            run_id="existing",
            sink=record,
            record_source=record,
            num_workers=1,
        )
    else:
        invoke = partial(
            static_training.train_static,
            "test",
            num_iterations=2,
            runs_dir=tmp_path,
            run_id="existing",
            sink=record,
            record_source=record,
            num_workers=1,
        )

    return invoke, resolve, train, record


def test_changed_abstraction_refuses_before_training_or_record_mutation(continuation):
    invoke, resolve, train, record = continuation
    resolve.return_value = "replacement-with-the-same-bucket-counts"
    before = list(record.rows)
    with pytest.raises(ValueError, match="Card abstraction hash does not match"):
        invoke()
    train.assert_not_called()
    assert record.rows == before


def test_matching_abstraction_reaches_training_with_the_recorded_identity(continuation):
    invoke, _resolve, train, _record = continuation
    with pytest.raises(ReachedTrainingError):
        invoke()
    assert train.call_args.kwargs["abstraction_id"] == "original-buckets"
