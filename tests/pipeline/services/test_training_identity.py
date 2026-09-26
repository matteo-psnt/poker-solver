"""A continuation must preserve the meaning of every checkpoint row."""

from __future__ import annotations

from functools import partial
from unittest.mock import Mock

import pytest

from src.core.actions.action_model import ActionModel
from src.pipeline.services import pcs_training, run_opening, static_training
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
    monkeypatch.setattr(run_opening, "load_training_config", lambda *_a, **_kw: config)
    build = Mock(return_value=Buckets())
    monkeypatch.setattr(run_opening.blueprint, "build_card_abstraction", build)
    resolve = Mock(return_value="original-buckets")
    monkeypatch.setattr(run_opening.blueprint, "resolve_card_abstraction_hash", resolve)
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

    return kernel, invoke, resolve, build, train, record


def test_changed_abstraction_refuses_before_training_or_record_mutation(continuation):
    kernel, invoke, resolve, _build, train, record = continuation
    if kernel == "pcs":
        pytest.skip("PCS pins the recorded abstraction instead; see the test below")
    resolve.return_value = "replacement-with-the-same-bucket-counts"
    before = list(record.rows)
    with pytest.raises(ValueError, match="Card abstraction hash does not match"):
        invoke()
    train.assert_not_called()
    assert record.rows == before


def test_matching_abstraction_reaches_training_with_the_recorded_identity(continuation):
    _kernel, invoke, _resolve, _build, train, _record = continuation
    with pytest.raises(ReachedTrainingError):
        invoke()
    assert train.call_args.kwargs["abstraction_id"] == "original-buckets"


def test_pcs_continues_on_its_recorded_abstraction_when_the_preset_moved(continuation):
    """PCS resumes by the hash the run RECORDED, so a preset renamed or rebuilt
    since cannot mix two abstractions into one ladder -- the run keeps training
    on the one its rows already mean."""
    kernel, invoke, resolve, build, train, _record = continuation
    if kernel != "pcs":
        pytest.skip("the scalar service refuses instead; see the test above")
    resolve.return_value = "replacement-with-the-same-bucket-counts"
    with pytest.raises(ReachedTrainingError):
        invoke()
    assert build.call_args.kwargs["abstraction_hash"] == "original-buckets"
    assert train.call_args.kwargs["abstraction_id"] == "original-buckets"


@pytest.mark.parametrize(
    ("recorded", "service"), [("pcs", "scalar"), ("scalar", "pcs"), (None, "pcs")]
)
def test_a_run_continues_only_under_the_kernel_that_trained_it(
    recorded, service, monkeypatch, tmp_path
):
    """Two kernels in one ladder is two lineages. Only PCS used to refuse;
    a run recorded before the field (None) was scalar, so PCS refuses it too."""
    config = _config()
    record = MemoryRecord()
    tracker = RunTracker(
        run_dir=tmp_path / "existing",
        config_name="test",
        config=config,
        action_config_hash=ActionModel(config).get_config_hash(),
        card_abstraction_hash="original-buckets",
        kernel=recorded,
        sink=record,
        source=record,
    )
    tracker.mark_completed()
    monkeypatch.setattr(run_opening, "load_training_config", lambda *_a, **_kw: config)
    monkeypatch.setattr(
        run_opening.blueprint, "resolve_card_abstraction_hash", lambda _c: "original-buckets"
    )
    invoke = (
        partial(pcs_training.train_pcs, "test", iterations=2)
        if service == "pcs"
        else partial(static_training.train_static, "test", num_iterations=2)
    )
    with pytest.raises(ValueError, match="kernel"):
        invoke(
            num_workers=1, runs_dir=tmp_path, run_id="existing", sink=record, record_source=record
        )
