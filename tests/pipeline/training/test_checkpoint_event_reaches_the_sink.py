"""The checkpoint event goes where every other event goes.

It was appended straight to `run.jsonl` by the trainer, so it was the ONE event
type the record sink never saw. `progress` reads checkpoint events, and once it
answered from the database it found none for any run since the last import --
which a comparison over historical runs cannot catch, because those had all been
imported. Caught by `backfill-record --verify` on a live run instead: share 6
events, database 5.
"""

from __future__ import annotations

from typing import Any

from src.core.actions.action_model import ActionModel
from src.pipeline.training.run_tracker import RunTracker
from src.shared import run_events
from src.shared.config import Config


class _Sink:
    def __init__(self) -> None:
        self.events: list[str] = []

    def opened(self, run_id: str, body: Any) -> None:
        self.events.append("created")

    def closed(self, run_id: str, status: str, body: Any) -> None:
        self.events.append("status")

    def emit(self, run_id: str, event: str, body: Any) -> None:
        self.events.append(event)

    def claim(self, run_id: str, iteration: int, uri: str) -> None: ...

    def flush(self, timeout: float) -> bool:
        return True


def _tracker(tmp_path, sink):
    config = Config.default()
    return RunTracker(
        run_dir=tmp_path / "run-a",
        config_name="test",
        config=config,
        action_config_hash=ActionModel(config).get_config_hash(),
        sink=sink,
    )


def test_a_checkpoint_reaches_the_sink(tmp_path):
    sink = _Sink()
    tracker = _tracker(tmp_path, sink)
    tracker.record_checkpoint(iteration=1000, coverage=0.5)
    assert run_events.CHECKPOINT in sink.events


def test_it_also_reaches_the_log(tmp_path):
    """Both stores, like every other event -- the share stays the source of
    truth."""
    sink = _Sink()
    tracker = _tracker(tmp_path, sink)
    tracker.record_checkpoint(iteration=1000, coverage=0.5)
    kinds = [e.get("event") for e in run_events.read(tracker.run_dir)]
    assert run_events.CHECKPOINT in kinds


def test_a_sink_that_explodes_does_not_fail_the_rung(tmp_path):
    """It runs immediately after `save_checkpoint` succeeded, so an error here
    would throw away a good rung and mark the run failed over telemetry."""

    class _Explodes(_Sink):
        def emit(self, run_id: str, event: str, body: Any) -> None:
            raise RuntimeError("gone")

    tracker = _tracker(tmp_path, _Explodes())
    tracker.record_checkpoint(iteration=1000)


def test_the_trainer_offers_the_hook_and_the_services_pass_it():
    """Two services drive the trainer and both must hand it the tracker's
    recorder; one that forgets is a run whose checkpoint series never leaves the
    file."""
    import inspect

    from src.pipeline.services import pcs_training, static_training
    from src.pipeline.training import static_parallel

    assert "on_checkpoint" in inspect.signature(static_parallel.train_static_parallel).parameters
    for module in (static_training, pcs_training):
        source = inspect.getsource(module)
        assert "on_checkpoint=tracker.record_checkpoint" in source, module.__name__
