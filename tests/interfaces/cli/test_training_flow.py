"""The interactive training flow, now a cloud client.

Two things are pinned here. The first is old: selecting a config must realign
``ctx.runs_dir``, or every later run-related prompt looks in the wrong place.
The second is the reason the flow was rewired at all -- the menu must build the
same leg spec the headless ``submit`` builds, with an ABSOLUTE target and room
for experiment tags. It previously did neither, which made it the one surface
speaking a relative-target dialect and left its runs permanently invisible to
``report --experiment``.
"""

from unittest.mock import MagicMock

from src.interfaces.cli.flows import training
from src.interfaces.cli.ui.context import CliContext
from src.interfaces.cloud import spec
from src.shared.config import Config


def _make_ctx(tmp_path):
    return CliContext(
        base_dir=tmp_path.resolve(),
        config_dir=tmp_path / "config",
        runs_dir=tmp_path / "data" / "runs",
        equity_buckets_dir=tmp_path / "data" / "equity_buckets",
        style=MagicMock(),
    )


def _stub_flow(monkeypatch, config, *, target=1000, tags=("", "", ""), continuing=False):
    """Drive the prompts, and capture the legs the flow would have queued."""
    captured: list[list[spec.LegSpec]] = []

    monkeypatch.setattr(training, "select_config", lambda _ctx: config)
    monkeypatch.setattr(training, "_ensure_combo_abstraction", lambda _ctx, _config: True)
    monkeypatch.setattr(training, "_prompt_experiment_tags", lambda _ctx: tags)
    monkeypatch.setattr(training.prompts, "confirm", lambda *a, **k: continuing)
    monkeypatch.setattr(training.prompts, "prompt_int", lambda *a, **k: target)
    monkeypatch.setattr(training.ui, "header", lambda _title: None)
    monkeypatch.setattr(training.ui, "pause", lambda: None)
    monkeypatch.setattr(training, "_queue", lambda make_legs: captured.append(make_legs("snap-1")))
    return captured


def test_selecting_a_config_realigns_runs_dir_relative(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    config = Config.default().merge({"training": {"runs_dir": "custom_runs"}})
    captured = _stub_flow(monkeypatch, config)

    training.submit_training_leg(ctx)

    assert ctx.runs_dir == (tmp_path / "custom_runs").resolve()
    assert len(captured) == 1


def test_selecting_a_config_realigns_runs_dir_absolute(tmp_path, monkeypatch):
    ctx = _make_ctx(tmp_path)
    absolute = (tmp_path / "alt" / "runs").resolve()
    config = Config.default().merge({"training": {"runs_dir": str(absolute)}})
    _stub_flow(monkeypatch, config)

    training.submit_training_leg(ctx)

    assert ctx.runs_dir == absolute


class TestTheMenuSpeaksTheSameContractAsSubmit:
    def test_the_prompted_number_is_the_absolute_target(self, tmp_path, monkeypatch):
        """Not an increment. The menu used to ask for 'additional iterations'."""
        ctx = _make_ctx(tmp_path)
        captured = _stub_flow(monkeypatch, Config.default(), target=25_000_000)

        training.submit_training_leg(ctx)

        (leg,) = captured[0]
        assert leg.to == 25_000_000
        assert leg.op == spec.TRAIN

    def test_experiment_tags_reach_the_leg(self, tmp_path, monkeypatch):
        """Without these a run can never be paired against a control, and that
        cannot be repaired after the fact."""
        ctx = _make_ctx(tmp_path)
        captured = _stub_flow(
            monkeypatch, Config.default(), tags=("exp-7", "variant:pruning", "run-base")
        )

        training.submit_training_leg(ctx)

        (leg,) = captured[0]
        assert (leg.experiment, leg.arm, leg.parent) == ("exp-7", "variant:pruning", "run-base")

    def test_a_fresh_leg_carries_a_config_and_no_run_id(self, tmp_path, monkeypatch):
        ctx = _make_ctx(tmp_path)
        config = Config.default().merge({"system": {"config_name": "production"}})
        captured = _stub_flow(monkeypatch, config)

        training.submit_training_leg(ctx)

        (leg,) = captured[0]
        assert leg.config == "production"
        assert leg.run_id == ""
        leg.validate()

    def test_continuing_a_run_carries_a_run_id_and_no_config(self, tmp_path, monkeypatch):
        ctx = _make_ctx(tmp_path)
        captured = _stub_flow(monkeypatch, Config.default(), continuing=True)
        monkeypatch.setattr(training, "select_run", lambda *a, **k: "run-a")

        training.submit_training_leg(ctx)

        (leg,) = captured[0]
        assert leg.run_id == "run-a"
        assert leg.config == ""
        leg.validate()


class TestNoLocalComputeDoorRemains:
    def test_the_local_training_helpers_are_gone(self):
        """Their removal is the point of the rewiring, not a side effect."""
        for name in ("_start_training", "_resume_training", "_prompt_num_workers"):
            assert not hasattr(training, name), name
