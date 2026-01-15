"""Tests for training service-layer orchestration."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.pipeline.evaluation.hunl_local_best_response import HandOutcome
from src.pipeline.training import services
from src.pipeline.training.services import TrainingOutput
from src.shared.config_loader import load_training_config


def _fake_config(runs_dir: str = "data/runs", abstraction: str = "quick_test") -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(runs_dir=runs_dir),
        card_abstraction=SimpleNamespace(config=abstraction),
    )


def _fake_metadata() -> SimpleNamespace:
    return SimpleNamespace(
        run_id="run-xyz",
        config_name="quick_test",
        iterations=2000,
        num_infosets=1234,
        runtime_seconds=10.0,
        storage_capacity=1_000_000,
        status="completed",
    )


def test_train_builds_output_and_forwards_seed(monkeypatch):
    """train should apply the seed override and build a portable TrainingOutput."""
    seen = {}
    config = _fake_config()

    def _mock_load(config_name, **overrides):
        seen["config_name"] = config_name
        seen["overrides"] = overrides
        return config

    monkeypatch.setattr(services, "load_training_config", _mock_load)
    monkeypatch.setattr(
        services,
        "create_training_session",
        lambda cfg: SimpleNamespace(run_dir=Path("data/runs/run-xyz")),
    )
    monkeypatch.setattr(services, "run_training", lambda sess, **kw: seen.update(run_kwargs=kw))
    monkeypatch.setattr(services, "load_run_metadata", lambda run_dir: _fake_metadata())

    out = services.train("quick_test", num_workers=4, num_iterations=2000, seed=7)

    assert isinstance(out, TrainingOutput)
    assert seen["config_name"] == "quick_test"
    assert seen["overrides"] == {"system__seed": 7}
    assert seen["run_kwargs"] == {"num_workers": 4, "num_iterations": 2000}
    assert out.run_id == "run-xyz"
    assert out.runs_dir == "data/runs"
    assert out.iterations == 2000
    assert out.iterations_per_second == pytest.approx(200.0)
    assert out.status == "completed"


def test_train_omits_seed_override_when_absent(monkeypatch):
    """train should not inject a seed override when seed is None."""
    seen = {}
    monkeypatch.setattr(
        services,
        "load_training_config",
        lambda name, **ov: seen.update(overrides=ov) or _fake_config(),
    )
    monkeypatch.setattr(
        services, "create_training_session", lambda cfg: SimpleNamespace(run_dir=Path("d"))
    )
    monkeypatch.setattr(services, "run_training", lambda sess, **kw: None)
    monkeypatch.setattr(services, "load_run_metadata", lambda run_dir: _fake_metadata())

    services.train("quick_test")

    assert seen["overrides"] == {}


def test_train_translates_missing_abstraction(monkeypatch):
    """A missing abstraction should surface an actionable precompute message."""
    monkeypatch.setattr(services, "load_training_config", lambda name, **ov: _fake_config())

    def _raise(cfg):
        raise FileNotFoundError("no such file")

    monkeypatch.setattr(services, "create_training_session", _raise)

    with pytest.raises(FileNotFoundError, match="Precompute it"):
        services.train("quick_test")


def test_train_translates_stale_abstraction(monkeypatch):
    """A hash-mismatch ValueError should surface an actionable recompute message."""
    monkeypatch.setattr(services, "load_training_config", lambda name, **ov: _fake_config())

    def _raise(cfg):
        raise ValueError("config hash mismatch")

    monkeypatch.setattr(services, "create_training_session", _raise)

    with pytest.raises(ValueError, match="stale"):
        services.train("quick_test")


def test_train_reraises_unrelated_value_error(monkeypatch):
    """Non-abstraction ValueErrors should propagate unchanged."""
    monkeypatch.setattr(services, "load_training_config", lambda name, **ov: _fake_config())

    def _raise(cfg):
        raise ValueError("something else entirely")

    monkeypatch.setattr(services, "create_training_session", _raise)

    with pytest.raises(ValueError, match="something else entirely"):
        services.train("quick_test")


def test_precompute_abstraction_skips_when_present(monkeypatch, tmp_path):
    """A complete abstraction on disk is reused (no rebuild) unless overwrite=True."""
    out = tmp_path / "data" / "combo_abstraction" / "buckets-x"
    out.mkdir(parents=True)
    (out / "metadata.json").write_text("{}")

    monkeypatch.setattr(services.PrecomputeConfig, "from_yaml", lambda name: SimpleNamespace())
    monkeypatch.setattr(services, "abstraction_output_path", lambda base, cfg: out)
    built = {"n": 0}
    monkeypatch.setattr(services, "PostflopPrecomputer", lambda cfg: built.update(n=built["n"] + 1))

    result = services.precompute_abstraction("quick_test", base_dir=tmp_path)

    assert result == out
    assert built["n"] == 0  # never constructed the precomputer


def test_precompute_abstraction_runs_and_saves(monkeypatch, tmp_path):
    """When missing, apply num_workers, run all three streets, and save to the dir."""
    out = tmp_path / "data" / "combo_abstraction" / "buckets-y"
    seen: dict = {}
    base_cfg = SimpleNamespace(
        model_copy=lambda update: seen.update(update=update) or SimpleNamespace(tag="copied")
    )
    monkeypatch.setattr(services.PrecomputeConfig, "from_yaml", lambda name: base_cfg)
    monkeypatch.setattr(services, "abstraction_output_path", lambda base, cfg: out)

    events: list = []

    class _FakePrecomputer:
        def __init__(self, cfg):
            events.append(("init", cfg))

        def precompute_all(self, streets):
            events.append(("all", tuple(streets)))

        def save(self, path):
            events.append(("save", path))

    monkeypatch.setattr(services, "PostflopPrecomputer", _FakePrecomputer)

    result = services.precompute_abstraction("quick_test", num_workers=8, base_dir=tmp_path)

    assert result == out
    assert seen["update"] == {"num_workers": 8}
    kind, init_cfg = events[0]
    assert kind == "init"
    assert getattr(init_cfg, "tag", None) == "copied"  # the num_workers-overridden config
    assert any(k == "all" and len(streets) == 3 for k, streets in events)
    assert ("save", out) in events


def test_load_training_config_reads_named_yaml_and_applies_overrides():
    """load_training_config should resolve config/training/<name>.yaml and apply overrides."""
    config = load_training_config("quick_test", system__seed=123)

    assert config.system.config_name == "quick_test"
    assert config.system.seed == 123
    assert config.training.num_iterations == 3600


def test_list_runs_delegates_to_run_tracker(monkeypatch, tmp_path):
    """list_runs should delegate to RunTracker.list_runs."""
    expected = ["run-a", "run-b"]
    seen = {}

    def _mock_list_runs(base_dir):
        seen["base_dir"] = base_dir
        return expected

    monkeypatch.setattr(services.RunTracker, "list_runs", _mock_list_runs)

    actual = services.list_runs(tmp_path)

    assert actual == expected
    assert seen["base_dir"] == tmp_path


def test_load_run_metadata_delegates_to_run_tracker(monkeypatch, tmp_path):
    """load_run_metadata should return loaded tracker metadata."""
    metadata = SimpleNamespace(status="running")
    tracker = SimpleNamespace(metadata=metadata)

    monkeypatch.setattr(services.RunTracker, "load", lambda run_dir: tracker)

    actual = services.load_run_metadata(tmp_path / "run-1")

    assert actual is metadata


def test_create_resumed_session_uses_metadata_iteration(monkeypatch, tmp_path):
    """create_resumed_session should return resumed session and latest iteration."""
    metadata = SimpleNamespace(iterations=321)
    session = MagicMock(name="session")

    monkeypatch.setattr(services, "load_run_metadata", lambda run_dir: metadata)
    monkeypatch.setattr(services.TrainingSession, "resume", lambda run_dir: session)

    actual_session, latest = services.create_resumed_session(tmp_path / "run-1")

    assert actual_session is session
    assert latest == 321


def test_run_training_passes_arguments():
    """run_training should forward arguments to session.train."""
    session = MagicMock()

    services.run_training(session, num_workers=4, num_iterations=1000)

    session.train.assert_called_once_with(num_workers=4, num_iterations=1000)


def test_start_training_uses_create_and_run(monkeypatch):
    """start_training should create a session and run it."""
    config = MagicMock()
    session = MagicMock()
    seen = {}

    monkeypatch.setattr(services, "create_training_session", lambda cfg: session)

    def _mock_run_training(sess, **kwargs):
        seen["session"] = sess
        seen["kwargs"] = kwargs

    monkeypatch.setattr(services, "run_training", _mock_run_training)

    actual = services.start_training(config, num_workers=6)

    assert actual is session
    assert seen["session"] is session
    assert seen["kwargs"] == {"num_workers": 6}


def test_evaluate_run_rollout_returns_output(monkeypatch, tmp_path):
    """evaluate_run_rollout should build solver, compute exploitability, and return output."""
    config = MagicMock(name="config")
    metadata = SimpleNamespace(config=config)
    storage = MagicMock(name="storage")
    storage.num_infosets.return_value = 1234

    class FakeSolver:
        pass

    expected_results = {
        "exploitability_mbb": 1.23,
        "std_error_mbb": 0.1,
        "confidence_95_mbb": (1.0, 1.4),
        "player_0_br_utility": 0.01,
        "player_1_br_utility": -0.01,
        "num_samples": 50,
    }

    monkeypatch.setattr(services, "load_run_metadata", lambda run_dir: metadata)
    monkeypatch.setattr(
        services,
        "build_evaluation_solver",
        lambda cfg, checkpoint_dir: (FakeSolver(), storage),
    )
    monkeypatch.setattr(
        services,
        "evaluate_solver_exploitability",
        lambda solver, **kwargs: expected_results,
    )

    output = services.evaluate_run_rollout(
        run_dir=tmp_path / "run-1",
        num_samples=50,
        num_rollouts=7,
        use_average_strategy=True,
        seed=42,
    )

    assert output.infosets == 1234
    assert output.results == expected_results


def test_evaluate_run_lbr_maps_result_and_builds_config(monkeypatch, tmp_path):
    """evaluate_run_lbr should run LBR and map LBRResult into the results dict."""
    metadata = SimpleNamespace(config=MagicMock(name="config"))
    metadata.config.game.big_blind = 100
    storage = MagicMock(name="storage")
    storage.num_infosets.return_value = 4321
    hand_outcomes = [
        (
            HandOutcome(value=150.0, terminal="showdown", pot=400),
            HandOutcome(value=-50.0, terminal="fold", pot=200),
        ),
        (
            HandOutcome(value=800.0, terminal="allin", pot=4000),
            HandOutcome(value=100.0, terminal="showdown", pot=600),
        ),
    ]
    lbr_result = SimpleNamespace(
        exploitability_mbb=42.0,
        exploitability_bb=0.042,
        std_error_mbb=1.5,
        confidence_95_mbb=(39.0, 45.0),
        lbr_utility_p0=0.02,
        lbr_utility_p1=0.03,
        num_hands=2000,
        base_seed=7,
        hand_outcomes=hand_outcomes,
    )
    seen = {}

    monkeypatch.setattr(services, "load_run_metadata", lambda run_dir: metadata)
    monkeypatch.setattr(
        services, "build_evaluation_solver", lambda cfg, checkpoint_dir: (object(), storage)
    )
    monkeypatch.setattr(
        services,
        "compute_lbr_exploitability",
        lambda solver, cfg, **kw: seen.update(cfg=cfg) or lbr_result,
    )

    output = services.evaluate_run_lbr(
        run_dir=tmp_path / "run-1", num_hands=2000, equity_runouts=8, seed=7
    )

    assert output.infosets == 4321
    assert output.results["exploitability_mbb"] == 42.0
    assert output.results["confidence_95_mbb"] == (39.0, 45.0)
    assert output.results["lbr_utility_p0"] == 0.02
    assert output.results["num_hands"] == 2000
    # Per-hand records + ready-made paired samples + base seed travel with the
    # aggregate for paired comparisons.
    assert output.results["base_seed"] == 7
    assert output.results["big_blind"] == 100
    assert output.results["pair_samples_mbb"] == [
        pytest.approx((150.0 - 50.0) / 2 / 100 * 1000),
        pytest.approx((800.0 + 100.0) / 2 / 100 * 1000),
    ]
    assert output.results["hand_records"] == [
        {
            "u0": 150.0,
            "u1": -50.0,
            "terminal_p0": "showdown",
            "terminal_p1": "fold",
            "pot_p0": 400,
            "pot_p1": 200,
        },
        {
            "u0": 800.0,
            "u1": 100.0,
            "terminal_p0": "allin",
            "terminal_p1": "showdown",
            "pot_p0": 4000,
            "pot_p1": 600,
        },
    ]
    # Deal-level groups take the higher-variance terminal: showdown+fold -> showdown,
    # allin+showdown -> allin.
    decomposition = output.results["variance_decomposition"]
    assert set(decomposition["groups"]) == {"showdown", "allin"}
    # LBRConfig constructed from the call args, with the rigorous off-tree default.
    assert seen["cfg"].num_hands == 2000
    assert seen["cfg"].equity_runouts == 8
    assert seen["cfg"].seed == 7
    assert seen["cfg"].include_off_tree is False
    assert seen["cfg"].allin_runouts == 50
