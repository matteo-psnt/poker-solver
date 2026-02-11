"""Tests for training service-layer orchestration."""

from types import SimpleNamespace
from unittest.mock import MagicMock

from src.training import services


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


def test_evaluate_run_returns_output(monkeypatch, tmp_path):
    """evaluate_run should build solver, compute exploitability, and return output."""
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
    monkeypatch.setattr(services, "InMemoryStorage", lambda checkpoint_dir: storage)
    monkeypatch.setattr(services, "build_action_abstraction", lambda cfg: "action_abs")
    monkeypatch.setattr(services, "build_card_abstraction", lambda cfg, **kwargs: "card_abs")
    monkeypatch.setattr(services, "build_solver", lambda cfg, action, card, st: FakeSolver())
    monkeypatch.setattr(services, "MCCFRSolver", FakeSolver)
    monkeypatch.setattr(
        services,
        "compute_exploitability",
        lambda solver, **kwargs: expected_results,
    )

    output = services.evaluate_run(
        run_dir=tmp_path / "run-1",
        num_samples=50,
        num_rollouts=7,
        use_average_strategy=True,
        seed=42,
    )

    assert output.infosets == 1234
    assert output.results == expected_results
