"""Tests for training service-layer orchestration."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.engine.solver.storage.static_array import StaticArrayStorage
from src.pipeline import services
from src.pipeline.abstraction.resolver import AbstractionHashMismatchError
from src.pipeline.evaluation.estimators.lbr.config import LBRConfig
from src.pipeline.evaluation.estimators.lbr.hunl_local_best_response import HandOutcome
from src.pipeline.services import bucketing as services_bucketing
from src.pipeline.services import runs as services_runs
from src.pipeline.services import static_training as services_training
from src.pipeline.services.scoring import _shared as services_shared
from src.pipeline.services.scoring import lbr as services_lbr
from src.shared.config.loader import load_training_config


def _solver() -> SimpleNamespace:
    """A blueprint stub with the one real attribute the recorder reads: the
    betting-tree fingerprint that says WHICH GAME the number describes."""
    return SimpleNamespace(tree=SimpleNamespace(fingerprint=lambda: "treefp0000000000"))


def _fake_config(runs_dir: str = "data/runs", abstraction: str = "quick_test") -> SimpleNamespace:
    return SimpleNamespace(
        training=SimpleNamespace(runs_dir=runs_dir, num_iterations=2000),
        card_abstraction=SimpleNamespace(config=abstraction),
        # train_static applies the run's own log level once loaded.
        system=SimpleNamespace(log_level="INFO"),
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


def _patch_abstraction_failure(monkeypatch, tmp_path, exc):
    """Point train_static at a fresh run dir whose abstraction resolution fails."""
    monkeypatch.setattr(
        services_training,
        "load_training_config",
        lambda name, **ov: _fake_config(runs_dir=str(tmp_path)),
    )
    monkeypatch.setattr(services_training, "ActionModel", lambda cfg: MagicMock())

    def _raise(_cfg):
        raise exc

    monkeypatch.setattr(services_training.blueprint, "resolve_card_abstraction_hash", _raise)


def test_train_static_translates_missing_abstraction(monkeypatch, tmp_path):
    """A missing abstraction should surface an actionable precompute message."""
    _patch_abstraction_failure(monkeypatch, tmp_path, FileNotFoundError("no such file"))

    with pytest.raises(FileNotFoundError, match="Precompute it"):
        services.train_static("quick_test")


def test_train_static_translates_stale_abstraction(monkeypatch, tmp_path):
    """A hash mismatch should surface an actionable recompute message."""
    _patch_abstraction_failure(
        monkeypatch, tmp_path, AbstractionHashMismatchError("config hash mismatch")
    )

    with pytest.raises(AbstractionHashMismatchError, match="stale"):
        services.train_static("quick_test")


def test_train_static_reraises_unrelated_value_error(monkeypatch, tmp_path):
    """Non-abstraction ValueErrors should propagate unchanged."""
    _patch_abstraction_failure(monkeypatch, tmp_path, ValueError("something else entirely"))

    with pytest.raises(ValueError, match="something else entirely"):
        services.train_static("quick_test")


def test_precompute_abstraction_skips_when_present(monkeypatch, tmp_path):
    """A complete abstraction on disk is reused (no rebuild) unless overwrite=True."""
    out = tmp_path / "data" / "combo_abstraction" / "buckets-x"
    out.mkdir(parents=True)
    (out / "metadata.json").write_text("{}")

    monkeypatch.setattr(
        services_bucketing.PrecomputeConfig, "from_yaml", lambda name: SimpleNamespace()
    )
    monkeypatch.setattr(services_bucketing, "abstraction_output_path", lambda base, cfg: out)
    built = {"n": 0}
    monkeypatch.setattr(
        services_bucketing, "PostflopPrecomputer", lambda cfg: built.update(n=built["n"] + 1)
    )

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
    monkeypatch.setattr(services_bucketing.PrecomputeConfig, "from_yaml", lambda name: base_cfg)
    monkeypatch.setattr(services_bucketing, "abstraction_output_path", lambda base, cfg: out)

    events: list = []

    class _FakePrecomputer:
        def __init__(self, cfg):
            events.append(("init", cfg))

        def precompute_all(self, streets, on_progress=None):
            events.append(("all", tuple(streets)))

        def save(self, path):
            events.append(("save", path))

    monkeypatch.setattr(services_bucketing, "PostflopPrecomputer", _FakePrecomputer)

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

    monkeypatch.setattr(services_runs.RunTracker, "list_runs", _mock_list_runs)

    actual = services.list_runs(tmp_path)

    assert actual == expected
    assert seen["base_dir"] == tmp_path


def test_load_run_metadata_delegates_to_run_tracker(monkeypatch, tmp_path):
    """load_run_metadata should return loaded tracker metadata."""
    metadata = SimpleNamespace(status="running")
    tracker = SimpleNamespace(metadata=metadata)

    monkeypatch.setattr(services_runs.RunTracker, "load", lambda run_dir: tracker)

    actual = services_runs.load_run_metadata(tmp_path / "run-1")

    assert actual is metadata


def test_evaluate_run_lbr_refuses_run_without_recorded_abstraction(monkeypatch, tmp_path):
    """A run with no recorded abstraction hash must not be evaluated by name.

    Name-only resolution silently rebuckets the checkpoint under whatever abstraction
    the name now points at, producing plausible but invalid numbers.
    """
    metadata = SimpleNamespace(config=MagicMock(name="config"), card_abstraction_hash=None)
    metadata.config.game.big_blind = 100
    monkeypatch.setattr(services_shared, "load_run_metadata", lambda run_dir: metadata)

    with pytest.raises(ValueError, match="does not record which card abstraction"):
        services.evaluate_run_lbr(tmp_path / "run-legacy", LBRConfig(num_hands=1))


def test_evaluate_run_lbr_pins_hash_recorded_on_run(monkeypatch, tmp_path):
    """The run's own recorded hash pins the abstraction without the caller passing it."""
    metadata = SimpleNamespace(config=MagicMock(name="config"), card_abstraction_hash="recorded99")
    metadata.config.game.big_blind = 100
    storage = MagicMock(name="storage", spec=StaticArrayStorage)
    storage.num_infosets.return_value = 1
    seen = {}

    monkeypatch.setattr(services_shared, "load_run_metadata", lambda run_dir: metadata)
    monkeypatch.setattr(
        services_shared,
        "build_static_evaluation_solver",
        lambda cfg, checkpoint_dir, abstraction_hash=None, at_iteration=None: (
            seen.update(abstraction_hash=abstraction_hash) or (_solver(), storage)
        ),
    )
    monkeypatch.setattr(
        services_lbr,
        "compute_lbr_exploitability",
        lambda solver, cfg, **kw: SimpleNamespace(
            exploitability_mbb=1.0,
            exploitability_bb=0.001,
            std_error_mbb=0.1,
            confidence_95_mbb=(0.9, 1.1),
            lbr_utility_p0=0.0,
            lbr_utility_p1=0.0,
            num_hands=1,
            base_seed=1,
            hand_outcomes=[],
        ),
    )

    services.evaluate_run_lbr(tmp_path / "run-1", LBRConfig(num_hands=1))

    assert seen["abstraction_hash"] == "recorded99"


def test_evaluate_run_lbr_pins_abstraction_hash(monkeypatch, tmp_path):
    """A run must be evaluated under the abstraction it was trained against.

    Resolving by config name alone lets a recomputed abstraction silently rebucket an
    existing checkpoint (same bucket counts => in-range ids => no error, wrong strategy).
    """
    metadata = SimpleNamespace(config=MagicMock(name="config"), card_abstraction_hash="rec1")
    metadata.config.game.big_blind = 100
    storage = MagicMock(name="storage", spec=StaticArrayStorage)
    storage.num_infosets.return_value = 1
    seen = {}

    monkeypatch.setattr(services_shared, "load_run_metadata", lambda run_dir: metadata)
    monkeypatch.setattr(
        services_shared,
        "build_static_evaluation_solver",
        lambda cfg, checkpoint_dir, abstraction_hash=None, at_iteration=None: (
            seen.update(abstraction_hash=abstraction_hash) or (_solver(), storage)
        ),
    )
    monkeypatch.setattr(
        services_lbr,
        "compute_lbr_exploitability",
        lambda solver, cfg, **kw: SimpleNamespace(
            exploitability_mbb=1.0,
            exploitability_bb=0.001,
            std_error_mbb=0.1,
            confidence_95_mbb=(0.9, 1.1),
            lbr_utility_p0=0.0,
            lbr_utility_p1=0.0,
            num_hands=1,
            base_seed=1,
            hand_outcomes=[],
        ),
    )

    services.evaluate_run_lbr(tmp_path / "run-1", LBRConfig(num_hands=1), abstraction_hash="abc123")

    assert seen["abstraction_hash"] == "abc123"


def test_evaluate_run_lbr_maps_result_and_builds_config(monkeypatch, tmp_path):
    """evaluate_run_lbr should run LBR and map LBRResult into the results dict."""
    metadata = SimpleNamespace(config=MagicMock(name="config"), card_abstraction_hash="rec1")
    metadata.config.game.big_blind = 100
    storage = MagicMock(name="storage", spec=StaticArrayStorage)
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

    monkeypatch.setattr(services_shared, "load_run_metadata", lambda run_dir: metadata)
    monkeypatch.setattr(
        services_shared,
        "build_static_evaluation_solver",
        lambda cfg, checkpoint_dir, abstraction_hash=None, at_iteration=None: (
            seen.update(abstraction_hash=abstraction_hash) or (_solver(), storage)
        ),
    )
    monkeypatch.setattr(
        services_lbr,
        "compute_lbr_exploitability",
        lambda solver, cfg, **kw: seen.update(cfg=cfg) or lbr_result,
    )

    output = services.evaluate_run_lbr(
        tmp_path / "run-1", LBRConfig(num_hands=2000, equity_runouts=8, seed=7)
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
    # Scorer settings default to the myopic exploiter and are recorded.
    assert seen["cfg"].scorer == "myopic"
    assert output.results["scorer"] == "myopic"
    assert "lookahead_depth" not in output.results


def test_evaluate_run_lbr_threads_lookahead_scorer(monkeypatch, tmp_path):
    """scorer/lookahead knobs must reach LBRConfig and the results dict."""
    metadata = SimpleNamespace(config=MagicMock(name="config"), card_abstraction_hash="rec1")
    metadata.config.game.big_blind = 100
    storage = MagicMock(name="storage", spec=StaticArrayStorage)
    storage.num_infosets.return_value = 1
    lbr_result = SimpleNamespace(
        exploitability_mbb=1.0,
        exploitability_bb=0.001,
        std_error_mbb=0.5,
        confidence_95_mbb=(0.0, 2.0),
        lbr_utility_p0=0.0,
        lbr_utility_p1=0.0,
        num_hands=1,
        base_seed=7,
        hand_outcomes=[
            (
                HandOutcome(value=1.0, terminal="fold", pot=200),
                HandOutcome(value=-1.0, terminal="fold", pot=200),
            )
        ],
    )
    seen = {}
    monkeypatch.setattr(services_shared, "load_run_metadata", lambda run_dir: metadata)
    monkeypatch.setattr(
        services_shared,
        "build_static_evaluation_solver",
        lambda cfg, checkpoint_dir, abstraction_hash=None, at_iteration=None: (_solver(), storage),
    )
    monkeypatch.setattr(
        services_lbr,
        "compute_lbr_exploitability",
        lambda solver, cfg, **kw: seen.update(cfg=cfg) or lbr_result,
    )

    output = services.evaluate_run_lbr(
        tmp_path / "run-1",
        LBRConfig(num_hands=1, seed=7, scorer="lookahead", lookahead_depth=3, lookahead_top_k=5),
    )

    assert seen["cfg"].scorer == "lookahead"
    assert seen["cfg"].lookahead_depth == 3
    assert seen["cfg"].lookahead_top_k == 5
    assert output.results["scorer"] == "lookahead"
    assert output.results["lookahead_depth"] == 3
    assert output.results["lookahead_top_k"] == 5
