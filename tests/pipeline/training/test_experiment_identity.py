"""Experiment lineage on run metadata: experiment_id / arm / parent_run_id / config_hash.

These four exist so a base-fork experiment is representable. The properties that
matter: they survive a round trip, they default on every pre-experiment run, and
``config_hash`` distinguishes two runs that ``config_name`` cannot.
"""

from src.core.actions.action_model import ActionModel
from src.pipeline.training.run_tracker import ExperimentTag, RunMetadata, RunTracker
from src.shared import run_events
from src.shared.config import Config


def _action_config_hash(config: Config | None = None) -> str:
    return ActionModel(config or Config.default()).get_config_hash()


class TestConfigContentHash:
    def test_is_stable_across_equal_configs(self):
        assert Config.default().content_hash() == Config.default().content_hash()

    def test_distinguishes_configs_that_share_a_name(self):
        # The exact gap this closes: config_name comes from system.config_name
        # inside the YAML, so a run and its override-variant record the same name.
        base = Config.default()
        variant = base.merge({"storage": {"initial_capacity": 999_999}})
        assert base.system.config_name == variant.system.config_name
        assert base.content_hash() != variant.content_hash()

    def test_is_a_short_hex_digest(self):
        digest = Config.default().content_hash()
        assert len(digest) == 16
        int(digest, 16)


class TestExperimentTag:
    def test_defaults_to_empty(self):
        assert ExperimentTag().is_empty

    def test_any_field_makes_it_non_empty(self):
        assert not ExperimentTag(arm="control").is_empty
        assert not ExperimentTag(experiment_id="e1").is_empty
        assert not ExperimentTag(parent_run_id="run-a").is_empty


class TestMetadataRoundTrip:
    def test_new_run_records_the_tag_and_a_config_hash(self, tmp_path):
        config = Config.default()
        tracker = RunTracker(
            run_dir=tmp_path / "run-x",
            config_name="test",
            config=config,
            action_config_hash=_action_config_hash(config),
            experiment_id="exp-1",
            arm="variant:pruning",
            parent_run_id="run-base",
        )
        meta = tracker.metadata
        assert (meta.experiment_id, meta.arm, meta.parent_run_id) == (
            "exp-1",
            "variant:pruning",
            "run-base",
        )
        assert meta.config_hash == config.content_hash()

    def test_survives_a_write_and_fold_cycle(self, tmp_path):
        """Through the event log, which is how a run is actually persisted."""
        config = Config.default()
        run_dir = tmp_path / "run-x"
        run_dir.mkdir()
        metadata = RunMetadata.new(
            "run-x",
            "test",
            config,
            action_config_hash=_action_config_hash(config),
            experiment_id="exp-1",
            arm="control",
            parent_run_id="run-base",
        )
        run_events.append(run_dir, run_events.CREATED, **metadata.creation_facts())

        loaded = RunMetadata.load(run_dir)
        assert loaded.experiment_id == "exp-1"
        assert loaded.arm == "control"
        assert loaded.parent_run_id == "run-base"
        assert loaded.config_hash == config.content_hash()

    def test_unaffiliated_run_records_none_not_empty_string(self, tmp_path):
        config = Config.default()
        run_dir = tmp_path / "run-x"
        run_dir.mkdir()
        metadata = RunMetadata.new(
            "run-x", "test", config, action_config_hash=_action_config_hash(config)
        )
        run_events.append(run_dir, run_events.CREATED, **metadata.creation_facts())

        loaded = RunMetadata.load(run_dir)
        assert loaded.experiment_id is None
        assert loaded.arm is None
        assert loaded.parent_run_id is None


class TestLegacyRunsStillLoad:
    """Every existing run predates these fields; none may become unloadable."""

    def _legacy_dict(self, tmp_path) -> dict:
        config = Config.default()
        data = RunMetadata.new(
            "run-legacy", "test", config, action_config_hash=_action_config_hash(config)
        ).to_dict()
        for key in ("experiment_id", "arm", "parent_run_id", "config_hash"):
            del data[key]
        return data

    def test_metadata_without_the_fields_loads(self, tmp_path):
        meta = RunMetadata.from_dict(self._legacy_dict(tmp_path))
        assert meta.experiment_id is None
        assert meta.config_hash is None, "absent is not the same as 'hash of today's config'"

    def test_persisted_nulls_load_as_none(self, tmp_path):
        data = self._legacy_dict(tmp_path)
        data.update(experiment_id=None, arm=None, parent_run_id=None, config_hash=None)
        assert RunMetadata.from_dict(data).arm is None

    def test_empty_strings_normalize_to_none(self, tmp_path):
        # An unset CLI flag must not become an arm named "".
        data = self._legacy_dict(tmp_path)
        data.update(experiment_id="", arm="", parent_run_id="")
        meta = RunMetadata.from_dict(data)
        assert (meta.experiment_id, meta.arm, meta.parent_run_id) == (None, None, None)
