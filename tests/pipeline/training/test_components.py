"""Tests for training component builders."""

import json

import pytest

from src.pipeline.abstraction.config import PrecomputeConfig
from src.pipeline.training import components
from src.shared.config import Config
from tests.test_helpers import DummyCardAbstraction


class TestBuildCardAbstraction:
    """Tests for build_card_abstraction."""

    def test_build_fails_with_invalid_config_name(self):
        """Test that building fails when config has no matching abstraction."""
        config = Config.default().merge({"card_abstraction": {"config": "nonexistent_config_xyz"}})

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            components.build_card_abstraction(config)

    def test_build_loads_unique_hash_match(self, tmp_path, monkeypatch):
        """Build uses the unique abstraction path matching the expected config hash."""
        expected_hash = PrecomputeConfig.from_yaml("default").get_config_hash()
        base_path = tmp_path / "data" / "combo_abstraction"
        candidate = base_path / "default-a"
        candidate.mkdir(parents=True)
        with (candidate / "metadata.json").open("w") as f:
            json.dump(
                {
                    "config_hash": expected_hash,
                    "config": {
                        "config_name": "default",
                    },
                },
                f,
            )

        loaded_path = None

        def _mock_load(path):
            nonlocal loaded_path
            loaded_path = path
            return DummyCardAbstraction()

        monkeypatch.setattr(components.PostflopPrecomputer, "load", _mock_load)
        config = Config.default().merge({"card_abstraction": {"config": "default"}})

        abstraction = components.build_card_abstraction(
            config,
            abstractions_dir=base_path,
        )

        assert isinstance(abstraction, DummyCardAbstraction)
        assert loaded_path == candidate

    def test_build_fails_when_multiple_hash_matches(self, tmp_path):
        """Multiple matching abstractions should fail and ask for explicit path."""
        expected_hash = PrecomputeConfig.from_yaml("default").get_config_hash()
        base_path = tmp_path / "data" / "combo_abstraction"
        for name in ["default-a", "default-b"]:
            candidate = base_path / name
            candidate.mkdir(parents=True)
            with (candidate / "metadata.json").open("w") as f:
                json.dump(
                    {
                        "config_hash": expected_hash,
                        "config": {
                            "config_name": "default",
                        },
                    },
                    f,
                )

        config = Config.default().merge({"card_abstraction": {"config": "default"}})
        with pytest.raises(ValueError, match="Multiple combo abstractions found"):
            components.build_card_abstraction(
                config,
                abstractions_dir=base_path,
            )
