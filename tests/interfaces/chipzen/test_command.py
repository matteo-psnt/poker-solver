"""`chipzen-seat` refuses cheaply, before a minute of checkpoint loading.

Live play cannot be tested here -- it needs an account, a token and a match. What
CAN be tested is everything the command does before it opens a socket, which is
where every mistake a user is likely to make lands: a missing token, a bad run, a
recording that is not one.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from src.interfaces.commands import chipzen_seat
from src.interfaces.errors import CommandError

FIXTURE = Path(__file__).parent / "fixtures" / "turn_preflop_sb.json"


def parsed(**overrides) -> argparse.Namespace:
    """The namespace the CLI would build, with the flags a test cares about."""
    parser = argparse.ArgumentParser()
    chipzen_seat.add_arguments(parser)
    defaults = {"run": "whatever", "runs_dir": "/nonexistent"}
    defaults.update(overrides)
    return parser.parse_args(
        [
            item
            for key, value in defaults.items()
            for item in (f"--{key.replace('_', '-')}", str(value))
        ]
    )


class TestTheRecordedTurnFixture:
    def test_it_is_the_shape_the_command_documents(self):
        recorded = json.loads(FIXTURE.read_text())
        assert set(recorded) >= {"game_config", "state", "seat"}
        assert recorded["game_config"]["num_players"] == 2

    def test_it_parses_as_the_protocol_types(self):
        from src.interfaces.chipzen.protocol import GameConfig, TurnState

        recorded = json.loads(FIXTURE.read_text())
        config = GameConfig.parse(recorded["game_config"])
        turn = TurnState.parse(recorded["state"])
        assert config.depth_in_blinds == 100.0
        assert turn.button_seat() == 0
        assert turn.round_wager(1) == 0  # the flop is a fresh betting round


@pytest.fixture
def unconfigured(monkeypatch):
    """A machine with no `chipzen.toml` anywhere on the SDK's search path.

    Without this the refusal tests pass only on a machine that has never been
    set up: `sdk_config_path` reads the real `~/.chipzen/`, so writing a genuine
    credential file there turned both of them red. A test that depends on the
    developer's home directory is a test that reports the wrong thing twice.
    """
    monkeypatch.setattr(chipzen_seat, "CONFIG_SEARCH", ())


@pytest.fixture
def configured(monkeypatch, tmp_path):
    """A machine whose `chipzen.toml` carries both credentials."""
    config = tmp_path / "chipzen.toml"
    config.write_text('[external_api]\ntoken = "cz_extbot_x"\nbot_id = "b"\n')
    monkeypatch.setattr(chipzen_seat, "CONFIG_SEARCH", (config,))


class TestRefusals:
    def test_a_missing_run_refuses_before_anything_loads(self, tmp_path):
        args = parsed(run="nope", runs_dir=str(tmp_path))
        with pytest.raises(CommandError, match="Run not found"):
            chipzen_seat.run(args)

    def test_a_missing_bot_id_refuses(self, tmp_path, monkeypatch, unconfigured):
        monkeypatch.delenv(chipzen_seat.BOT_ENV, raising=False)
        (tmp_path / "run-1").mkdir()
        args = parsed(run="run-1", runs_dir=str(tmp_path))
        with pytest.raises(CommandError, match="--bot-id"):
            chipzen_seat.run(args)

    def test_a_missing_token_refuses(self, tmp_path, monkeypatch, unconfigured):
        monkeypatch.setenv(chipzen_seat.BOT_ENV, "a-bot")
        monkeypatch.delenv(chipzen_seat.TOKEN_ENV, raising=False)
        (tmp_path / "run-1").mkdir()
        args = parsed(run="run-1", runs_dir=str(tmp_path))
        with pytest.raises(CommandError, match="cz_extbot_"):
            chipzen_seat.run(args)

    def test_a_config_file_stands_in_for_both_env_vars(self, tmp_path, monkeypatch, configured):
        """The setup we actually ship: credentials in the SDK's own toml."""
        monkeypatch.delenv(chipzen_seat.BOT_ENV, raising=False)
        monkeypatch.delenv(chipzen_seat.TOKEN_ENV, raising=False)
        (tmp_path / "run-1").mkdir()
        args = parsed(run="run-1", runs_dir=str(tmp_path))
        payload = chipzen_seat.run(args)
        assert payload.mode == "live"
        # Left as None on purpose: the SDK reads it from the file it just found.
        assert payload.bot_id is None

    def test_the_default_environment_is_where_the_division_lives(self):
        """Measured: a token valid on prod is rejected by staging."""
        assert chipzen_seat.DEFAULT_ENV == "prod"

    def test_the_token_is_never_a_flag(self):
        """It is shown once and cannot be read back; a shell history is the wrong place."""
        parser = argparse.ArgumentParser()
        chipzen_seat.add_arguments(parser)
        assert not any("token" in action.dest for action in parser._actions)


class TestLoadingARecording:
    def test_a_missing_file_refuses(self, tmp_path):
        with pytest.raises(CommandError, match="No such file"):
            chipzen_seat._load_recording(tmp_path / "absent.json")

    def test_a_non_json_file_refuses(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json")
        with pytest.raises(CommandError, match="is not JSON"):
            chipzen_seat._load_recording(path)

    def test_a_recording_without_a_state_refuses(self, tmp_path):
        path = tmp_path / "partial.json"
        path.write_text(json.dumps({"game_config": {}}))
        with pytest.raises(CommandError, match="'state'"):
            chipzen_seat._load_recording(path)

    def test_the_shipped_fixture_loads(self):
        assert chipzen_seat._load_recording(FIXTURE)["seat"] == 1
