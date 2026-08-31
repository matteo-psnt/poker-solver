"""`chipzen-seat` refuses cheaply, before a minute of checkpoint loading.

Live play cannot be tested here -- it needs an account, a token and a match. What
CAN be tested is everything the command does before it opens a socket, which is
where every mistake a user is likely to make lands: a missing token, a bad run, a
recording that is not one.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pytest

from src.interfaces.commands import chipzen_seat
from src.interfaces.errors import CommandError
from tests.test_helpers import build_trained_test_solver

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


class TestALadderSurvivesABadRung:
    """A rung that will not load must not take the seat down with it.

    This runs unattended for a six-day competition. Losing coverage at one depth
    costs mbb; refusing to start costs every fixture until a human notices.
    """

    def test_a_broken_shallow_rung_is_skipped(self, tmp_path, monkeypatch, caplog):
        from src.interfaces.commands import chipzen_seat as module

        calls: list[int] = []

        def _load(run_dir, at, threshold=0.0, *, play_only=False):
            calls.append(len(calls))
            if len(calls) == 2:
                raise RuntimeError("this rung's checkpoint is corrupt")
            return build_trained_test_solver(iterations=2, starting_stack=100 * len(calls))

        monkeypatch.setattr(module, "_build_blueprint", _load)
        monkeypatch.setattr(module, "_parse_rungs", lambda p: [(tmp_path / "b", None, 0.02)])
        payload = module.ChipzenSeatPayload(
            run="a", run_dir=str(tmp_path / "a"), runs_dir=str(tmp_path), mode="live"
        )
        with caplog.at_level(logging.ERROR, logger="src.interfaces.commands.chipzen_seat"):
            ladder = module._build_ladder(payload)
        assert [r.depth for r in ladder.rungs] == [1.0]
        assert "continues without it" in caplog.text

    def test_a_broken_deepest_rung_still_raises(self, tmp_path, monkeypatch):
        # Nothing left to play, so failing loudly into `Restart=always` is the
        # only honest outcome.
        from src.interfaces.commands import chipzen_seat as module

        def _load(*_args, **_kwargs):
            raise RuntimeError("no checkpoint")

        monkeypatch.setattr(module, "_build_blueprint", _load)
        monkeypatch.setattr(module, "_parse_rungs", lambda p: [])
        payload = module.ChipzenSeatPayload(
            run="a", run_dir=str(tmp_path / "a"), runs_dir=str(tmp_path), mode="live"
        )
        with pytest.raises(RuntimeError, match="no checkpoint"):
            module._build_ladder(payload)


class TestPerRungThresholds:
    """`run:at:threshold` — the optimum moves with depth.

    0.02 is the measured point at 100 bb, but 0.10 beat it by 38.0 +/- 15.6
    mbb/hand at 25 bb and 0.05 by 19.1 +/- 6.4 at 6 bb. One threshold for every
    rung would leave that on the table.
    """

    def _payload(self, tmp_path, rungs, threshold=0.02):
        # `resolve_run_dir` insists the directory exists, which is the point of
        # it -- a typo'd rung should fail here rather than at load.
        for spec in rungs:
            (tmp_path / spec.partition(":")[0]).mkdir(exist_ok=True)
        return chipzen_seat.ChipzenSeatPayload(
            run="a",
            run_dir=str(tmp_path / "a"),
            runs_dir=str(tmp_path),
            mode="live",
            rungs=rungs,
            policy_threshold=threshold,
        )

    def test_a_rung_can_carry_its_own_threshold(self, tmp_path):
        parsed = chipzen_seat._parse_rungs(self._payload(tmp_path, ["b:4000:0.1"]))
        assert parsed[0][1] == 4000
        assert parsed[0][2] == 0.1

    def test_a_rung_without_one_inherits_the_default(self, tmp_path):
        parsed = chipzen_seat._parse_rungs(self._payload(tmp_path, ["b:4000"]))
        assert parsed[0][2] == 0.02

    def test_a_bare_rung_name_still_works(self, tmp_path):
        parsed = chipzen_seat._parse_rungs(self._payload(tmp_path, ["b"]))
        assert parsed[0][1] is None
        assert parsed[0][2] == 0.02

    def test_each_rung_is_loaded_with_its_own(self, tmp_path, monkeypatch):
        seen: list[float] = []

        def _load(run_dir, at, threshold=0.0, *, play_only=False):
            seen.append(threshold)
            return build_trained_test_solver(iterations=2, starting_stack=100 * len(seen))

        monkeypatch.setattr(chipzen_seat, "_build_blueprint", _load)
        chipzen_seat._build_ladder(self._payload(tmp_path, ["b:4000:0.1", "c:4000:0.05"]))
        # The deepest takes `--policy-threshold`; each rung takes its own.
        assert seen == [0.02, 0.1, 0.05]
