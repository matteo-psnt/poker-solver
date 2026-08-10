"""The activity log: what it records, what it must never do, and the seam.

Two properties carry the whole thing.

**It is a bystander.** A command that works must not start failing because a
disposable log could not be written. That is easy to get right today and easy
to lose to a tidy-up, since the `except` that guarantees it looks like a
swallowed error.

**Both surfaces go through one place.** `invoke` is NOT that place: the command
line parses argv and calls the handler directly, so anything wrapped around
`invoke` would observe the console and miss the CLI entirely. `execute` is one
level down, and a surface reaching past it is the regression that would make the
log quietly partial rather than absent -- the worse failure, because a partial
log still reports numbers.
"""

from __future__ import annotations

import argparse
import ast
import json

import pytest

from src.interfaces import telemetry
from src.interfaces.commands._base import Command
from src.interfaces.errors import CommandError
from src.shared import records, repo


@pytest.fixture(autouse=True)
def recording(tmp_path, monkeypatch):
    """Recording on, into a temporary log rather than the developer's cache."""
    monkeypatch.setenv(telemetry.ENV_VAR, "1")
    monkeypatch.setenv("POKER_SOLVER_CACHE", str(tmp_path))
    return tmp_path / "telemetry" / "invocations.jsonl"


def _command(handler, add_arguments=None) -> Command:
    return Command(
        name="probe",
        add_arguments=add_arguments or (lambda parser: parser.add_argument("--limit", default=0)),
        run=handler,
        render=lambda payload: None,
    )


def _rows(path) -> list[dict]:
    return records.read_log(path)


class TestItRecordsWhatRan:
    def test_a_successful_command_writes_one_row(self, recording):
        _command(lambda args: {"op": "probe"}).invoke()
        (row,) = _rows(recording)
        assert row["command"] == "probe"
        assert row["outcome"] == "ok"
        assert row["seconds"] >= 0

    def test_a_refusal_is_not_an_error(self, recording):
        """A refusal is the command answering 'no' to a question it understood.

        Filing it as an error would make a page of unpublished-run lookups read
        as an incident.
        """

        def refuse(args):
            raise CommandError("'run-x' is not published")

        with pytest.raises(CommandError):
            _command(refuse).invoke()

        (row,) = _rows(recording)
        assert row["outcome"] == "refusal"
        assert "not published" in row["error"]

    def test_a_bug_is_recorded_and_still_propagates(self, recording):
        """Observing must not swallow. A traceback is the right output for a bug."""

        def crash(args):
            raise ValueError("kaboom")

        with pytest.raises(ValueError, match="kaboom"):
            _command(crash).invoke()

        (row,) = _rows(recording)
        assert row["outcome"] == "error"
        assert row["error_type"] == "ValueError"

    def test_the_azure_ladder_is_not_duplicated_here(self, recording):
        """Its exception TYPE is recorded, not a classification of it.

        Which failures are survivable is decided once, in `errors.attempt`, and
        a guard fails if anything under `interfaces/` names those SDK types
        again. Recording the type name gives `activity` the same grouping
        without a second opinion about what it means.
        """

        class ClientAuthenticationError(Exception):
            pass

        def expired(args):
            raise ClientAuthenticationError("token expired")

        with pytest.raises(ClientAuthenticationError):
            _command(expired).invoke()

        (row,) = _rows(recording)
        assert row["error_type"] == "ClientAuthenticationError"
        assert row["outcome"] == "error"

    def test_only_the_arguments_that_differ_from_their_defaults(self, recording):
        """The rest is the same values in every row, and drowns what varies."""
        _command(lambda args: {}).invoke(limit=5)
        (row,) = _rows(recording)
        assert row["asked"] == {"limit": 5}

    def test_an_untouched_command_records_no_arguments(self, recording):
        _command(lambda args: {}).invoke()
        (row,) = _rows(recording)
        assert row["asked"] == {}

    def test_the_surface_is_recorded(self, recording):
        with telemetry.surface("console"):
            _command(lambda args: {}).invoke()
        assert _rows(recording)[0]["surface"] == "console"

    def test_an_unattributed_call_says_unknown_rather_than_guessing(self, recording):
        """Defaulting to `cli` would file the console's polling as terminal use."""
        _command(lambda args: {}).invoke()
        assert _rows(recording)[0]["surface"] == "unknown"

    def test_the_surface_is_restored_afterwards(self, recording):
        with telemetry.surface("console"):
            pass
        _command(lambda args: {}).invoke()
        assert _rows(recording)[0]["surface"] == "unknown"


class TestItIsABystander:
    """A disposable log must never be why a command fails."""

    def test_an_unwritable_log_does_not_fail_the_command(self, recording, monkeypatch):
        monkeypatch.setattr(telemetry, "log_path", lambda: recording.parent / "x" / "y")
        blocker = recording.parent
        blocker.mkdir(parents=True, exist_ok=True)
        (blocker / "x").write_text("not a directory")

        assert _command(lambda args: {"op": "probe"}).invoke() == {"op": "probe"}

    def test_a_value_that_will_not_serialise_does_not_fail_the_command(self, recording):
        """`asked` carries whatever a flag was set to, and a caller using
        `invoke` can set it to anything at all."""

        def add(parser):
            parser.add_argument("--thing", default=None)

        assert _command(lambda args: {"ok": True}, add).invoke(thing=object()) == {"ok": True}

    def test_switching_it_off_writes_nothing(self, recording, monkeypatch):
        monkeypatch.setenv(telemetry.ENV_VAR, "0")
        _command(lambda args: {}).invoke()
        assert not recording.exists()

    def test_off_is_only_the_values_that_mean_off(self, monkeypatch):
        for value in ("0", "off", "false", "NO", " off "):
            monkeypatch.setenv(telemetry.ENV_VAR, value)
            assert not telemetry.enabled(), value
        for value in ("1", "on", "", "yes"):
            monkeypatch.setenv(telemetry.ENV_VAR, value)
            assert telemetry.enabled(), value


class TestRotation:
    def test_a_large_log_is_moved_aside_rather_than_grown(self, recording, monkeypatch):
        monkeypatch.setattr(telemetry, "MAX_BYTES", 200)
        for _ in range(20):
            _command(lambda args: {}).invoke()

        assert recording.with_suffix(".jsonl.1").is_file()
        assert recording.stat().st_size < 4 * telemetry.MAX_BYTES

    def test_rotation_keeps_exactly_one_generation(self, recording, monkeypatch):
        """Two would double the disk for marginal history; zero would lose the
        minute before whatever is being investigated."""
        monkeypatch.setattr(telemetry, "MAX_BYTES", 200)
        for _ in range(60):
            _command(lambda args: {}).invoke()

        assert not recording.with_suffix(".jsonl.2").exists()


class TestTheSeamIsTheOnlyWayIn:
    """`execute`, on both surfaces. A partial log is worse than none."""

    @pytest.mark.parametrize("module", ["cli/headless.py", "web/app.py"])
    def test_a_surface_does_not_call_the_handler_directly(self, module):
        """`command.run(args)` bypasses observation entirely.

        The command line did exactly this until `execute` existed, which is why
        this is pinned rather than trusted: it is the natural thing to write and
        it silently removes the surface that runs the expensive work.
        """
        tree = ast.parse((repo.SRC / "interfaces" / module).read_text())
        direct = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "run"
        ]
        assert not direct, (
            f"{module} calls `.run(...)` directly, which skips `Command.execute` "
            "and so records nothing. Call `execute` (or `invoke`) instead."
        )

    def test_invoke_goes_through_execute(self, recording, monkeypatch):
        """Otherwise the console would be the surface that is missing."""
        seen: list[str] = []
        monkeypatch.setattr(Command, "execute", lambda self, args: seen.append(self.name) or {})

        _command(lambda args: {}).invoke()

        assert seen == ["probe"]

    def test_the_handler_stays_an_ordinary_function(self, recording):
        """`run` is still directly callable — that is what tests use, and
        wrapping it on the dataclass would take that away."""
        command = _command(lambda args: {"op": "probe"})
        assert command.run(argparse.Namespace(limit=0)) == {"op": "probe"}
        assert not recording.exists()


class TestTheRowIsReadableByTheReader:
    def test_every_row_is_one_json_object_per_line(self, recording):
        _command(lambda args: {}).invoke()
        _command(lambda args: {}).invoke(limit=2)

        lines = recording.read_text().splitlines()
        assert len(lines) == 2
        assert all(isinstance(json.loads(line), dict) for line in lines)

    def test_rows_carry_the_artifact_version(self, recording):
        """So a shape change is visible rather than inferred."""
        _command(lambda args: {}).invoke()
        assert _rows(recording)[0]["schema_version"] == records.REGISTRY[telemetry.ARTIFACT].version
