"""Level resolution and the spawn-boundary contract in ``src.shared.log``."""

import logging
import multiprocessing as mp
import pathlib
import re
import sys

import pytest

from src.shared import log as log_module
from src.shared.log import (
    LEVEL_ENV_VAR,
    configure_logging,
    pin_level_for_children,
    progress_bars_enabled,
    resolve_level,
)


class _FakeStream:
    """Minimal stand-in for a stream whose tty-ness the test decides."""

    def __init__(self, *, isatty: bool) -> None:
        self._isatty = isatty

    def isatty(self) -> bool:
        return self._isatty


class _CapturingStream(_FakeStream):
    """A fake stderr that also keeps what was written to it."""

    def __init__(self, *, isatty: bool) -> None:
        super().__init__(isatty=isatty)
        self._chunks: list[str] = []

    def write(self, chunk: str) -> int:
        self._chunks.append(chunk)
        return len(chunk)

    def flush(self) -> None:
        return None

    def text(self) -> str:
        return "".join(self._chunks)


@pytest.fixture(autouse=True)
def _restore_package_logger(monkeypatch):
    """Leave the ``src`` logger exactly as found.

    The whole suite shares one logger; a test that left it at ERROR would make
    unrelated caplog assertions fail in a way that reads as a code defect.
    """
    monkeypatch.delenv(LEVEL_ENV_VAR, raising=False)
    package_logger = logging.getLogger("src")
    level, handlers, propagate = (
        package_logger.level,
        list(package_logger.handlers),
        package_logger.propagate,
    )
    yield
    package_logger.setLevel(level)
    package_logger.handlers[:] = handlers
    package_logger.propagate = propagate


class TestResolveLevel:
    def test_defaults_to_info(self):
        assert resolve_level() == logging.INFO

    def test_accepts_a_name(self):
        assert resolve_level("DEBUG") == logging.DEBUG

    def test_name_is_case_insensitive(self):
        assert resolve_level("debug") == logging.DEBUG

    def test_passes_a_numeric_level_through(self):
        assert resolve_level(logging.ERROR) == logging.ERROR

    def test_environment_outranks_the_argument(self, monkeypatch):
        monkeypatch.setenv(LEVEL_ENV_VAR, "ERROR")
        assert resolve_level("DEBUG") == logging.ERROR

    def test_unknown_name_raises(self):
        with pytest.raises(ValueError, match="unknown log level"):
            resolve_level("DEGUB")

    def test_unknown_environment_value_raises(self, monkeypatch):
        """A typo'd override must fail loudly, not silently keep the default."""
        monkeypatch.setenv(LEVEL_ENV_VAR, "DEGUB")
        with pytest.raises(ValueError, match="unknown log level"):
            resolve_level("INFO")


class TestConfigureLogging:
    def test_sets_the_requested_level(self):
        configure_logging("WARNING")
        assert logging.getLogger("src").level == logging.WARNING

    def test_is_idempotent_in_handler_count(self):
        configure_logging()
        configure_logging()
        configure_logging()
        assert len(logging.getLogger("src").handlers) == 1

    def test_repeat_calls_still_adjust_the_level(self):
        configure_logging("ERROR")
        configure_logging("DEBUG")
        assert logging.getLogger("src").level == logging.DEBUG

    def test_environment_outranks_a_config_supplied_level(self, monkeypatch):
        """A --log-level pin must beat config.system.log_level in every process."""
        monkeypatch.setenv(LEVEL_ENV_VAR, "DEBUG")
        configure_logging("ERROR")
        assert logging.getLogger("src").level == logging.DEBUG

    def test_does_not_propagate_to_root(self):
        configure_logging()
        assert logging.getLogger("src").propagate is False


class TestPinLevelForChildren:
    def test_publishes_to_the_environment(self, monkeypatch):
        monkeypatch.setenv(LEVEL_ENV_VAR, "INFO")
        pin_level_for_children("DEBUG")
        assert resolve_level() == logging.DEBUG

    def test_normalizes_case(self, monkeypatch):
        monkeypatch.setenv(LEVEL_ENV_VAR, "INFO")
        pin_level_for_children("debug")
        assert resolve_level() == logging.DEBUG

    def test_rejects_a_typo_before_publishing(self, monkeypatch):
        monkeypatch.setenv(LEVEL_ENV_VAR, "INFO")
        with pytest.raises(ValueError, match="unknown log level"):
            pin_level_for_children("DEGUB")
        assert resolve_level() == logging.INFO


def _child_effective_level(queue: mp.Queue, configure: bool) -> None:
    """Report what a spawned child would actually emit at.

    Mirrors a real worker entrypoint: fresh interpreter, then optionally the
    one call every worker is required to make.
    """
    if configure:
        configure_logging()
    package_logger = logging.getLogger("src")
    queue.put(
        {
            "handlers": len(package_logger.handlers),
            "info_enabled": package_logger.isEnabledFor(logging.INFO),
        }
    )


def _run_child(configure: bool) -> dict:
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    process = ctx.Process(target=_child_effective_level, args=(queue, configure))
    process.start()
    result = queue.get(timeout=20)
    process.join(timeout=20)
    return result


@pytest.mark.timeout(60)
class TestSpawnBoundary:
    """The defect these guard: spawn children inherit no handler.

    An unconfigured child falls through to ``logging.lastResort`` at WARNING,
    so every ``logger.info`` it makes disappears without a trace.
    """

    def test_unconfigured_child_drops_info(self):
        assert _run_child(configure=False) == {"handlers": 0, "info_enabled": False}

    def test_configured_child_keeps_info(self):
        assert _run_child(configure=True) == {"handlers": 1, "info_enabled": True}


class TestAdaptiveFormat:
    """One stream, two readers: a human watching, and a log being grepped later."""

    def _emit(self, monkeypatch, *, isatty, level=logging.INFO, msg="checkpointed", exc=None):
        stream = _CapturingStream(isatty=isatty)
        monkeypatch.setattr(sys, "stderr", stream)
        configure_logging()
        logging.getLogger("src.pipeline.training.static_parallel").log(level, msg, exc_info=exc)
        return stream.text()

    def test_terminal_output_is_the_bare_message(self, monkeypatch):
        assert self._emit(monkeypatch, isatty=True).strip() == "checkpointed"

    def test_captured_output_carries_level_and_logger(self, monkeypatch):
        line = self._emit(monkeypatch, isatty=False, level=logging.WARNING)
        assert "WARN" in line
        assert "pipeline.training.static_parallel" in line
        assert "checkpointed" in line

    def test_captured_output_is_greppable_by_severity(self, monkeypatch):
        """The whole point: `just leg-log <task> errors` must be able to work."""
        info = self._emit(monkeypatch, isatty=False, level=logging.INFO)
        error = self._emit(monkeypatch, isatty=False, level=logging.ERROR)

        pattern = re.compile(r" (WARN|ERROR|CRIT) ")
        assert pattern.search(error)
        assert not pattern.search(info), "an INFO line must not match the error filter"

    def test_long_level_names_are_shortened_to_stay_aligned(self, monkeypatch):
        line = self._emit(monkeypatch, isatty=False, level=logging.CRITICAL)
        assert " CRIT " in line
        assert "CRITICAL" not in line

    def test_timestamp_is_utc_and_sortable(self, monkeypatch):
        line = self._emit(monkeypatch, isatty=False)
        assert re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z ", line)

    def test_src_prefix_is_stripped_from_the_logger_name(self, monkeypatch):
        line = self._emit(monkeypatch, isatty=False)
        assert "src.pipeline" not in line
        assert "pipeline.training.static_parallel" in line

    def test_tracebacks_survive_in_captured_form(self, monkeypatch):
        try:
            raise RuntimeError("blosc decompression failed")
        except RuntimeError:
            line = self._emit(monkeypatch, isatty=False, level=logging.ERROR, exc=True)
        assert "Traceback" in line
        assert "blosc decompression failed" in line

    def test_the_record_is_not_mutated_for_other_handlers(self, monkeypatch):
        """The formatter edits name/levelname in place, so it must restore them."""
        record = logging.LogRecord(
            "src.pipeline.demo", logging.WARNING, __file__, 1, "msg", None, None
        )
        monkeypatch.setattr(sys, "stderr", _CapturingStream(isatty=False))

        log_module._AdaptiveFormatter().format(record)

        assert record.name == "src.pipeline.demo"
        assert record.levelname == "WARNING"

    def test_format_follows_a_stream_swapped_after_configuration(self, monkeypatch):
        """A worker configures while interactive, then has stderr redirected."""
        configure_logging()
        interactive = self._emit(monkeypatch, isatty=True)
        captured = self._emit(monkeypatch, isatty=False)

        assert interactive.strip() == "checkpointed"
        assert "INFO" in captured


class TestNoHandWrittenSeverityPrefixes:
    def test_messages_do_not_restate_their_own_level(self):
        """They only existed because the old formatter dropped the level.

        ``logger.warning("Warning: ...")`` is the missing structure said out
        loud; with the level in the line it is duplication that also breaks a
        severity grep by matching INFO lines that merely mention "Error".
        """
        repo_src = pathlib.Path(__file__).resolve().parents[2] / "src"
        offenders = []
        pattern = re.compile(r"logger\.(info|warning|error)\(f?\"[^\"]*(WARN|Warning|ERROR|Error)")
        for path in repo_src.rglob("*.py"):
            for number, line in enumerate(path.read_text().splitlines(), 1):
                if pattern.search(line):
                    offenders.append(f"{path.name}:{number}")
        assert not offenders, f"messages restating their own severity: {offenders}"


class TestProgressBarsEnabled:
    """Driven off an explicit fake stderr, never the ambient one.

    Asserting on pytest's own stderr would make the result depend on whether
    the suite was invoked with ``-s``.
    """

    def test_enabled_on_a_terminal(self, monkeypatch):
        monkeypatch.setattr(sys, "stderr", _FakeStream(isatty=True))
        assert progress_bars_enabled() is True

    def test_disabled_when_redirected(self, monkeypatch):
        monkeypatch.setattr(sys, "stderr", _FakeStream(isatty=False))
        assert progress_bars_enabled() is False


class TestEveryProgressBarIsGuarded:
    """Re-homed from the deleted test_memory_telemetry.py.

    A bar that forgets to opt out is the cloud-log failure again: a multi-hour
    leg once wrote a stderr file too large to download.
    """

    def test_every_tqdm_site_consults_the_predicate(self):
        src = pathlib.Path(__file__).resolve().parents[2] / "src"
        for path in src.rglob("*.py"):
            text = path.read_text()
            bars = text.count("tqdm(")
            if not bars:
                continue
            # COUNT, do not merely look for one mention: a file-granular check
            # passes as soon as any bar in the file opts out, so a second bar
            # added beside an existing one would slip through.
            guarded = text.count("progress_bars_enabled()")
            assert guarded >= bars, (
                f"{path.name} has {bars} tqdm call site(s) but only {guarded} guard(s)"
            )
