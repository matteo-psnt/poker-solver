"""Reading the Terraform coordinates must cost one subprocess, not one per caller.

``_outputs`` is `functools.cache`d, which was read as "once per process". It is
not: the cache makes the LOOKUP atomic, not the computation, so callers arriving
together on a cold cache all miss and all shell out. That stayed invisible while
every caller was a single-threaded command, and appeared the moment something
read two panels at once -- measured on `status`, `terraform output` ran twice
against `infra` and twice against `infra/store`.
"""

from __future__ import annotations

import subprocess
import threading
from typing import Any

import pytest

from src.interfaces.cloud import config as cloud_config


@pytest.fixture
def counting_terraform(monkeypatch):
    """Stand in for `terraform output -json`, counting invocations.

    The stub sleeps: without it the first call can finish before the others
    start, and the test would pass against the unlocked version too.
    """
    calls: list[str] = []
    lock = threading.Lock()

    def _run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        with lock:
            calls.append(command[1])
        threading.Event().wait(0.05)
        return subprocess.CompletedProcess(command, 0, stdout='{"pool_id": {"value": "p"}}')

    monkeypatch.setattr(cloud_config.shutil, "which", lambda _name: "/usr/bin/terraform")
    monkeypatch.setattr(cloud_config.subprocess, "run", _run)
    cloud_config._read_outputs.cache_clear()
    yield calls
    cloud_config._read_outputs.cache_clear()


def test_concurrent_readers_shell_out_once(counting_terraform):
    results: list[dict] = []

    def _read():
        results.append(cloud_config._outputs("infra"))

    threads = [threading.Thread(target=_read) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert len(counting_terraform) == 1, f"shelled out {len(counting_terraform)} times"
    assert len(results) == 6
    assert all(result == results[0] for result in results)


def test_separate_states_are_still_read_separately(counting_terraform):
    """The lock must not collapse two DIFFERENT states into one answer.

    `infra` and `infra/store` are deliberately separate so `just destroy`
    cannot reach the experiment record; sharing a cache entry would point the
    share lookups at the compute state.
    """
    cloud_config._outputs("infra")
    cloud_config._outputs("infra/store")
    assert counting_terraform == ["-chdir=infra", "-chdir=infra/store"]
