"""Reading the Terraform coordinates costs one shell-out an HOUR, not one a process.

Measured from the laptop against the remote state: `terraform output -json`
takes 3.0-3.7 s, and `tasks` paid it twice -- once for the DSN and once for
`CloudConfig` -- before touching Batch or Postgres. Seven of its fifteen
seconds were reading a dozen strings that change only at an apply.
"""

from __future__ import annotations

import json
import stat
import subprocess
from typing import Any

import pytest

from src.interfaces.cloud import config as cloud_config
from src.shared import cache


@pytest.fixture
def terraform(monkeypatch, tmp_path):
    """A counting stand-in for `terraform output -json`, and a private cache root."""
    calls: list[str] = []

    def _run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
        calls.append(command[1])
        return subprocess.CompletedProcess(
            command, 0, stdout=json.dumps({"pool_id": {"value": f"p{len(calls)}"}})
        )

    monkeypatch.setenv(cache.ENV_OVERRIDE, str(tmp_path))
    monkeypatch.setattr(cloud_config.shutil, "which", lambda _name: "/usr/bin/terraform")
    monkeypatch.setattr(cloud_config.subprocess, "run", _run)
    cloud_config._read_outputs.cache_clear()
    yield calls
    cloud_config._read_outputs.cache_clear()


def _next_process() -> None:
    """What a fresh CLI invocation starts with: no in-process memo."""
    cloud_config._read_outputs.cache_clear()


def test_the_next_process_reads_the_file_not_terraform(terraform):
    assert cloud_config._value("infra", "pool_id") == "p1"
    _next_process()
    assert cloud_config._value("infra", "pool_id") == "p1"
    assert terraform == ["-chdir=infra"]


def test_an_expired_copy_is_read_again(terraform, monkeypatch):
    cloud_config._value("infra", "pool_id")
    _next_process()
    monkeypatch.setattr(cloud_config, "OUTPUTS_TTL_SECONDS", -1.0)
    assert cloud_config._value("infra", "pool_id") == "p2"
    assert terraform == ["-chdir=infra", "-chdir=infra"]


def test_the_copy_is_owner_only(terraform, tmp_path):
    """It carries the store's access key and the record DSN."""
    cloud_config._value("infra/store", "pool_id")
    (written,) = (tmp_path / cloud_config.OUTPUTS_CACHE).glob("*.json")
    assert stat.S_IMODE(written.stat().st_mode) == 0o600


def test_a_failed_read_is_not_cached(terraform, monkeypatch):
    """An apply that has not happened yet must not become an hour of 'no outputs'."""
    monkeypatch.setattr(
        cloud_config.subprocess,
        "run",
        lambda command, **_k: subprocess.CompletedProcess(command, 0, stdout="{}"),
    )
    with pytest.raises(cloud_config.CloudConfigError):
        cloud_config._value("infra", "pool_id")
    _next_process()
    monkeypatch.setattr(cloud_config.subprocess, "run", terraform_run_ok)
    assert cloud_config._value("infra", "pool_id") == "ok"


def terraform_run_ok(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(command, 0, stdout='{"pool_id": {"value": "ok"}}')
