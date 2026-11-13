"""`doctor` reports every check and refuses, with the fixing step, when any fails."""

from __future__ import annotations

import re

import pytest

from src.interfaces.commands import doctor
from src.interfaces.errors import CommandError


def _passes() -> str:
    return "fine"


def _fails() -> str:
    raise CommandError("not signed in")


def test_all_passing_returns_every_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "CHECKS", (("a", _passes, "fix a"), ("b", _passes, "fix b")))
    payload = doctor.COMMAND.invoke_as(doctor.DoctorPayload)
    assert [(check.name, check.ok) for check in payload.checks] == [("a", True), ("b", True)]


def test_one_failure_refuses_with_the_whole_report(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "CHECKS", (("a", _passes, "fix a"), ("azure", _fails, "az login")))
    with pytest.raises(CommandError) as refused:
        doctor.COMMAND.invoke()
    report = str(refused.value)
    assert "✓ a" in report
    assert "✗ azure" in report
    assert "not signed in" in report
    assert "fix: az login" in report
    assert "fix a" not in report


def test_node_major_must_match_the_nvmrc(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(doctor, "_output", lambda *argv, **kw: "v22.11.0")
    with pytest.raises(CommandError, match=re.escape("node 22.11.0, the console wants")):
        doctor._node()


def test_the_nvmrc_names_one_major() -> None:
    assert doctor.NVMRC.read_text().strip().isdigit()
