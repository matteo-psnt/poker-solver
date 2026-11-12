"""Two operator laptops on different networks must not evict each other.

Measured 11-11: with one shared rule, a failed read on the Air re-admitted the
Air and locked out the Pro, and the Pro's next read swapped it back. The same
day a 202-accepted PUT for the second rule ended without creating it, silently.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from src.interfaces.cloud import record_firewall
from src.interfaces.errors import CommandError

OPERATION = "https://management.azure.com/operations/op1"


class _Response:
    def __init__(self, body: dict[str, Any], headers: dict[str, str] | None = None) -> None:
        self._body = json.dumps(body).encode()
        self.headers = headers or {}

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _fake_azure(monkeypatch: pytest.MonkeyPatch, final: str) -> list[str]:
    """PUTs answer 202 with an operation that is InProgress once, then `final`."""
    calls: list[str] = []
    polls = iter(["InProgress", final])

    def _urlopen(request: Any, timeout: float) -> _Response:
        calls.append(request.full_url)
        if request.full_url == OPERATION:
            return _Response({"status": next(polls), "error": {"code": "ServerIsBusy"}})
        return _Response({}, {"Azure-AsyncOperation": OPERATION})

    monkeypatch.setattr(
        record_firewall.credential,
        "shared",
        lambda: SimpleNamespace(get_token=lambda scope: SimpleNamespace(token="t")),
    )
    monkeypatch.setattr(record_firewall.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(record_firewall.time, "sleep", lambda seconds: None)
    return calls


@pytest.mark.parametrize(
    ("host", "rule"),
    [
        ("Matteos-MacBook-Pro.local", "allow-operator-matteos-macbook-pro"),
        ("Matteos-MacBook-Air", "allow-operator-matteos-macbook-air"),
        ("odd_name.lan", "allow-operator-odd-name"),
        ("...", "allow-operator-unknown"),
    ],
)
def test_rule_name_is_the_short_host_in_azure_characters(host: str, rule: str) -> None:
    assert record_firewall.rule_name(host) == rule


def test_two_hosts_put_two_different_rules(monkeypatch: pytest.MonkeyPatch) -> None:
    rules = []
    for host in ("Matteos-MacBook-Pro", "Matteos-MacBook-Air"):
        calls = _fake_azure(monkeypatch, "Succeeded")
        monkeypatch.setattr(record_firewall.socket, "gethostname", lambda h=host: h)
        rules.append(record_firewall.admit("sub", "rg", "srv", "1.2.3.4").rule)
        assert calls[0].split("/firewallRules/")[1].split("?")[0] == rules[-1]
    assert rules[0] != rules[1]


def test_an_accepted_put_that_then_fails_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _fake_azure(monkeypatch, "Failed")
    with pytest.raises(CommandError, match="ended Failed"):
        record_firewall.admit("sub", "rg", "srv", "1.2.3.4")
