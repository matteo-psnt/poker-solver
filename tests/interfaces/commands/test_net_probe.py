"""What the probe must never do: raise, or collapse two different failures into one.

A connectivity answer is only worth dispatching if `refused` and `blocked` stay
apart -- one says egress reached the host, the other says a filter ate it. Every
test here pins that distinction or the no-raise property that surrounds it.
"""

from __future__ import annotations

import argparse
import socket
from typing import Any

import pytest

from src.interfaces.commands import net_probe


def _args(**overrides: Any) -> argparse.Namespace:
    base = {"ports": None, "oracle": "oracle.invalid", "host": None, "tls": False}
    return argparse.Namespace(**(base | overrides))


class TestOutcomesStayApart:
    """`refused` and `blocked` are different findings and must not merge."""

    def test_a_refusal_reports_egress_reached_the_host(self, monkeypatch):
        monkeypatch.setattr(net_probe.socket, "getaddrinfo", _resolves_to("10.0.0.1"))
        monkeypatch.setattr(
            net_probe.socket, "create_connection", _raises(ConnectionRefusedError())
        )
        outcome, detail = net_probe._connect("db.example", 5432)
        assert outcome == "refused"
        assert "nothing listening" in detail

    def test_a_timeout_reports_a_filter_not_an_empty_server(self, monkeypatch):
        monkeypatch.setattr(net_probe.socket, "getaddrinfo", _resolves_to("10.0.0.1"))
        monkeypatch.setattr(net_probe.socket, "create_connection", _raises(TimeoutError()))
        outcome, detail = net_probe._connect("db.example", 5432)
        assert outcome == "blocked"
        assert "filtered" in detail

    def test_an_unresolvable_name_is_dns_not_blocked(self, monkeypatch):
        monkeypatch.setattr(net_probe.socket, "getaddrinfo", _raises(socket.gaierror("nope")))
        outcome, _ = net_probe._connect("db.example", 5432)
        assert outcome == "dns"


class TestItReportsFailuresRatherThanRaising:
    """A probe that raises tells you nothing about the other checks."""

    def test_a_check_that_explodes_becomes_an_error_row(self):
        check = net_probe._timed("boom", _raises(RuntimeError("detonated")))
        assert check.outcome == "error"
        assert "detonated" in check.detail

    def test_a_run_with_every_network_call_failing_still_returns_a_payload(self, monkeypatch):
        monkeypatch.setattr(net_probe, "_imds", _raises(OSError("no imds here")))
        monkeypatch.setattr(net_probe.socket, "getaddrinfo", _raises(socket.gaierror("nope")))
        payload = net_probe.run(_args())
        assert [check.outcome for check in payload.checks] == ["error", "error", "dns", "dns"]

    def test_the_renderer_survives_a_payload_of_pure_failure(self, monkeypatch, capsys):
        monkeypatch.setattr(net_probe, "_imds", _raises(OSError("no imds here")))
        monkeypatch.setattr(net_probe.socket, "getaddrinfo", _raises(socket.gaierror("nope")))
        net_probe.render(net_probe.run(_args()))
        assert "error" in capsys.readouterr().out


class TestTheCommandLineReachesTheChecks:
    """A flag that is accepted and then ignored is worse than one that is refused."""

    def test_repeated_ports_each_become_a_check(self, monkeypatch):
        monkeypatch.setattr(net_probe, "_imds", _raises(OSError("no imds")))
        monkeypatch.setattr(net_probe.socket, "getaddrinfo", _raises(socket.gaierror("nope")))
        payload = net_probe.run(_args(ports=[5432, 6432, 443]))
        names = [check.name for check in payload.checks]
        assert "tcp[oracle.invalid:5432]" in names
        assert "tcp[oracle.invalid:6432]" in names
        assert "tcp[oracle.invalid:443]" in names

    @pytest.mark.parametrize(
        ("host", "expected"),
        [("db.example:5432", "tcp[db.example:5432]"), ("db.example", "tcp[db.example:5432]")],
    )
    def test_host_defaults_to_the_postgres_port(self, monkeypatch, host, expected):
        """`--host db.example` must not silently probe port 0."""
        monkeypatch.setattr(net_probe, "_imds", _raises(OSError("no imds")))
        monkeypatch.setattr(net_probe.socket, "getaddrinfo", _raises(socket.gaierror("nope")))
        payload = net_probe.run(_args(host=host))
        assert expected in [check.name for check in payload.checks]


def _raises(error: BaseException):
    def boom(*_args: object, **_kwargs: object):
        raise error

    return boom


def _resolves_to(address: str):
    def resolve(*_args: object, **_kwargs: object):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", (address, 5432))]

    return resolve
