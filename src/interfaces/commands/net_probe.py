"""The `net-probe` subcommand: what a node can actually reach, measured from one.

Every part of moving the run record into a database rests on two unverified
assumptions -- that a node may open outbound TCP on 5432, and that it can prove
an identity to something that is not an Azure SDK endpoint. Nothing in this repo
has ever opened a non-SDK connection from a node, so both are guesses until a
node answers.

Stdlib only, deliberately: the same code has to be movable into
``shared/cloudtask/`` if the answer is that the wrapper should ship events.
"""

from __future__ import annotations

import json
import socket
import ssl
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, Field

from src.interfaces.commands._base import Command

if TYPE_CHECKING:
    import argparse

# Azure's link-local metadata endpoint. A SHORT timeout is not politeness: the
# default credential chain probes this address and, off a node, hangs rather
# than refusing -- measured at >120s. A probe that hangs answers nothing.
IMDS = "169.254.169.254"
IMDS_TIMEOUT = 5.0

# The AAD audience an Azure Database for PostgreSQL server accepts. Asking for a
# token in this scope is the real managed-identity question; asking for the ARM
# scope would pass on a node that still cannot reach a database.
POSTGRES_SCOPE = "https://ossrdbms-aad.database.windows.net"

# Accepts a TCP connection on ANY port, which is what makes "is 5432 filtered?"
# separable from "is anything listening?". No payload is ever sent.
PORT_ORACLE = "portquiz.net"

CONNECT_TIMEOUT = 8.0


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver net-probe`."""
    parser.add_argument(
        "--port",
        type=int,
        action="append",
        dest="ports",
        default=None,
        help="TCP port to test outbound against the port oracle; repeatable (default 443, 5432).",
    )
    parser.add_argument(
        "--oracle",
        default=PORT_ORACLE,
        help=f"Host that accepts a connection on any port (default {PORT_ORACLE}).",
    )
    parser.add_argument(
        "--host",
        default=None,
        help="A REAL endpoint to connect to, host:port. Use once a server exists "
        "to re-run this same probe end to end.",
    )
    parser.add_argument(
        "--tls",
        action="store_true",
        help="Complete a TLS handshake against --host rather than stopping at TCP.",
    )


class Check(BaseModel):
    """One question the node answered, and how long it took to answer it."""

    name: str
    # "open" | "refused" | "blocked" | "dns" | "error" -- `refused` and `blocked`
    # are the pair that matters: a refusal proves egress reached something, a
    # timeout means a filter ate it, and collapsing them loses the finding.
    outcome: str
    detail: str = ""
    ms: float = 0.0


class ProbePayload(BaseModel):
    """What this node can reach, from this node."""

    op: Literal["net-probe"] = "net-probe"
    hostname: str = ""
    region: str = ""
    vm_size: str = ""
    egress_ip: str = ""
    identity: str = ""
    checks: list[Check] = Field(default_factory=list)


def _timed(name: str, work: Any) -> Check:
    """Run one check, charging it its own wall clock and never raising."""
    started = time.monotonic()
    try:
        outcome, detail = work()
    except Exception as exc:  # noqa: BLE001 -- a probe reports failures, it does not raise them
        outcome, detail = "error", f"{type(exc).__name__}: {exc}"
    return Check(name=name, outcome=outcome, detail=detail, ms=(time.monotonic() - started) * 1000)


def _imds(path: str, timeout: float = IMDS_TIMEOUT) -> dict[str, Any]:
    """Read one IMDS document. The `Metadata` header is what makes it answer."""
    request = urllib.request.Request(f"http://{IMDS}{path}", headers={"Metadata": "true"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def _connect(host: str, port: int, *, tls: bool = False) -> tuple[str, str]:
    """Classify what happens on the way to one TCP endpoint.

    The three outcomes are the whole point: `refused` proves egress reached the
    host, `blocked` means a filter dropped it, and only `open` means we could
    speak. A caller that treats a timeout as a refusal will report a firewall as
    an empty server.
    """
    try:
        # Resolved separately from the connect so the report can name the ADDRESS
        # a firewall rule would have to allow, which the hostname does not give.
        resolved = str(socket.getaddrinfo(host, port, proto=socket.IPPROTO_TCP)[0][4][0])
    except socket.gaierror as exc:
        return "dns", f"{host} did not resolve: {exc}"
    try:
        with socket.create_connection((host, port), timeout=CONNECT_TIMEOUT) as sock:
            if not tls:
                return "open", f"tcp to {resolved}:{port}"
            context = ssl.create_default_context()
            with context.wrap_socket(sock, server_hostname=host) as tunnel:
                return "open", f"tls {tunnel.version()} to {resolved}:{port}"
    except ConnectionRefusedError:
        return "refused", f"{resolved}:{port} refused -- egress reached it, nothing listening"
    except TimeoutError:
        return "blocked", f"{resolved}:{port} timed out after {CONNECT_TIMEOUT:.0f}s -- filtered"
    except ssl.SSLError as exc:
        return "open", f"tcp open, TLS refused ({exc.reason or exc})"


def _token_check() -> tuple[str, str]:
    """Whether this node can mint an AAD token a Postgres server would accept."""
    try:
        document = _imds(
            f"/metadata/identity/oauth2/token?api-version=2018-02-01&resource={POSTGRES_SCOPE}"
        )
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()[:200]
        return "error", f"HTTP {exc.code}: {body}"
    token = str(document.get("access_token", ""))
    return (
        ("open", f"token issued, {len(token)} chars") if token else ("error", str(document)[:200])
    )


def run(args: argparse.Namespace) -> ProbePayload:
    """Ask the node what it can reach, and report every answer including the failures."""
    payload = ProbePayload(hostname=socket.gethostname())

    instance = _timed("imds-instance", lambda: _summarise_instance(payload))
    payload.checks.append(instance)
    payload.checks.append(_timed(f"aad-token[{POSTGRES_SCOPE}]", _token_check))

    for port in args.ports or (443, 5432):
        payload.checks.append(
            _timed(
                f"tcp[{args.oracle}:{port}]",
                lambda host=args.oracle, p=port: _connect(host, p),
            )
        )

    if args.host:
        host, _, raw = args.host.partition(":")
        port = int(raw or 5432)
        payload.checks.append(
            _timed(
                f"tcp[{host}:{port}]",
                lambda: _connect(host, port, tls=args.tls),
            )
        )
    return payload


def _summarise_instance(payload: ProbePayload) -> tuple[str, str]:
    """Fill the node's own identity onto the payload, and say whether IMDS answered."""
    document = _imds("/metadata/instance?api-version=2021-02-01")
    compute = document.get("compute", {})
    payload.region = str(compute.get("location", ""))
    payload.vm_size = str(compute.get("vmSize", ""))
    payload.identity = str(compute.get("identity", "")) or "none declared"
    interfaces = document.get("network", {}).get("interface", [])
    if interfaces:
        addresses = interfaces[0].get("ipv4", {}).get("ipAddress", [{}])
        payload.egress_ip = str(addresses[0].get("publicIpAddress", "")) or "no public ip"
    return "open", f"{payload.vm_size} in {payload.region}"


def render(payload: ProbePayload) -> None:
    print(f"node {payload.hostname}  {payload.vm_size} {payload.region}".rstrip())
    if payload.egress_ip:
        print(f"egress: {payload.egress_ip}")
    print()
    width = max((len(check.name) for check in payload.checks), default=0)
    for check in payload.checks:
        print(f"  {check.outcome:<8} {check.name:<{width}}  {check.ms:7.0f} ms  {check.detail}")


COMMAND = Command(
    name="net-probe",
    add_arguments=add_arguments,
    run=run,
    render=render,
    help="Report what a node can reach outbound: ports, IMDS, and an AAD token for Postgres.",
)
