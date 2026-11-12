"""Admitting this machine to the record server's firewall.

The server is public with an IP allowlist, and a home ISP rotates the address:
every few days a read times out and looks like an outage. Each operator machine
owns ONE rule named after its host, so two laptops on different networks no
longer evict each other -- a shared rule did, on every failed read.
"""

from __future__ import annotations

import json
import re
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.interfaces.cloud import credential
from src.interfaces.errors import CommandError

ARM = "https://management.azure.com"
API_VERSION = "2024-08-01"
RULE_PREFIX = "allow-operator-"
# A plain-text echo of the caller's address, over TLS. Nothing else is trusted
# with the answer, and the ARM call below is what actually needs it.
WHOAMI = "https://api.ipify.org"
# How long to follow Azure's async operation. Measured: ~5 s when the server is
# idle; it serialises firewall changes, so a second one queues behind the first.
SETTLE_SECONDS = 180


@dataclass(frozen=True)
class Admission:
    server: str
    address: str
    rule: str


def public_address() -> str:
    """This machine's address as the server will see it."""
    with urllib.request.urlopen(WHOAMI, timeout=10) as response:
        address = response.read().decode().strip()
    if not address or any(not part.isdigit() for part in address.split(".")):
        raise CommandError(f"Could not learn this machine's public address: got {address!r}.")
    return address


def rule_name(host: str | None = None) -> str:
    """This machine's rule: the short hostname, in the characters Azure accepts."""
    short = (host or socket.gethostname()).split(".", 1)[0].lower()
    return RULE_PREFIX + (re.sub(r"[^a-z0-9-]+", "-", short).strip("-") or "unknown")


def admit(subscription_id: str, resource_group: str, server: str, address: str) -> Admission:
    """PUT this machine's rule so it names `address`. Idempotent."""
    rule = rule_name()
    token = credential.shared().get_token(f"{ARM}/.default").token
    url = (
        f"{ARM}/subscriptions/{subscription_id}/resourceGroups/{resource_group}"
        f"/providers/Microsoft.DBforPostgreSQL/flexibleServers/{server}"
        f"/firewallRules/{rule}?api-version={API_VERSION}"
    )
    body = json.dumps({"properties": {"startIpAddress": address, "endIpAddress": address}})
    request = urllib.request.Request(
        url,
        data=body.encode(),
        method="PUT",
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            response.read()
            operation = response.headers.get("Azure-AsyncOperation")
    except urllib.error.HTTPError as error:
        raise CommandError(
            f"Azure refused the firewall update ({error.code}): {error.read().decode()[:400]}"
        ) from error
    if operation:
        _settle(operation, token)
    return Admission(server=server, address=address, rule=rule)


def _settle(operation: str, token: str) -> None:
    """Follow the 202 to its end: an accepted PUT can still fail, and did, silently."""
    headers = {"Authorization": f"Bearer {token}"}
    deadline = time.monotonic() + SETTLE_SECONDS
    while True:
        request = urllib.request.Request(operation, headers=headers)
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read())
        status = body.get("status", "")
        if status not in ("InProgress", "Accepted"):
            break
        if time.monotonic() > deadline:
            raise CommandError(f"The firewall update was still {status} after {SETTLE_SECONDS} s.")
        time.sleep(3)
    if status != "Succeeded":
        raise CommandError(
            f"Azure accepted the firewall update, then it ended {status}: {body.get('error')}"
        )
