"""Admitting this machine to the record server's firewall.

The server is public with an IP allowlist, and a home ISP rotates the address:
every few days a read times out and looks like an outage. Terraform holds the
rule so it exists; this updates it in place, and the rule's `ignore_changes`
keeps the next apply from putting the old address back.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from src.interfaces.cloud import credential
from src.interfaces.errors import CommandError

ARM = "https://management.azure.com"
API_VERSION = "2024-08-01"
RULE_NAME = "allow-operator"
# A plain-text echo of the caller's address, over TLS. Nothing else is trusted
# with the answer, and the ARM call below is what actually needs it.
WHOAMI = "https://api.ipify.org"


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


def admit(subscription_id: str, resource_group: str, server: str, address: str) -> Admission:
    """PUT the operator rule so it names `address`. Idempotent."""
    token = credential.shared().get_token(f"{ARM}/.default").token
    url = (
        f"{ARM}/subscriptions/{subscription_id}/resourceGroups/{resource_group}"
        f"/providers/Microsoft.DBforPostgreSQL/flexibleServers/{server}"
        f"/firewallRules/{RULE_NAME}?api-version={API_VERSION}"
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
    except urllib.error.HTTPError as error:
        raise CommandError(
            f"Azure refused the firewall update ({error.code}): {error.read().decode()[:400]}"
        ) from error
    return Admission(server=server, address=address, rule=RULE_NAME)
