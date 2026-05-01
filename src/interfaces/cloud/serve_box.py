"""Starting and stopping the blueprint host.

The box wakes on demand and turns itself off when nobody is using it, so
something has to be able to wake it again -- and that something is the console,
which means this has to be a normal Azure call rather than a thing you SSH in to
do.

**Deallocate, never just stop.** ``power_off`` leaves the VM allocated and
BILLING at the full rate while reporting itself as stopped; only
``deallocate`` releases the hardware. The distinction is invisible in the portal
until the invoice arrives, so this module does not expose the wrong one at all.

Waking is fast because nothing is re-fetched. The run and the card abstraction
live on a managed data disk that survives deallocation, so a start is a boot
plus a checkpoint load -- around two minutes -- rather than the ~1.6 GB of
small-file copying a fresh box pays.

``AzureCliCredential``, not ``DefaultAzureCredential``: the default chain probes
the link-local IMDS address, which on a laptop hangs rather than refusing
(measured: >120s vs 1.3s). Same rule as :mod:`src.interfaces.cloud.batch`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from azure.identity import AzureCliCredential
from azure.mgmt.compute import ComputeManagementClient

from src.interfaces.errors import CommandError

# Azure reports power state as an instance-view status code. These are the two
# that matter; anything else is a transition, and callers are told so by name
# rather than left to guess from a raw string.
_RUNNING = "PowerState/running"
_DEALLOCATED = "PowerState/deallocated"


@dataclass(frozen=True)
class BoxState:
    """What the box is doing, in the vocabulary the console renders.

    ``power`` is one of ``running``, ``deallocated``, ``starting``, ``stopping``
    or ``unknown``. The three transitional values exist because a UI that only
    knew the two stable ones would show "stopped" for the whole two minutes a
    box takes to wake, which reads as "the button did nothing".
    """

    name: str
    power: str
    location: str

    @property
    def usable(self) -> bool:
        return self.power == "running"


def _client(subscription_id: str) -> ComputeManagementClient:
    return ComputeManagementClient(AzureCliCredential(), subscription_id)


def _power_from(statuses: list[Any]) -> str:
    for status in statuses or []:
        code = getattr(status, "code", "") or ""
        if not code.startswith("PowerState/"):
            continue
        if code == _RUNNING:
            return "running"
        if code == _DEALLOCATED:
            return "deallocated"
        return code.removeprefix("PowerState/")
    return "unknown"


def status(subscription_id: str, resource_group: str, vm_name: str) -> BoxState:
    """Ask Azure what the box is doing right now.

    Raises :class:`CommandError` when the VM does not exist, which is the
    ordinary state before ``just serve-create`` has ever been run -- a caller
    should render that as "not provisioned", not as a fault.
    """
    client = _client(subscription_id)
    try:
        vm = client.virtual_machines.get(resource_group, vm_name, expand="instanceView")
    except Exception as error:
        raise CommandError(
            f"No blueprint host '{vm_name}' in '{resource_group}': {error}"
        ) from error
    view = getattr(vm, "instance_view", None)
    return BoxState(
        name=vm_name,
        power=_power_from(getattr(view, "statuses", []) if view else []),
        location=getattr(vm, "location", "") or "",
    )


def start(subscription_id: str, resource_group: str, vm_name: str, *, wait: bool = False) -> None:
    """Wake the box. Returns as soon as Azure accepts, unless ``wait``.

    Not waiting is the default because the caller is usually a web request and a
    two-minute hold is not a response. The console polls :func:`status` instead,
    which is also what lets it show the wake happening.
    """
    poller = _client(subscription_id).virtual_machines.begin_start(resource_group, vm_name)
    if wait:
        poller.result()


def deallocate(
    subscription_id: str, resource_group: str, vm_name: str, *, wait: bool = False
) -> None:
    """Release the hardware. This is the one that stops the bill."""
    poller = _client(subscription_id).virtual_machines.begin_deallocate(resource_group, vm_name)
    if wait:
        poller.result()
