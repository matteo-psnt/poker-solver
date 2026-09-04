"""The `record-admit` subcommand: put this machine's address in the record
server's firewall."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel

from src.interfaces.cloud import config as cloud_config
from src.interfaces.cloud import record_firewall
from src.interfaces.commands._base import Command

if TYPE_CHECKING:
    import argparse


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Flags for `poker-solver record-admit`."""
    parser.add_argument(
        "--address",
        default="",
        help="Admit this address instead of the one the internet reports for this machine.",
    )


class AdmittedPayload(BaseModel):
    """Which address the rule now names."""

    op: Literal["record-admit"] = "record-admit"
    server: str
    address: str


def run(args: argparse.Namespace) -> AdmittedPayload:
    """Learn the address, PUT the rule, report."""
    address = args.address or record_firewall.public_address()
    host = cloud_config.store_value("postgres_host")
    admission = record_firewall.admit(
        cloud_config.store_value("subscription_id"),
        cloud_config.store_value("resource_group"),
        host.split(".", 1)[0],
        address,
    )
    return AdmittedPayload(server=admission.server, address=admission.address)


def render(payload: AdmittedPayload) -> None:
    print(f"{payload.server}: firewall admits {payload.address}")
    print("  Azure applies the rule within a minute or two; a read that still times out is early.")


COMMAND = Command(
    name="record-admit",
    help="Admit this machine's current public IP to the record server's firewall.",
    add_arguments=add_arguments,
    run=run,
    render=render,
)
