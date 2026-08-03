"""Where the deployed infrastructure lives, asked of the thing that built it.

Terraform owns what exists, so Terraform is the authority on its coordinates.
This module is the single place that asks. Two states are read, and the split
is deliberate: ``infra/`` holds the Batch account and pool, ``infra/store/``
holds the durable share in its own resource group, so ``just destroy`` can tear
down compute without being able to reach the experiment record.

``terraform output -json`` is shelled rather than reading a generated outputs
file. The store's ``access_key`` is marked ``sensitive``; materialising it into
a second on-disk artifact would spread a secret to buy nothing -- the state
file already has it, and it is already gitignored.
"""

from __future__ import annotations

import functools
import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from src.interfaces.errors import CommandError

INFRA_DIR = Path("infra")
STORE_DIR = Path("infra/store")

_TERRAFORM_MISSING = (
    "terraform is not on PATH. The cloud commands read the deployed pool and "
    "share coordinates from Terraform state; install terraform, or run the "
    "operation from a checkout where `just create` has been applied."
)


class CloudConfigError(CommandError):
    """Infrastructure coordinates could not be read.

    Raised in preference to letting a ``FileNotFoundError`` on ``terraform``
    surface: the cloud commands carry an undeclared dependency on terraform and
    the az CLI credential, and a bare OSError gives no hint which is missing.

    A :class:`CommandError` because it is one: a checkout with no applied
    Terraform state is something the caller can fix, and every surface should be
    able to say so in the same clause it uses for every other readable failure.
    """


@functools.cache
def _outputs(chdir: str) -> dict[str, Any]:
    """Read one Terraform state's outputs, once per process.

    Cached because a single command can want several values from both states
    and ``terraform output`` is a slow way to ask twice. Keyed by ``str``
    rather than ``Path`` so the cache is hashable and stable.
    """
    if shutil.which("terraform") is None:
        raise CloudConfigError(_TERRAFORM_MISSING)
    try:
        completed = subprocess.run(
            ["terraform", f"-chdir={chdir}", "output", "-json"],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise CloudConfigError(
            f"`terraform -chdir={chdir} output -json` failed. Has `just create` been "
            f"applied against this checkout?\n{exc.stderr.strip()}"
        ) from exc
    parsed = json.loads(completed.stdout)
    if not parsed:
        raise CloudConfigError(
            f"Terraform state at {chdir} has no outputs -- nothing has been applied yet. "
            "Run `just store-create` then `just create`."
        )
    return parsed


def _value(chdir: str, name: str) -> str:
    """Pull one output's value, failing with the name that was missing."""
    entry = cast("dict[str, Any] | None", _outputs(chdir).get(name))
    if entry is None or "value" not in entry:
        raise CloudConfigError(
            f"Terraform state at {chdir} has no output named '{name}'. "
            "The infra and the code are out of step -- re-apply, or check infra/outputs.tf."
        )
    return str(entry["value"])


@dataclass(frozen=True)
class CloudConfig:
    """Everything the control plane needs to talk to Azure.

    ``batch_endpoint`` carries a scheme even though Terraform's
    ``account_endpoint`` does not. ``azurerm`` reports the bare host
    (``acct.region.batch.azure.com``) and ``BatchClient`` wants a URL, so the
    normalisation happens here rather than at each of the call sites that would
    otherwise each have to remember.
    """

    batch_endpoint: str
    pool_id: str
    storage_account: str
    share_name: str
    share_key: str
    autoscale_formula: str
    hourly_cost: str

    @classmethod
    def load(cls) -> CloudConfig:
        """Read both Terraform states and assemble the coordinates."""
        endpoint = _value(str(INFRA_DIR), "batch_account_endpoint")
        if not endpoint.startswith(("http://", "https://")):
            endpoint = f"https://{endpoint}"
        return cls(
            batch_endpoint=endpoint,
            pool_id=_value(str(INFRA_DIR), "pool_id"),
            storage_account=_value(str(STORE_DIR), "storage_account"),
            share_name=_value(str(STORE_DIR), "share_name"),
            share_key=_value(str(STORE_DIR), "access_key"),
            autoscale_formula=_value(str(INFRA_DIR), "autoscale_formula"),
            hourly_cost=_value(str(INFRA_DIR), "hourly_cost"),
        )
